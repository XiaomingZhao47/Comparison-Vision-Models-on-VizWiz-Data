import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


#############################################
#              MODEL DEFINITION
#############################################

class CLIPAnswerabilityClassifier(nn.Module):
    def __init__(
        self,
        base_model_name="openai/clip-vit-base-patch32",
        num_labels=2,
        freeze_mode="partial",          # "head_only" | "partial" | "full"
        num_unfrozen_vision_layers=4    # only used when freeze_mode == "partial"
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(base_model_name)
        self.num_labels = num_labels

        # get_image_features output dimension (usually 512 for ViT-B/32)
        embed_dim = self.clip.visual_projection.out_features
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_labels)
        )

        # --------- Freezing strategy ---------
        # First freeze all CLIP parameters
        for p in self.clip.parameters():
            p.requires_grad = False

        if freeze_mode == "head_only":
            msg = "👉 Freeze mode = head_only: only classifier head is trainable."

        elif freeze_mode == "partial":
            # Partially unfreeze: last N vision encoder layers + visual_projection
            vision_layers = list(self.clip.vision_model.encoder.layers)
            n_layers = len(vision_layers)
            start_idx = max(0, n_layers - num_unfrozen_vision_layers)

            for i in range(start_idx, n_layers):
                for p in vision_layers[i].parameters():
                    p.requires_grad = True
            for p in self.clip.visual_projection.parameters():
                p.requires_grad = True

            msg = (
                f"👉 Freeze mode = partial: unfreeze vision encoder last "
                f"{num_unfrozen_vision_layers} layers (layer {start_idx}~{n_layers-1}) "
                f"+ visual_projection + classifier."
            )

        elif freeze_mode == "full":
            # Fully unfreeze all CLIP parameters
            for p in self.clip.parameters():
                p.requires_grad = True
            msg = "👉 Freeze mode = full: CLIP backbone fully trainable."

        else:
            raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

        # Classifier head should always be trainable
        for p in self.classifier.parameters():
            p.requires_grad = True

        print(msg)

        # Print parameter stats
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        """
        pixel_values: [B, 3, H, W]
        returns: (loss, logits) if labels is not None
                 (None, logits) otherwise
        """
        # get_image_features → [B, D], already normalized
        image_embeds = self.clip.get_image_features(pixel_values=pixel_values)  # [B, D]

        logits = self.classifier(image_embeds)  # [B, num_labels]

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return loss, logits


#############################################
#              CONFIG
#############################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 4

PREPROCESSED_JSON = "data/preprocessed/preprocessed.json"
VAL_ANN_JSON = "data/Annotations/val.json"
MODEL_PATH = "clip_vizwiz_answerability_best_partial.pth"


#############################################
# 1. 从 val.json 构建 filename -> label 字典
#############################################

def build_label_dict(val_ann_path):
    with open(val_ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_dict = {}
    for item in data:
        filename = item["image"]
        label = int(item["answerable"])
        label_dict[filename] = label

    print(f"[Info] Loaded {len(label_dict)} labels from {val_ann_path}")
    return label_dict



#############################################
# 2. Dataset：一对原图 & 增强图
#############################################

class OriginalEnhancedDataset(Dataset):
    def __init__(self, preprocessed_json, label_dict, clip_processor):
        # 读取 preprocessed.json
        with open(preprocessed_json, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        self.label_dict = label_dict
        self.processor = clip_processor
        self.items = []

        skipped = 0
        for item in raw_items:
            orig_path = Path(str(item["image"]).replace("\\", "/"))
            filename = orig_path.name  # e.g. VizWiz_val_00000005.jpg

            if filename in self.label_dict:
                self.items.append(item)
            else:
                skipped += 1

        print(f"[Info] Loaded {len(self.items)} items from {preprocessed_json} with labels.")
        print(f"[Info] Skipped {skipped} items not found in {VAL_ANN_JSON}.")


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        orig_path = Path(str(item["image"]).replace("\\", "/"))
        enh_path  = Path(str(item["preprocessed_path"]).replace("\\", "/"))
        category  = item.get("category", "unknown")

        filename = orig_path.name

        # 这里不再需要 raise KeyError 了
        label = self.label_dict[filename]

        orig_img = Image.open(orig_path).convert("RGB")
        enh_img  = Image.open(enh_path).convert("RGB")

        orig_tensor = self.processor(images=orig_img, return_tensors="pt")["pixel_values"].squeeze(0)
        enh_tensor  = self.processor(images=enh_img,  return_tensors="pt")["pixel_values"].squeeze(0)

        return orig_tensor, enh_tensor, torch.tensor(label, dtype=torch.long), category



#############################################
# 3. 加载 finetuned 模型
#############################################

def load_model():
    model = CLIPAnswerabilityClassifier(
        base_model_name="openai/clip-vit-base-patch32",
        num_labels=2,
        freeze_mode="partial",
        num_unfrozen_vision_layers=8,
    )

    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[Info] Loaded model weights from {MODEL_PATH}")
    return model


#############################################
# 4. 评估原图 vs 增强图
#############################################

@torch.no_grad()
def evaluate(model, dataloader, device):
    total_orig = 0
    correct_orig = 0

    total_enh = 0
    correct_enh = 0

    # 按类别统计
    stats = {
        "dark":   {"orig_correct": 0, "orig_total": 0, "enh_correct": 0, "enh_total": 0},
        "bright": {"orig_correct": 0, "orig_total": 0, "enh_correct": 0, "enh_total": 0},
        "blur":   {"orig_correct": 0, "orig_total": 0, "enh_correct": 0, "enh_total": 0},
    }

    for orig_imgs, enh_imgs, labels, categories in tqdm(dataloader, desc="Evaluating"):
        orig_imgs = orig_imgs.to(device)
        enh_imgs  = enh_imgs.to(device)
        labels    = labels.to(device)

        # 原图预测
        _, logits_orig = model(orig_imgs)
        preds_orig = torch.argmax(logits_orig, dim=1)

        # 增强图预测
        _, logits_enh = model(enh_imgs)
        preds_enh = torch.argmax(logits_enh, dim=1)

        # overall
        batch_size = labels.size(0)
        total_orig += batch_size
        total_enh  += batch_size

        correct_orig += (preds_orig == labels).sum().item()
        correct_enh  += (preds_enh == labels).sum().item()

        # per-category
        for i, cat in enumerate(categories):
            cat = str(cat).lower()
            if cat not in stats:
                continue
            stats[cat]["orig_total"] += 1
            stats[cat]["enh_total"] += 1
            if preds_orig[i] == labels[i]:
                stats[cat]["orig_correct"] += 1
            if preds_enh[i] == labels[i]:
                stats[cat]["enh_correct"] += 1

    acc_orig = correct_orig / total_orig if total_orig > 0 else 0.0
    acc_enh  = correct_enh  / total_enh  if total_enh  > 0 else 0.0

    print("\n=== Overall ===")
    print(f"Origin   ACC: {acc_orig:.4f}")
    print(f"Enhanced ACC: {acc_enh:.4f}")
    print(f"Delta (enh - orig): {acc_enh - acc_orig:.4f}")

    print("\n=== Per Category ===")
    for cat, s in stats.items():
        if s["orig_total"] == 0:
            continue
        acc_o = s["orig_correct"] / s["orig_total"]
        acc_e = s["enh_correct"] / s["enh_total"] if s["enh_total"] > 0 else 0.0
        print(f"[{cat}] origin: {acc_o:.4f}, enhanced: {acc_e:.4f}, Δ = {acc_e - acc_o:.4f}")


#############################################
# 5. main
#############################################

def main():
    label_dict = build_label_dict(VAL_ANN_JSON)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    dataset = OriginalEnhancedDataset(
        preprocessed_json=PREPROCESSED_JSON,
        label_dict=label_dict,
        clip_processor=processor,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = load_model()

    evaluate(model, dataloader, DEVICE)


if __name__ == "__main__":
    main()
