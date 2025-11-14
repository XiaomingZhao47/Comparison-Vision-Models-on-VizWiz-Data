import os
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


TARGET_SIZE = (512, 512)


# -------------------------------------------------
# 图像增强方法
# -------------------------------------------------

def gamma_correction(img, gamma):
    invGamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** invGamma * 255
    return cv2.LUT(img, table.astype("uint8"))


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def unsharp_mask(img, k=0.5):
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1 + k, blur, -k, 0)


# -------------------------------------------------
# 根据分类增强
# -------------------------------------------------

def preprocess_based_on_label(img, label):

    if label == "too_dark":
        img = gamma_correction(img, gamma=0.6)
        img = apply_clahe(img)
        return img

    elif label == "too_bright":
        img = cv2.convertScaleAbs(img, alpha=0.8, beta=-20)
        img = gamma_correction(img, gamma=1.4)
        return img

    elif label == "blurry":
        img = unsharp_mask(img, k=0.6)
        return img

    return img


# -------------------------------------------------
# 主处理流程
# -------------------------------------------------

def preprocess_all(json_path, output_dir):

    output_dir = Path(output_dir)
    dark_dir   = output_dir / "dark"
    bright_dir = output_dir / "bright"
    blurry_dir = output_dir / "blur"

    dark_dir.mkdir(parents=True, exist_ok=True)
    bright_dir.mkdir(parents=True, exist_ok=True)
    blurry_dir.mkdir(parents=True, exist_ok=True)

    # Load filtered JSON
    with open(json_path, "r") as f:
        items = json.load(f)

    results = []
    skipped = 0
    processed = 0

    for item in tqdm(items, desc="Preprocessing"):

        label = item.get("category", "none")

        # 跳过 none
        if label == "none":
            skipped += 1
            continue

        img_path = Path(item["image"])
        img_path = Path(str(img_path).replace("\\", "/"))

        if not img_path.exists():
            print(f"[Warning] Not found: {img_path}")
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)

            # 增强
            img = preprocess_based_on_label(img, label)

            # resize
            img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

            # 转为 BGR 保存
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 选择存储目录
            if label == "dark":
                save_path = dark_dir / img_path.name
            elif label == "bright":
                save_path = bright_dir / img_path.name
            elif label == "blur":
                save_path = blurry_dir / img_path.name
            else:
                skipped += 1
                continue

            cv2.imwrite(str(save_path), img_bgr)

            # 写入结果 JSON
            item["preprocessed_path"] = str(save_path)
            results.append(item)
            processed += 1

        except Exception as e:
            print(f"[Error] {img_path}: {e}")
            skipped += 1

    # 保存结果 JSON
    out_json = output_dir / "preprocessed.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    print("\n=== Done ===")
    print(f"Processed: {processed}")
    print(f"Skipped (including 'none'): {skipped}")
    print(f"Output JSON: {out_json}")


if __name__ == "__main__":
    preprocess_all(
        json_path="data/val_filtered.json",
        output_dir="data/preprocessed"
    )
