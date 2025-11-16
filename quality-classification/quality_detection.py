"""
VizWiz Quality Classification Dataset

Dataset for training ViT to classify image quality issues.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import json
from collections import Counter
from tqdm.auto import tqdm

import sys
sys.path.append('/mnt/user-data/outputs')
from quality_detection import get_comprehensive_quality_label, get_simple_quality_label


class VizWizQualityDataset(Dataset):
    """
    Dataset for image quality classification.
    
    Labels:
        0: Good quality
        1: Blurry
        2: Dark/Underexposed
        3: Poor framing
        4: Low contrast
        5: Overexposed
        6: Multiple issues
    """
    
    def __init__(
        self,
        annotations_dict,
        images_dir,
        feature_extractor,
        max_samples=None,
        use_simple_labels=False,
        cache_labels=True,
        label_cache_file=None
    ):
        """
        Args:
            annotations_dict: VizWiz annotations dictionary
            images_dir: Directory containing images
            feature_extractor: ViT feature extractor
            max_samples: Maximum number of samples to use
            use_simple_labels: If True, use 3-class (good/poor/very poor)
            cache_labels: Cache quality labels to avoid recomputation
            label_cache_file: Path to save/load cached labels
        """
        self.annotations = annotations_dict['annotations']
        if max_samples:
            self.annotations = self.annotations[:max_samples]
        
        self.images = {img['id']: img for img in annotations_dict['images']}
        self.images_dir = Path(images_dir)
        self.feature_extractor = feature_extractor
        self.use_simple_labels = use_simple_labels
        
        # Generate or load quality labels
        if label_cache_file and Path(label_cache_file).exists():
            print(f"Loading cached labels from {label_cache_file}...")
            import pickle
            with open(label_cache_file, 'rb') as f:
                self.quality_labels, self.quality_details = pickle.load(f)
            print(f"✅ Loaded {len(self.quality_labels)} cached labels")
        else:
            print("Generating quality labels...")
            self.quality_labels, self.quality_details = self._generate_quality_labels()
            
            if cache_labels and label_cache_file:
                import pickle
                with open(label_cache_file, 'wb') as f:
                    pickle.dump((self.quality_labels, self.quality_details), f)
                print(f"✅ Cached labels saved to {label_cache_file}")
        
        self._print_distribution()
    
    def _generate_quality_labels(self):
        """Generate quality labels for all images."""
        labels = []
        details_list = []
        
        for ann in tqdm(self.annotations, desc="Analyzing image quality"):
            image_info = self.images[ann['image_id']]
            img_path = self.images_dir / image_info['file_name']
            
            if not img_path.exists():
                labels.append(0)
                details_list.append({})
                continue
            
            if self.use_simple_labels:
                label, details = get_simple_quality_label(img_path)
            else:
                label, details = get_comprehensive_quality_label(img_path)
            
            labels.append(label)
            details_list.append(details)
        
        return labels, details_list
    
    def _print_distribution(self):
        """Print label distribution."""
        dist = Counter(self.quality_labels)
        
        if self.use_simple_labels:
            label_names = {
                0: "Good quality",
                1: "Poor quality",
                2: "Very poor quality"
            }
        else:
            label_names = {
                0: "Good quality",
                1: "Blurry",
                2: "Dark/Underexposed",
                3: "Poor framing",
                4: "Low contrast",
                5: "Overexposed",
                6: "Multiple issues"
            }
        
        print(f"\n{'='*60}")
        print("Quality Distribution:")
        print(f"{'='*60}")
        for label in sorted(dist.keys()):
            count = dist[label]
            percentage = count / len(self.quality_labels) * 100
            name = label_names.get(label, f"Unknown-{label}")
            print(f"  {label}: {name:20s} - {count:4d} ({percentage:5.1f}%)")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_info = self.images[ann['image_id']]
        image_path = self.images_dir / image_info['file_name']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='gray')
        
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        label = self.quality_labels[idx]
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def get_quality_stats(self):
        """Get statistics about quality issues."""
        stats = {
            'total': len(self.quality_labels),
            'distribution': Counter(self.quality_labels),
            'avg_blur_score': 0.0,
            'avg_brightness': 0.0,
            'avg_contrast': 0.0,
        }
        
        if self.quality_details:
            blur_scores = [d.get('blur_score', 0) for d in self.quality_details if d]
            brightness = [d.get('brightness', 0) for d in self.quality_details if d]
            contrast = [d.get('contrast', 0) for d in self.quality_details if d]
            
            if blur_scores:
                stats['avg_blur_score'] = sum(blur_scores) / len(blur_scores)
            if brightness:
                stats['avg_brightness'] = sum(brightness) / len(brightness)
            if contrast:
                stats['avg_contrast'] = sum(contrast) / len(contrast)
        
        return stats


def create_quality_dataset(
    split='train',
    max_samples=1000,
    use_simple_labels=False,
    annotations_path='data/annotations',
    images_path='data'
):
    """
    Quick helper to create quality dataset.
    
    Args:
        split: 'train' or 'val'
        max_samples: Max samples to use
        use_simple_labels: Use 3-class instead of 7-class
        annotations_path: Path to annotations
        images_path: Path to images
        
    Returns:
        VizWizQualityDataset
    """
    from transformers import ViTImageProcessor
    
    with open(f'{annotations_path}/{split}.json', 'r') as f:
        data = json.load(f)
    
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    cache_file = f'quality_labels_{split}_{max_samples}{"_simple" if use_simple_labels else ""}.pkl'
    
    dataset = VizWizQualityDataset(
        data,
        f'{images_path}/{split}',
        feature_extractor,
        max_samples=max_samples,
        use_simple_labels=use_simple_labels,
        cache_labels=True,
        label_cache_file=cache_file
    )
    
    return dataset


if __name__ == "__main__":
    print("Testing VizWizQualityDataset...")
    
    dataset = create_quality_dataset(
        split='val',
        max_samples=500,
        use_simple_labels=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Pixel values shape: {sample['pixel_values'].shape}")
    print(f"  Label: {sample['labels'].item()}")
    
    # get stats
    stats = dataset.get_quality_stats()
    print(f"\nQuality Stats:")
    print(f"  Total: {stats['total']}")
    print(f"  Avg blur score: {stats['avg_blur_score']:.2f}")
    print(f"  Avg brightness: {stats['avg_brightness']:.2f}")
    print(f"  Avg contrast: {stats['avg_contrast']:.2f}")