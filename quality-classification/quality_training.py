"""
Image Quality Detection Utilities for VizWiz

Detects blur, darkness, poor contrast, and framing issues.
"""

import cv2
import numpy as np
from pathlib import Path


def detect_blur(image_path, threshold=100):
    """
    Detect if image is blurry using Laplacian variance.
    
    Args:
        image_path: Path to image
        threshold: Variance threshold (lower = more blurry)
        
    Returns:
        tuple: (is_blurry: bool, blur_score: float)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, 0.0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        is_blurry = variance < threshold
        return is_blurry, variance
    except:
        return False, 0.0


def detect_darkness(image_path, threshold=50):
    """
    Detect if image is too dark.
    
    Args:
        image_path: Path to image
        threshold: Brightness threshold (0-255, lower = darker)
        
    Returns:
        tuple: (is_dark: bool, brightness: float)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, 0.0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        is_dark = mean_brightness < threshold
        return is_dark, mean_brightness
    except:
        return False, 0.0


def detect_low_contrast(image_path, threshold=30):
    """
    Detect if image has low contrast.
    
    Args:
        image_path: Path to image
        threshold: Contrast threshold (std dev of pixel values)
        
    Returns:
        tuple: (is_low_contrast: bool, contrast_score: float)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, 0.0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        
        is_low_contrast = contrast < threshold
        return is_low_contrast, contrast
    except:
        return False, 0.0


def detect_poor_framing(image_path, edge_threshold=0.1):
    """
    Detect poor framing (object cut off at edges).
    
    Uses edge detection at image borders to identify cut-off objects.
    
    Args:
        image_path: Path to image
        edge_threshold: Ratio of edge pixels to total border pixels
        
    Returns:
        tuple: (is_poor_framing: bool, edge_ratio: float)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, 0.0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        
        h, w = edges.shape
        border_width = 10  # pixels
        
        top = edges[:border_width, :]
        bottom = edges[-border_width:, :]
        left = edges[:, :border_width]
        right = edges[:, -border_width:]
        
        border_edges = np.sum([
            np.sum(top > 0),
            np.sum(bottom > 0),
            np.sum(left > 0),
            np.sum(right > 0)
        ])
        
        total_border = 2 * border_width * (h + w)
        
        edge_ratio = border_edges / total_border if total_border > 0 else 0.0
        
        is_poor_framing = edge_ratio > edge_threshold
        return is_poor_framing, edge_ratio
        
    except:
        return False, 0.0


def detect_overexposure(image_path, threshold=200, ratio=0.3):
    """
    Detect if image is overexposed (too bright/washed out).
    
    Args:
        image_path: Path to image
        threshold: Brightness threshold for overexposure
        ratio: Minimum ratio of bright pixels
        
    Returns:
        tuple: (is_overexposed: bool, bright_ratio: float)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, 0.0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Count very bright pixels
        bright_pixels = np.sum(gray > threshold)
        total_pixels = gray.size
        
        bright_ratio = bright_pixels / total_pixels
        
        is_overexposed = bright_ratio > ratio
        return is_overexposed, bright_ratio
        
    except:
        return False, 0.0


def get_comprehensive_quality_label(image_path):
    """
    Get comprehensive quality assessment and assign label.
    
    Labels:
        0: Good quality
        1: Blurry
        2: Dark/Underexposed
        3: Poor framing
        4: Low contrast
        5: Overexposed
        6: Multiple issues (2+ problems)
        
    Args:
        image_path: Path to image
        
    Returns:
        tuple: (label: int, details: dict)
    """
    is_blurry, blur_score = detect_blur(image_path, threshold=100)
    is_dark, brightness = detect_darkness(image_path, threshold=50)
    is_low_contrast, contrast = detect_low_contrast(image_path, threshold=30)
    is_poor_frame, frame_score = detect_poor_framing(image_path, edge_threshold=0.15)
    is_overexposed, bright_ratio = detect_overexposure(image_path, threshold=200, ratio=0.3)
    
    issues = [is_blurry, is_dark, is_low_contrast, is_poor_frame, is_overexposed]
    num_issues = sum(issues)
    
    details = {
        'blur_score': blur_score,
        'brightness': brightness,
        'contrast': contrast,
        'frame_score': frame_score,
        'bright_ratio': bright_ratio,
        'is_blurry': is_blurry,
        'is_dark': is_dark,
        'is_low_contrast': is_low_contrast,
        'is_poor_frame': is_poor_frame,
        'is_overexposed': is_overexposed,
        'num_issues': num_issues
    }
    
    if num_issues >= 2:
        label = 6  # Multiple issues
    elif is_blurry:
        label = 1  # Blurry
    elif is_dark:
        label = 2  # Dark
    elif is_poor_frame:
        label = 3  # Poor framing
    elif is_low_contrast:
        label = 4  # Low contrast
    elif is_overexposed:
        label = 5  # Overexposed
    else:
        label = 0  # Good quality
    
    return label, details


def get_simple_quality_label(image_path):
    """
    Simplified 3-class quality assessment.
    
    Labels:
        0: Good quality
        1: Poor quality (single issue)
        2: Very poor quality (multiple issues)
        
    Args:
        image_path: Path to image
        
    Returns:
        tuple: (label: int, details: dict)
    """
    label, details = get_comprehensive_quality_label(image_path)
    
    if details['num_issues'] >= 2:
        simple_label = 2  # Very poor
    elif details['num_issues'] == 1:
        simple_label = 1  # Poor
    else:
        simple_label = 0  # Good
    
    return simple_label, details


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        print(f"Analyzing: {image_path}")
        print("="*60)
        
        label, details = get_comprehensive_quality_label(image_path)
        
        label_names = {
            0: "Good quality",
            1: "Blurry",
            2: "Dark/Underexposed",
            3: "Poor framing",
            4: "Low contrast",
            5: "Overexposed",
            6: "Multiple issues"
        }
        
        print(f"Label: {label} - {label_names[label]}")
        print(f"\nDetailed metrics:")
        print(f"  Blur score: {details['blur_score']:.2f} (blurry: {details['is_blurry']})")
        print(f"  Brightness: {details['brightness']:.2f} (dark: {details['is_dark']})")
        print(f"  Contrast: {details['contrast']:.2f} (low: {details['is_low_contrast']})")
        print(f"  Frame score: {details['frame_score']:.3f} (poor: {details['is_poor_frame']})")
        print(f"  Bright ratio: {details['bright_ratio']:.3f} (overexposed: {details['is_overexposed']})")
        print(f"  Total issues: {details['num_issues']}")
    else:
        print("Usage: python quality_detection.py <image_path>")