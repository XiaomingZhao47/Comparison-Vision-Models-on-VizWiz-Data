import cv2
import os
import json
from tqdm import tqdm


#########################################
#           CONFIGURATION
#########################################

VAL_DIR = "data/val"      # path to VizWiz val/ images
OUTPUT_JSON = "data/val_filtered.json"

BLUR_THRESHOLD = 30       # variance of Laplacian 10%  ≈ 25 20%  ≈ 53 30%  ≈ 89
BRIGHT_THRESHOLD = 159    # L channel mean
DARK_THRESHOLD = 76      # L channel mean
# dark threshold candidate = 10% ≈ 75.8
# bright threshold candidate = 90% ≈ 159.1

#########################################
#           BRISQUE LOADING
#########################################

def load_brisque_model():
    model_path = "models/brisque_model_live.yml"
    range_path = "models/brisque_range_live.yml"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}")
    if not os.path.exists(range_path):
        raise FileNotFoundError(f"Missing {range_path}")

    print("Loading BRISQUE model...")
    return cv2.quality_QualityBRISQUE(model_path, range_path)


#########################################
#           QUALITY METRICS
#########################################

def compute_brisque(img, brisque):
    # Ensure uint8 3-channel image
    if img.dtype != "uint8":
        img = img.astype("uint8")

    # Fix alpha channel if present
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Ensure image is large enough
    if img.shape[0] < 32 or img.shape[1] < 32:
        return None

    try:
        score = brisque.compute(img)
        return float(score[0])
    except Exception:
        return None  # skip invalid images

def compute_blur(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def compute_brightness(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    return L.mean()


#########################################
#           FILTERING LOGIC
#########################################


def filter_image(img_path, brisque):
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Compute BRISQUE safely
    brisque_score = compute_brisque(img, brisque)
    
    # If BRISQUE failed, still compute blur/brightness
    blur_score = compute_blur(img)
    brightness = compute_brightness(img)

    # Determine category
    if blur_score < BLUR_THRESHOLD:
        category = "blur"
    elif brightness > BRIGHT_THRESHOLD:
        category = "bright"
    elif brightness < DARK_THRESHOLD:
        category = "dark"
    else:
        category = "none"

    return {
        "image": img_path,
        "brisque": brisque_score,
        "blur": blur_score,
        "brightness": brightness,
        "category": category,
    }

#########################################
#           MAIN PROCESSING
#########################################

def process_val_images():
    brisque_model = load_brisque_model()

    image_files = [
        f for f in os.listdir(VAL_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    results = []

    print(f"Processing {len(image_files)} images...")

    for fname in tqdm(image_files):
        img_path = os.path.join(VAL_DIR, fname)
        result = filter_image(img_path, brisque_model)
        if result is not None:
            results.append(result)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print("Done! Saved JSON to:", OUTPUT_JSON)


#########################################
#               RUN
#########################################

if __name__ == "__main__":
    process_val_images()
