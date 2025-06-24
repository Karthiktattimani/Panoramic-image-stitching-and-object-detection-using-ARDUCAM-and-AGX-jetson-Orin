import cv2
import sys
import argparse
import os

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Could not load image at {path}")
    return img

def resize_image(img, max_dim=1600):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def check_features(img, detector):
    keypoints = detector.detect(img)
    return len(keypoints)

# Argument parsing
parser = argparse.ArgumentParser(description="Stitch images together.")
parser.add_argument('--iteration', type=int, required=True, help='Iteration number (1 or 2)')
args = parser.parse_args()
iteration = args.iteration

print(f"🔄 Stitching images for iteration {iteration}...")

# File paths
capture_dir = "/home/nvidia/team16/captured_images"
stitched_dir = "/home/nvidia/team16/stitched_images"
os.makedirs(stitched_dir, exist_ok=True)

path1 = os.path.join(capture_dir, f"capture_0_{iteration}.jpg")
path2 = os.path.join(capture_dir, f"capture_1_{iteration}.jpg")

# Load and optionally resize
img1 = load_image(path1)
img2 = load_image(path2)
if img1 is None or img2 is None:
    sys.exit(1)

img1 = resize_image(img1)
img2 = resize_image(img2)

# Check features (optional but helps debugging)
orb = cv2.ORB_create()
features1 = check_features(img1, orb)
features2 = check_features(img2, orb)
print(f"🔍 Features detected - Image 1: {features1}, Image 2: {features2}")

if features1 < 20 or features2 < 20:
    print("⚠️ Not enough features detected in one or both images. Stitching may fail.")

# Create stitcher
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

# Stitch
status, stitched = stitcher.stitch([img1, img2])

# Retry with SCANS mode if PANORAMA fails
if status != cv2.Stitcher_OK:
    print(f"⚠️ PANORAMA stitching failed with status {status}, retrying with SCANS mode...")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, stitched = stitcher.stitch([img1, img2])

# Save or report failure
if status == cv2.Stitcher_OK:
    out_path = os.path.join(stitched_dir, f"stitched_image_{iteration}.jpg")
    cv2.imwrite(out_path, stitched)
    print(f"✅ Successfully saved stitched image to '{out_path}'")
else:
    print(f"❌ Stitching failed with status code: {status}.")
    sys.exit(1)

