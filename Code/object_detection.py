import os
import glob
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

print("🔍 Performing YOLOv10 object detection on the latest stitched image...")

# Paths
stitched_dir = '/home/nvidia/team16/stitched_images'
final_dir = '/home/nvidia/team16/final_images'

# Classes to detect
target_classes = {"person", "chair", "car", "bus", "truck", "tv", "laptop", "cell phone"}

# Find the latest stitched image
stitched_images = glob.glob(os.path.join(stitched_dir, 'stitched_image_*.jpg'))
if not stitched_images:
    raise Exception("❌ No stitched images found!")

latest_stitched_image = max(stitched_images, key=os.path.getctime)
print(f"🖼️ Found latest stitched image: {latest_stitched_image}")

# Output filename
timestamp = os.path.basename(latest_stitched_image).split('_')[-1].split('.')[0]
final_filename = os.path.join(final_dir, f'final_image_{timestamp}.jpg')

# Ensure output directory exists
os.makedirs(final_dir, exist_ok=True)

# Load YOLOv10 model
model = YOLO('yolov10s.pt')  # You can use 'yolov10m.pt', 'yolov10l.pt', etc.

# Show available class labels (debug)
print("📋 Model class names:")
print(model.names)

# Run inference
results = model(latest_stitched_image)

# Load image to draw
img = Image.open(latest_stitched_image).convert("RGB")
draw = ImageDraw.Draw(img)
font = ImageFont.load_default()
line_width = 5
text_size_multiplier = 2

# Parse detections
for result in results:
    print(f"📦 Total detections: {len(result.boxes)}")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.25:  # Confidence threshold
            continue

        label = model.names[cls_id]
        if label not in target_classes:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0])
        label_text = f"{label} {conf:.2f}"
        print(f"🔍 Detected: {label_text} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)

        # Draw label and confidence
        text_size = font.getsize(label_text)
        text_bg = [x1, y1 - text_size[1] * text_size_multiplier, x1 + text_size[0] * text_size_multiplier, y1]
        draw.rectangle(text_bg, fill="red")
        draw.text((x1, y1 - text_size[1] * text_size_multiplier), label_text, fill="white", font=font)

# Save final annotated image
img.save(final_filename)
print(f"✅ Detection complete! Saved final image to {final_filename}")

