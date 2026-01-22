import os
import cv2
from ultralytics import YOLO

# Load COCO-pretrained YOLOv8 nano
model = YOLO("yolov8n.pt")

IMAGE_DIR = "../images"
OUTPUT_DIR = "../outputs"
ELEPHANT_CLASS_ID = 20  # COCO class ID

os.makedirs(OUTPUT_DIR, exist_ok=True)

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    results = model(img)

    elephant_found = False

    for box in results[0].boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)

        if cls_id == ELEPHANT_CLASS_ID:
            elephant_found = True
            print(f"[{img_name}] Elephant detected | confidence: {conf:.2f}")

    output_path = os.path.join(OUTPUT_DIR, img_name)
    results[0].save(filename=output_path)

    if not elephant_found:
        print(f"[{img_name}] No elephant detected")

print("Inference completed successfully.")
