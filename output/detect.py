import os
import time
import cv2
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_DIR = "../images"
OUTPUT_DIR = "../outputs"
ELEPHANT_CLASS_ID = 20
BENCHMARK_RUNS = 50

# -----------------------------
# LOAD MODEL (COCO pretrained)
# -----------------------------
model = YOLO("yolov8n.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# NORMAL INFERENCE
# -----------------------------
print("===== RUNNING INFERENCE =====")

image_list = [
    img for img in os.listdir(IMAGE_DIR)
    if img.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(image_list) == 0:
    print("No images found in images/ folder")
    exit()

for img_name in image_list:
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    results = model(img)

    elephant_found = False
    non_wild_found = False

    for box in results[0].boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)

        if cls_id == ELEPHANT_CLASS_ID:
            elephant_found = True
            print(f"[{img_name}] ELEPHANT | confidence: {conf:.2f}")
        else:
            non_wild_found = True

    if non_wild_found:
        print(f"[{img_name}] NON-WILD object detected")

    output_path = os.path.join(OUTPUT_DIR, img_name)
    results[0].save(filename=output_path)

print("Inference completed.\n")

# -----------------------------
# BENCHMARKING
# -----------------------------
print("===== RUNNING BENCHMARK =====")

# Use first image for benchmarking
benchmark_image = cv2.imread(
    os.path.join(IMAGE_DIR, image_list[0])
)

# Warm-up (VERY IMPORTANT)
for _ in range(10):
    model(benchmark_image)

start = time.time()

for _ in range(BENCHMARK_RUNS):
    model(benchmark_image)

end = time.time()

avg_time = (end - start) / BENCHMARK_RUNS
fps = 1 / avg_time

print("===== BENCHMARK RESULTS =====")
print(f"Device           : Cloud Edge")
print(f"Model            : YOLOv8 Nano (COCO pretrained)")
print(f"Image size       : {benchmark_image.shape[1]}x{benchmark_image.shape[0]}")
print(f"Inference time   : {avg_time*1000:.2f} ms")
print(f"FPS              : {fps:.2f}")
