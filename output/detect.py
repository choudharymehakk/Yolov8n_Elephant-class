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
