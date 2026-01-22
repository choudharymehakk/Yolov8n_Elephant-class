import os
import time
import csv
import cv2
import psutil
import subprocess
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_DIR = "../images"
OUTPUT_DIR = "../outputs"
ELEPHANT_CLASS_ID = 20
BENCHMARK_RUNS = 50
CSV_FILE = "../benchmark_results.csv"

# -----------------------------
# LOAD MODEL
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

if not image_list:
    print("No images found in images/")
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

    results[0].save(filename=os.path.join(OUTPUT_DIR, img_name))

print("Inference completed.\n")

# -----------------------------
# BENCHMARKING
# -----------------------------
print("===== RUNNING BENCHMARK =====")

benchmark_image = cv2.imread(os.path.join(IMAGE_DIR, image_list[0]))

# Warm-up
for _ in range(10):
    model(benchmark_image)

cpu_usages = []
gpu_usages = []

def get_gpu_usage():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        )
        return int(output.decode().strip())
    except Exception:
        return None

start = time.time()

for _ in range(BENCHMARK_RUNS):
    cpu_usages.append(psutil.cpu_percent(interval=None))

    gpu = get_gpu_usage()
    if gpu is not None:
        gpu_usages.append(gpu)

    model(benchmark_image)

end = time.time()

avg_time = (end - start) / BENCHMARK_RUNS
fps = 1 / avg_time
avg_cpu = sum(cpu_usages) / len(cpu_usages)
avg_gpu = sum(gpu_usages) / len(gpu_usages) if gpu_usages else "N/A"

# -----------------------------
# PRINT RESULTS
# -----------------------------
print("===== BENCHMARK RESULTS =====")
print(f"Model            : YOLOv8 Nano (COCO pretrained)")
print(f"Inference time   : {avg_time*1000:.2f} ms")
print(f"FPS              : {fps:.2f}")
print(f"Average CPU (%)  : {avg_cpu:.2f}")
print(f"Average GPU (%)  : {avg_gpu}")

# -----------------------------
# SAVE TO CSV
# -----------------------------
file_exists = os.path.isfile(CSV_FILE)

with open(CSV_FILE, mode="a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "Model",
            "Inference_time_ms",
            "FPS",
            "Avg_CPU_percent",
            "Avg_GPU_percent"
        ])

    writer.writerow([
        "YOLOv8n",
        round(avg_time * 1000, 2),
        round(fps, 2),
        round(avg_cpu, 2),
        avg_gpu
    ])

print(f"\nBenchmark results saved to {CSV_FILE}")
