import os
import time
import csv
import cv2
import psutil
import subprocess
from ultralytics import YOLO

# =============================
# CONFIGURATION
# =============================
IMAGE_DIR = "../images"
VIDEO_DIR = "../videos"
OUTPUT_DIR = "../outputs"
CSV_FILE = "../benchmark_results.csv"

ELEPHANT_CLASS_ID = 20
IMAGE_BENCHMARK_RUNS = 50

# =============================
# LOAD MODEL (COCO PRETRAINED)
# =============================
model = YOLO("yolov8n.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# GPU USAGE FUNCTION
# =============================
def get_gpu_usage():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits"
            ],
            stderr=subprocess.DEVNULL
        )
        return int(output.decode().strip())
    except:
        return None

# =============================
# IMAGE INFERENCE
# =============================
print("\n===== IMAGE INFERENCE =====")

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

print("Image inference completed.")

# =============================
# IMAGE BENCHMARKING
# =============================
print("\n===== IMAGE BENCHMARK =====")

benchmark_image = cv2.imread(os.path.join(IMAGE_DIR, image_list[0]))

# Warm-up
for _ in range(10):
    model(benchmark_image)

cpu_usages = []
gpu_usages = []

start = time.time()

for _ in range(IMAGE_BENCHMARK_RUNS):
    cpu_usages.append(psutil.cpu_percent(interval=None))

    gpu = get_gpu_usage()
    if gpu is not None:
        gpu_usages.append(gpu)

    model(benchmark_image)

end = time.time()

avg_time = (end - start) / IMAGE_BENCHMARK_RUNS
fps_image = 1 / avg_time
avg_cpu_image = sum(cpu_usages) / len(cpu_usages)
avg_gpu_image = sum(gpu_usages) / len(gpu_usages) if gpu_usages else "N/A"

print("----- IMAGE BENCHMARK RESULTS -----")
print(f"Inference Time (ms): {avg_time*1000:.2f}")
print(f"FPS                : {fps_image:.2f}")
print(f"Avg CPU (%)        : {avg_cpu_image:.2f}")
print(f"Avg GPU (%)        : {avg_gpu_image}")

# =============================
# VIDEO BENCHMARKING (GPU VISIBLE)
# =============================
print("\n===== VIDEO INFERENCE & GPU BENCHMARK =====")

video_files = [
    v for v in os.listdir(VIDEO_DIR)
    if v.lower().endswith((".mp4", ".avi", ".mov"))
]

if not video_files:
    print("No video found in videos/")
    exit()

video_path = os.path.join(VIDEO_DIR, video_files[0])
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

frame_count = 0
cpu_usages_video = []
gpu_usages_video = []

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    model(frame)

    cpu_usages_video.append(psutil.cpu_percent(interval=None))
    gpu = get_gpu_usage()
    if gpu is not None:
        gpu_usages_video.append(gpu)

    frame_count += 1

cap.release()
end_time = time.time()

total_time = end_time - start_time
fps_video = frame_count / total_time
avg_cpu_video = sum(cpu_usages_video) / len(cpu_usages_video)
avg_gpu_video = sum(gpu_usages_video) / len(gpu_usages_video)

print("----- VIDEO BENCHMARK RESULTS -----")
print(f"Total Frames       : {frame_count}")
print(f"Total Time (s)     : {total_time:.2f}")
print(f"FPS                : {fps_video:.2f}")
print(f"Avg CPU (%)        : {avg_cpu_video:.2f}")
print(f"Avg GPU (%)        : {avg_gpu_video:.2f}")

# =============================
# SAVE RESULTS TO CSV
# =============================
file_exists = os.path.isfile(CSV_FILE)

with open(CSV_FILE, mode="a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "Mode",
            "Model",
            "Inference_time_ms",
            "FPS",
            "Avg_CPU_percent",
            "Avg_GPU_percent"
        ])

    # Image benchmark row
    writer.writerow([
        "Image",
        "YOLOv8n",
        round(avg_time * 1000, 2),
        round(fps_image, 2),
        round(avg_cpu_image, 2),
        avg_gpu_image
    ])

    # Video benchmark row
    writer.writerow([
        "Video",
        "YOLOv8n",
        "N/A",
        round(fps_video, 2),
        round(avg_cpu_video, 2),
        round(avg_gpu_video, 2)
    ])

print(f"\nBenchmark results saved to {CSV_FILE}")
