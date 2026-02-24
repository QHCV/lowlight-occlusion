import cv2
import numpy as np
import random
import os
import sys
import time
from multiprocessing import Pool, cpu_count, freeze_support

# ==========================================
#                 Config
# ==========================================
CONFIG = {
    # Input folder path
    "SOURCE_FOLDER": "splited_video_all629",
    # Supported video extensions
    "VIDEO_EXTENSIONS": (".mp4", ".avi", ".mov", ".mkv", ".flv"),
    # Number of worker processes
    # None => use all available CPU cores (recommended)
    # If the machine stutters, set a fixed number (e.g., 4)
    "NUM_PROCESSES": None,
    # --- Low-light params ---
    "LOW_LIGHT_FACTOR": 0.3,
    # --- Occlusion params ---
    "OCCLUSION_CHANGE_INTERVAL": 30,  # Change positions every N frames
    "OCCLUSION_MAX_COUNT": 3,  # Max number of blocks
    "OCCLUSION_MIN_SIZE": 50,  # Min block size
    "OCCLUSION_MAX_SIZE": 150,  # Max block size
    "OCCLUSION_COLOR": (51, 51, 51),  # Block color
}


# ==========================================
#              Core Processing
# ==========================================


def simulate_low_light(frame, factor):
    """
    Apply low-light effect.
    """
    # Option A: HSV-based scaling (most natural, but heavier)
    # Use float32 to reduce repeated clip/cast overhead
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    frame_hsv[:, :, 2] *= factor
    # Clamp and convert back to uint8
    frame_hsv[:, :, 2] = np.clip(frame_hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(frame_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def generate_random_rects(frame_shape, max_count, min_size, max_size):
    """
    Generate random occlusion rectangles.
    """
    h, w, _ = frame_shape
    rects = []
    count = random.randint(1, max_count)
    for _ in range(count):
        rect_w = random.randint(min_size, max_size)
        rect_h = random.randint(min_size, max_size)
        max_x = max(0, w - rect_w)
        max_y = max(0, h - rect_h)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        rects.append((x, y, rect_w, rect_h))
    return rects


def worker_task(args):
    """
    Worker function: process one video task.
    Args is a tuple: (src_path, dst_low_light, dst_occlusion)
    """
    src_path, dst_low, dst_occ = args

    # Guard exceptions so one bad file won't crash the pool
    try:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            return f"[Error] Cannot open: {src_path}"

        # Read video properties
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ensure output dirs exist (safe under multiprocessing)
        os.makedirs(os.path.dirname(dst_low), exist_ok=True)
        os.makedirs(os.path.dirname(dst_occ), exist_ok=True)

        # Create two writers
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_low = cv2.VideoWriter(dst_low, fourcc, fps, (w, h))
        out_occ = cv2.VideoWriter(dst_occ, fourcc, fps, (w, h))

        frame_idx = 0
        current_occlusions = []

        # Cache config values to reduce dict lookups
        occ_interval = CONFIG["OCCLUSION_CHANGE_INTERVAL"]
        occ_max_cnt = CONFIG["OCCLUSION_MAX_COUNT"]
        occ_min_sz = CONFIG["OCCLUSION_MIN_SIZE"]
        occ_max_sz = CONFIG["OCCLUSION_MAX_SIZE"]
        occ_color = CONFIG["OCCLUSION_COLOR"]
        low_factor = CONFIG["LOW_LIGHT_FACTOR"]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- 1) Low-light stream ---
            # This is relatively expensive
            frame_low = simulate_low_light(frame, low_factor)
            out_low.write(frame_low)

            # --- 2) Occlusion stream ---
            # Update or keep occlusion blocks
            if frame_idx % occ_interval == 0:
                current_occlusions = generate_random_rects(
                    frame.shape, occ_max_cnt, occ_min_sz, occ_max_sz
                )

            # Copy the frame before drawing (safer than in-place edits)
            frame_occ = frame.copy()
            for x, y, rw, rh in current_occlusions:
                cv2.rectangle(frame_occ, (x, y), (x + rw, y + rh), occ_color, -1)

            out_occ.write(frame_occ)
            frame_idx += 1

        # Release resources
        cap.release()
        out_low.release()
        out_occ.release()

        return f"[Success] {os.path.basename(src_path)} ({frame_idx} frames)"

    except Exception as e:
        return f"[Exception] {os.path.basename(src_path)}: {str(e)}"


# ==========================================
#               Main Control
# ==========================================


def batch_process_multiprocessing():
    source_root = CONFIG["SOURCE_FOLDER"]

    if not os.path.exists(source_root):
        print(f"Error: input folder not found: '{source_root}'")
        create_dummy_data(source_root)

    # Output root folders
    output_low_root = os.path.normpath(source_root + "_LowLight")
    output_occ_root = os.path.normpath(source_root + "_Occlusion")

    # 1) Scan tasks
    tasks = []
    print("Scanning files...")
    for root, dirs, files in os.walk(source_root):
        # Skip output folders
        if output_low_root in root or output_occ_root in root:
            continue

        for file in files:
            if file.lower().endswith(CONFIG["VIDEO_EXTENSIONS"]):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, source_root)

                dst_low = os.path.join(output_low_root, rel_path)
                dst_occ = os.path.join(output_occ_root, rel_path)

                tasks.append((src_path, dst_low, dst_occ))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No video files found.")
        return

    # 2) Configure pool
    # Use all cores if not specified
    cpu_cores = CONFIG["NUM_PROCESSES"] if CONFIG["NUM_PROCESSES"] else cpu_count()
    # Optionally leave one core for OS responsiveness
    # cpu_cores = max(1, cpu_cores - 1)

    print("-" * 60)
    print(f"Detected {cpu_count()} CPU cores")
    print(f"Starting {cpu_cores} worker processes for {total_tasks} videos")
    print("Note: high CPU usage during processing is normal")
    print("-" * 60)

    start_time = time.time()

    # 3) Run in parallel
    # Pool.imap_unordered streams results as they finish
    with Pool(processes=cpu_cores) as pool:
        # Better for progress reporting
        for i, result in enumerate(pool.imap_unordered(worker_task, tasks), 1):
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (total_tasks - i) * avg_time

            # Print a simple progress line
            print(f"[{i}/{total_tasks}] {result} - ETA: {remaining / 60:.1f} min")

    print("-" * 60)
    print(f"All tasks completed! Total time: {(time.time() - start_time) / 60:.1f} min")


def create_dummy_data(folder_name):
    """Generate dummy data for quick testing."""
    print("Input folder not found; creating dummy data...")
    os.makedirs(os.path.join(folder_name, "subdir"), exist_ok=True)
    video_path = os.path.join(folder_name, "subdir", "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 30, (640, 480))
    for i in range(150):
        img = np.full((480, 640, 3), 255, dtype=np.uint8)
        cv2.circle(img, (320, 240), 50, (0, 0, 255), -1)
        out.write(img)
    out.release()
    print(f"Created test video: {video_path}")
    print("Please re-run the script.")
    sys.exit()


if __name__ == "__main__":
    freeze_support()
    batch_process_multiprocessing()
