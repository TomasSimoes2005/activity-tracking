import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import cv2
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from src.shared import WINDOW_SIZE, NUM_FEATURES, extract_features

# Selected YOLO model:
YOLO_MODEL = "yolo11n-pose.pt"

# Set concurrency level:
MAX_WORKERS = 6

# Map AVA Integer IDs to clean text string labels for pipeline:
TARGET_AVA_CLASSES = {
    5: "fall_floor",
    8: "lie_sleep",
    11: "sit",
    12: "stand",
    27: "drink",
    29: "eat",
    54: "smoke",
    57: "cellphone"
}


def compute_iou(box_a, box_b):
    """
    Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    :param box_a: first bounding box array [x1, y1, x2, y2].
    :param box_b: second bounding box array [x1, y1, x2, y2].
    :return: float IoU value between 0.0 and 1.0.
    """

    # Determine intersection rectangle coordinates:
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    # Calculate intersection area:
    intersection_area = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    if intersection_area == 0.0:
        return 0.0

    # Calculate union area:
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = float(box_a_area + box_b_area - intersection_area)

    return intersection_area / union_area if union_area > 0.0 else 0.0


def _worker_process_subset(worker_id, rows_subset, video_dir, temp_csv_path):
    """
    Worker function executed by each thread. Processes an assigned subset of AVA CSV annotation rows, extracts IoU-anchored skeletons from the video clips, and writes to an isolated CSV file.
    :param worker_id: integer ID of the thread worker.
    :param rows_subset: list of tuples representing valid AVA CSV rows.
    :param video_dir: root directory containing the downloaded AVA-Kinetics video files.
    :param temp_csv_path: path to the worker's isolated output CSV file.
    """

    # Disable verbose YOLO logging for this thread:
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # Initialize thread-local YOLO model:
    print(f"[Worker {worker_id}] Initializing {YOLO_MODEL} (Assigned {len(rows_subset)} action sequences)...")
    model = YOLO(YOLO_MODEL)

    # Valid video formats downloaded by yt-dlp:
    valid_vid_exts = ('.mp4', '.webm', '.mkv', '.avi')

    # Initialize temporary CSV file with header if first write:
    with open(temp_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["label"]
        for f_idx in range(WINDOW_SIZE):
            for k_idx in range(17):
                header.extend([f"f{f_idx}_p{k_idx}_x", f"f{f_idx}_p{k_idx}_y"])
            header.extend([
                f"f{f_idx}_lw_dist", f"f{f_idx}_rw_dist",
                f"f{f_idx}_lw_elev", f"f{f_idx}_rw_elev",
                f"f{f_idx}_prof_ratio", f"f{f_idx}_head_tilt",
                f"f{f_idx}_inter_wrist", f"f{f_idx}_chest_dist",
                f"f{f_idx}_lw_lear", f"f{f_idx}_rw_rear",
                f"f{f_idx}_lw_rear", f"f{f_idx}_rw_lear"
            ])
        writer.writerow(header)

    # Process each assigned AVA annotation row:
    for video_id, timestamp, x1_norm, y1_norm, x2_norm, y2_norm, action_id in rows_subset:

        # Map integer action ID to clean string label:
        label = TARGET_AVA_CLASSES.get(int(action_id))
        if not label:
            continue

        # Locate the corresponding video file across all supported container formats:
        video_path = None
        for ext in valid_vid_exts:
            candidate = os.path.join(video_dir, f"{video_id}{ext}")
            if os.path.exists(candidate):
                video_path = candidate
                break

        # If video file is not downloaded locally, skip:
        if not video_path:
            continue

        # Open video capture:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        # Get hardware properties:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30.0 if (fps <= 0 or np.isnan(fps)) else fps
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Convert normalized AVA ground-truth box to pixel coordinates:
        gt_box = [x1_norm * width, y1_norm * height, x2_norm * width, y2_norm * height]

        # Calculate frame window boundaries centered on the AVA timestamp:
        center_frame = int(float(timestamp) * fps)
        start_frame = max(0, center_frame - (WINDOW_SIZE // 2))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        feature_buffer = []

        # Read 30 continuous frames:
        for _ in range(WINDOW_SIZE):
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv11 pose detection (tracking not needed due to IoU anchoring):
            results = model(source=frame, show=False, verbose=False)

            best_iou = 0.0
            best_feats = None

            # Find the detected skeleton that best matches the AVA ground-truth box:
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                keypoints_batch = results[0].keypoints.xy.cpu().numpy()
                bboxes_batch = results[0].boxes.xyxy.cpu().numpy()

                for idx, det_box in enumerate(bboxes_batch):
                    iou = compute_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_feats = extract_features(keypoints_batch[idx], det_box)

            # If IoU match is strong (> 0.15), append features to buffer:
            if best_iou > 0.15 and best_feats is not None:
                feature_buffer.append(best_feats)
            else:

                # If person occluded/lost, apply Edge Padding (repeat last known posture):
                if len(feature_buffer) > 0:
                    feature_buffer.append(feature_buffer[-1])
                else:
                    feature_buffer.append([0.0] * NUM_FEATURES)

        cap.release()

        # If sequence has at least 10 valid frames, apply final Edge Padding up to WINDOW_SIZE and save:
        if len(feature_buffer) >= 10:
            while len(feature_buffer) < WINDOW_SIZE:
                feature_buffer.append(feature_buffer[-1])

            # Flatten 30x42 matrix into a single 1260-element row and write to CSV:
            sequence_row = [label] + [val for frame_feats in feature_buffer[:WINDOW_SIZE] for val in frame_feats]
            with open(temp_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(sequence_row)

    print(f"[Worker {worker_id}] Completed processing assigned subset!")


def process_ava_kinetics(csv_list, video_dir="input/ava_kinetics/videos", output_csv="output/ava_dataset.csv"):
    """
    Parses multiple AVA and Kinetics CSVs, standardizes their schemas, dispatches valid rows across worker threads to extract IoU-anchored skeleton sequences, and merges them into a master dataset.
    :param csv_list: list of filepaths to the downloaded AVA-Kinetics annotation CSV files.
    :param video_dir: directory containing the downloaded video clips.
    :param output_csv: path to save the final merged dataset CSV.
    """

    if not os.path.exists(video_dir):
        print(f"Error: Video directory '{video_dir}' not found.")
        return

    col_names = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"]
    df_list = []

    # Load spreadsheets:
    print(f"Loading and merging {len(csv_list)} annotation spreadsheets for ingestion...")
    for csv_path in csv_list:
        if not os.path.exists(csv_path):
            print(f"Warning: File not found -> '{csv_path}'. Skipping...")
            continue
        print(f"  -> Parsing: {csv_path}")

        # Standardize 7-col (Kinetics) and 8-col (AVA) schemas automatically:
        df_temp = pd.read_csv(csv_path, header=None, names=col_names, low_memory=False)

        # Drop empty frame rows where action_id is missing:
        df_temp = df_temp.dropna(subset=["action_id"])
        df_list.append(df_temp)
    if not df_list:
        print("Error: No valid CSV files were loaded. Exiting.")
        return

    # Combine all spreadsheets into a single master dataframe:
    df = pd.concat(df_list, ignore_index=True)
    df["action_id"] = df["action_id"].astype(int)

    # Filter strictly for our 10 target classes:
    valid_ids = list(TARGET_AVA_CLASSES.keys())
    df_filtered = df[df["action_id"].isin(valid_ids)].copy()

    # Remove duplicate timestamps for the same person/video across overlapping spreadsheets:
    df_filtered.drop_duplicates(subset=["video_id", "timestamp", "person_id", "action_id"], inplace=True)
    total_rows = len(df_filtered)
    print(f"Found {total_rows} total annotations matching 10 target classes across all spreadsheets")
    if total_rows == 0:
        return

    # Convert DataFrame to a list of tuples for thread slicing:
    rows_list = df_filtered[["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id"]].values.tolist()

    # Partition rows across workers:
    chunk_size = (total_rows + MAX_WORKERS - 1) // MAX_WORKERS
    chunks = [rows_list[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]
    temp_csv_files = [f"{output_csv}.part{i}" for i in range(len(chunks))]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    print(f"Launching parallel extraction across {MAX_WORKERS} workers...")

    # Launch thread pool:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(_worker_process_subset, i, chunk, video_dir, temp_csv_files[i])
            for i, chunk in enumerate(chunks)
        ]
        for future in futures:
            future.result()

    # Merge temporary part files into master CSV:
    print("\nAll workers finished! Merging temporary thread files into master dataset...")
    with open(output_csv, mode='w', newline='') as master_f:
        header_written = False
        for temp_file in temp_csv_files:
            if not os.path.exists(temp_file):
                continue
            with open(temp_file, mode='r', newline='') as part_f:
                lines = part_f.readlines()
                if not lines:
                    continue
                if not header_written:
                    master_f.write(lines[0])
                    header_written = True
                for line in lines[1:]:
                    master_f.write(line)
            os.remove(temp_file)

    print(f"AVA-Kinetics ingestion complete! Unified dataset successfully saved to: {output_csv}")


if __name__ == "__main__":
    process_ava_kinetics(
        csv_list=["kinetics_train_v1.0.csv", "kinetics_val_v1.0.csv", "ava_train_v2.2.csv", "ava_val_v2.2.csv"],
        video_dir="input/ava_kinetics/videos",
        output_csv="output/ava_dataset.csv"
    )