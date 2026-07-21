import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from src.shared import extract_features, crop_interaction_roi

# Map target AVA action IDs to clean text labels for ROI folders:
TARGET_ROI_MAP = {
    27: "drink",
    29: "eat",
    54: "smoke"
}

# Baseline AVA action IDs used to harvest "empty_hand" negative samples when wrist is near face:
BASELINE_ACTIONS = {11, 12}  # sit, stand


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


def harvest_crops(csv_list=None, video_dir="input/ava_kinetics/videos", output_dir="dataset/roi_train", max_per_class=500):
    """
    Parses AVA/Kinetics CSV spreadsheets, seeks to targeted timestamps, uses IoU to locate the correct person, 
    and saves cropped hand-to-face interaction patches into class-specific directories.
    :param csv_list: list of filepaths to the downloaded AVA-Kinetics annotation CSV files.
    :param video_dir: directory containing the downloaded video clips.
    :param output_dir: destination root directory to store the organized image crops.
    :param max_per_class: integer maximum number of saved crops allowed per action class.
    """

    # Default to standard spreadsheet list if none provided:
    if csv_list is None:
        csv_list = ["kinetics_train_v1.0.csv", "kinetics_val_v1.0.csv", "ava_train_v2.2.csv", "ava_val_v2.2.csv"]

    if not os.path.exists(video_dir):
        print(f"Error: Video directory '{video_dir}' not found.")
        return

    # Create target directories:
    os.makedirs(output_dir, exist_ok=True)
    all_target_classes = list(TARGET_ROI_MAP.values()) + ["empty_hand"]
    for class_name in all_target_classes:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    # Initialize tracking counts:
    counts = {cls: 0 for cls in all_target_classes}

    # Standardize 8-column schema:
    col_names = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"]
    df_list = []

    # Load spreadsheets:
    print(f"Loading and merging {len(csv_list)} annotation spreadsheets for ROI harvesting...")
    for csv_path in csv_list:
        if not os.path.exists(csv_path):
            print(f"Warning: File not found -> '{csv_path}'. Skipping...")
            continue
        print(f"  -> Parsing: {csv_path}")
        df_temp = pd.read_csv(csv_path, header=None, names=col_names, low_memory=False)
        df_temp = df_temp.dropna(subset=["action_id"])
        df_list.append(df_temp)

    if not df_list:
        print("Error: No valid CSV files were loaded. Exiting.")
        return

    # Combine spreadsheets and filter strictly for our target ROI + baseline classes:
    df = pd.concat(df_list, ignore_index=True)
    df["action_id"] = df["action_id"].astype(int)
    
    valid_ids = set(TARGET_ROI_MAP.keys()) | BASELINE_ACTIONS
    df_filtered = df[df["action_id"].isin(valid_ids)].copy()
    df_filtered.drop_duplicates(subset=["video_id", "timestamp", "person_id", "action_id"], inplace=True)

    # Group annotations by video_id to avoid reopening files:
    grouped_videos = df_filtered.groupby("video_id")
    print(f"Found {len(df_filtered)} relevant annotations across {len(grouped_videos)} unique videos.")

    # Initialize YOLO model:
    print("Initializing YOLOv11-Pose for keypoint extraction...")
    model = YOLO("yolo11n-pose.pt")
    valid_vid_exts = ('.mp4', '.webm', '.mkv', '.avi')

    # Process videos:
    for video_id, group in grouped_videos:
        
        # Stop completely if all quotas are met:
        if all(c >= max_per_class for c in counts.values()):
            print("\nReached maximum image quotas across all classes! Harvesting complete.")
            break

        # Locate physical video file on disk:
        video_path = None
        for ext in valid_vid_exts:
            candidate = os.path.join(video_dir, f"{video_id}{ext}")
            if os.path.exists(candidate):
                video_path = candidate
                break

        # Skip if video was not downloaded locally:
        if not video_path:
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30.0 if (fps <= 0 or np.isnan(fps)) else fps
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Iterate over targeted timestamp rows for this video:
        for _, row in group.iterrows():
            action_id = int(row["action_id"])
            
            # Map action_id to class string:
            if action_id in TARGET_ROI_MAP:
                target_cls = TARGET_ROI_MAP[action_id]
            else:
                target_cls = "empty_hand"

            # Skip if this specific class already reached its quota:
            if counts[target_cls] >= max_per_class:
                continue

            # Convert normalized ground-truth box to pixel coordinates:
            gt_box = [row["x1"] * width, row["y1"] * height, row["x2"] * width, row["y2"] * height]

            # Seek directly to the exact annotated frame:
            target_frame_idx = int(float(row["timestamp"]) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)

            ret, frame = cap.read()
            if not ret:
                continue

            # Run pose detection on the single targeted frame:
            results = model(source=frame, show=False, verbose=False)

            best_iou = 0.0
            best_kpts = None
            best_box = None

            # Find the detected skeleton that best matches the AVA ground-truth box:
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                keypoints_batch = results[0].keypoints.xy.cpu().numpy()
                bboxes_batch = results[0].boxes.xyxy.cpu().numpy()

                for idx, det_box in enumerate(bboxes_batch):
                    iou = compute_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_kpts = keypoints_batch[idx]
                        best_box = det_box

            # If IoU match is strong (> 0.15), evaluate interaction geometry:
            if best_iou > 0.15 and best_kpts is not None:
                feats = extract_features(best_kpts, best_box)
                lw_to_nose = feats[34]
                rw_to_nose = feats[35]

                # Is a wrist within the anatomical interaction zone:
                if (0.0 < lw_to_nose < 1.5) or (0.0 < rw_to_nose < 1.5):
                    roi_patch = crop_interaction_roi(frame, best_kpts, best_box, padding=40)
                    
                    if roi_patch is not None and roi_patch.size > 0:

                        # Resize to standard vision model dimensions:
                        crop_resized = cv2.resize(roi_patch, (224, 224))
                        
                        # Save patch and increment accurate counter:
                        save_path = os.path.join(output_dir, target_cls, f"{video_id}_{target_frame_idx}_{counts[target_cls]}.jpg")
                        cv2.imwrite(save_path, crop_resized)
                        counts[target_cls] += 1
                        
                        print(f"[{target_cls.upper():<10} | {counts[target_cls]}/{max_per_class}] Harvested crop -> {save_path}")

        cap.release()

    print("\n" + "="*50)
    print("ROI Harvesting Complete! Final Class Distribution:")
    for cls, count in counts.items():
        print(f"  -> {cls.upper():<12}: {count} images")
    print("="*50)


if __name__ == "__main__":
    harvest_crops(
        csv_list=["kinetics_train_v1.0.csv", "kinetics_val_v1.0.csv", "ava_train_v2.2.csv", "ava_val_v2.2.csv"],
        video_dir="input/ava_kinetics/videos",
        output_dir="dataset/roi_train",
        max_per_class=500
    )
