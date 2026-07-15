import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time
import random
import logging
import threading
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from src.shared import FrameBuffer
from src.server import run_server
from src.action_predictor import ActionPredictor

# Selected YOLO model:
YOLO_MODEL = "yolo11n-pose.pt"

# Map AVA Integer IDs to clean text string labels (Matches ingest_ava_kinetics.py):
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


def gather_random_samples_from_csv(csv_list, video_dir="input/ava_kinetics/videos", samples_per_label=2, target_labels=None):
    """
    Scans AVA-Kinetics annotation CSVs, links them to physical video files in video_dir, and randomly selects N video files per target label.
    :param csv_list: list of filepaths to the annotation CSV files.
    :param video_dir: root directory where downloaded videos are stored in a flat structure.
    :param samples_per_label: number of random samples to select per label.
    :param target_labels: list of allowed class strings. If None, all classes are sampled.
    :return: list of tuples in the format [(label, "video", full_path), ...].
    """

    # Convert target list to a lowercase set for lookups:
    valid_vid_exts = ('.mp4', '.webm', '.mkv', '.avi')
    target_set = {lbl.lower() for lbl in target_labels} if target_labels else set(TARGET_AVA_CLASSES.values())

    # Standardized 8-column header (accommodates both Kinetics and AVA formats):
    col_names = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"]
    df_list = []

    # Parse and filter spreadsheets:
    print(f"\nScanning annotation CSVs to build test playlist...")
    for csv_path in csv_list:
        if not os.path.exists(csv_path):
            continue
        df_temp = pd.read_csv(csv_path, header=None, names=col_names, low_memory=False)
        df_temp = df_temp.dropna(subset=["action_id", "video_id"])
        df_list.append(df_temp)

    # Verify at least one file loaded successfully:
    if not df_list:
        print("Error: No valid CSV spreadsheets found for sampling.")
        return []

    # Combine spreadsheets and map integer IDs to string labels:
    df = pd.concat(df_list, ignore_index=True)
    df["action_id"] = df["action_id"].astype(int)
    df["label"] = df["action_id"].map(TARGET_AVA_CLASSES)

    # Filter strictly for our target classes:
    df = df[df["label"].isin(target_set)]

    # Group unique video IDs by label:
    label_to_videos = {}
    for label, group in df.groupby("label"):
        unique_vids = group["video_id"].unique().tolist()
        label_to_videos[label] = unique_vids

    # Randomly sample N verified physical files from each valid label group:
    selected_samples = []
    print("\n--- Randomly Selected Test Playlist (Verified on Disk) ---")
    for lbl, vid_id_list in sorted(label_to_videos.items()):
        random.shuffle(vid_id_list)
        found_for_class = 0

        # Check all supported container formats on disk:
        for vid_id in vid_id_list:
            if found_for_class >= samples_per_label:
                break
            for ext in valid_vid_exts:
                candidate_path = os.path.join(video_dir, f"{vid_id}{ext}")
                if os.path.exists(candidate_path):
                    selected_samples.append((lbl, "video", candidate_path))
                    print(f"[{lbl.upper()}] -> {vid_id}{ext}")
                    found_for_class += 1
                    break

        if found_for_class == 0:
            print(f"[{lbl.upper()}] -> WARNING: No downloaded videos found on disk!")

    # Randomize the playback order across different classes:
    random.shuffle(selected_samples)
    return selected_samples


def draw_bottom_top5(frame, bbox, top_k_data):
    """
    Renders the predicted simultaneous actions and confidence percentages at the bottom of the bounding box.
    :param frame: annotated image frame.
    :param bbox: bounding box coordinates [x1, y1, x2, y2].
    :param top_k_data: list of (label, prob) tuples or buffering text string.
    """

    x1, y1, x2, y2 = bbox
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 18

    # Still buffering frames (< 30 frames accumulated):
    if isinstance(top_k_data, str):
        text = top_k_data
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (int(x1), int(y2)), (int(x1) + tw + 6, int(y2) + th + 8), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, text, (int(x1) + 3, int(y2) + th + 3), font, font_scale, (0, 255, 255), thickness)
        return

    # Calculate background block height for stacked lines of text:
    total_bg_height = (len(top_k_data) * line_height) + 6
    max_tw = max([cv2.getTextSize(f"{lbl}: {pr * 100:.1f}%", font, font_scale, thickness)[0][0] for lbl, pr in top_k_data])

    # Draw solid black background block below the bottom border (y1):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x1) + max_tw + 10, int(y1) + total_bg_height), (0, 0, 0), cv2.FILLED)

    # Draw each active action label stacked vertically downwards:
    current_y = int(y1) + line_height - 3
    for i, (label, prob) in enumerate(top_k_data):

        # Color coding is green for highest confidence, white for concurrent secondary actions:
        color = (0, 255, 0) if i == 0 else (255, 255, 255)
        text = f"{label}: {prob * 100:.1f}%"
        cv2.putText(frame, text, (int(x1) + 5, current_y), font, font_scale, color, thickness)
        current_y += line_height


def run_test_pipeline(csv_list, video_dir="input/ava_kinetics/videos", samples_per_label=2, target_labels=None, port=8080):
    """
    Main execution loop. Uses CSV mappings to find videos, initializes server and models, and streams annotated frames.
    :param csv_list: list of filepaths to the annotation CSV files.
    :param video_dir: input directory containing downloaded videos.
    :param samples_per_label: number of random samples to select per label.
    :param target_labels: list of allowed class strings. If None, all classes are sampled.
    :param port: port number.
    """

    # Gather playlist:
    playlist = gather_random_samples_from_csv(csv_list, video_dir, samples_per_label, target_labels)
    if not playlist:
        print(f"Error: No valid verified video files found in {video_dir} matching the CSV annotations.")
        return

    # Initialize shared buffer and browser server thread:
    annotated_buffer = FrameBuffer()
    t_server = threading.Thread(target=run_server, args=(annotated_buffer, port), daemon=True)
    t_server.start()
    print(f"\nServer started! Open your browser at: http://127.0.0.1:{port}")
    time.sleep(1.0)

    # Load ML models:
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    print(f"Loading {YOLO_MODEL} model...")
    model = YOLO(YOLO_MODEL)
    predictor = ActionPredictor()

    # Playback loop:
    for clip_idx, (ground_truth_label, kind, path) in enumerate(playlist, 1):
        print(f"\n[Clip {clip_idx}/{len(playlist)}] Streaming: {os.path.basename(path)} (Target: {ground_truth_label.upper()})...")

        # Gather frame generators (limit to first 5 seconds of playback to save testing time):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 30.0 if (fps <= 0 or np.isnan(fps)) else fps
        max_frames = int(fps * 5.0)  # 5 seconds of footage

        frames = []
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # Reset trackers and rolling buffers between independent video clips:
        if model.predictor is not None and hasattr(model.predictor, 'trackers'):
            model.predictor.trackers[0].reset()
        if predictor.session:
            predictor.track_buffers.clear()
            predictor.track_labels.clear()

        total_frames = len(frames)

        # Stream individual clip frames:
        for f_idx, frame in enumerate(frames):

            # Check if this is the absolute last frame of the video clip:
            is_final_frame = (f_idx == total_frames - 1)

            # Run ByteTrack inference:
            results = model.track(source=frame, show=False, tracker="bytetrack.yaml", persist=True, verbose=False)

            # Generate base plot (renders bounding box + ID/Confidence at the top border):
            res_annotated = results[0].plot()
            if results[0].boxes is not None and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints = results[0].keypoints.xy.cpu().numpy()
                bboxes = results[0].boxes.xyxy.cpu().numpy()

                # For each detected person:
                for i, track_id in enumerate(track_ids):
                    # Use multi-label prediction with a 35% sigmoid threshold to report concurrent actions:
                    top_data = predictor.predict_multi_label(
                        track_id,
                        keypoints[i],
                        bboxes[i],
                        threshold=0.35,
                        is_last_frame=is_final_frame,
                        min_frames=10
                    )

                    # Draw active action overlay at the bottom border of the bounding box:
                    draw_bottom_top5(res_annotated, bboxes[i], top_data)

            # Push frame to web server buffer (~30FPS):
            annotated_buffer.set_frame(res_annotated)
            time.sleep(0.033)

            # When we reach the final frame, pause for 2.5 seconds so the user can read the answer:
            if is_final_frame:
                print("End of clip reached -> Holding final prediction frame on browser screen for 2.5s...")
                time.sleep(2.5)

    print("\nTest playlist complete! Exiting...")


if __name__ == "__main__":
    run_test_pipeline(
        csv_list=["kinetics_train_v1.0.csv", "kinetics_val_v1.0.csv", "ava_train_v2.2.csv", "ava_val_v2.2.csv"],
        video_dir="input/ava_kinetics/videos",
        samples_per_label=3,
        target_labels=list(TARGET_AVA_CLASSES.values()),
        port=8080
    )
