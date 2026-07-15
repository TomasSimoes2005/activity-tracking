import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time
import random
import logging
import threading
from pathlib import Path
from ultralytics import YOLO
from src.shared import FrameBuffer
from src.server import run_server
from src.action_predictor import ActionPredictor

# Selected YOLO model:
YOLO_MODEL = "yolo11n-pose.pt"

# Only sample from these specific target classes:
TARGET_LABELS = [
    "fall_floor",
    "lie_sleep",
    "sit",
    "stand",
    "drink",
    "eat",
    "smoke",
    "cellphone"
]


def extract_label_from_path(filepath, root_dir):
    """
    Extracts the top-level action label from the video or sequence folder path.
    :param filepath: full path of the video file or clip folder.
    :param root_dir: root input directory.
    :return: cleaned action label.
    """

    try:
        abs_path = Path(filepath).resolve()
        abs_root = Path(root_dir).resolve()
        rel_path = abs_path.relative_to(abs_root)
        return rel_path.parts[0].lower() if len(rel_path.parts) > 0 else abs_path.stem.lower()
    except ValueError:
        return Path(filepath).parent.name.lower()


def gather_random_samples(input_dir, samples_per_label=2, target_labels=None):
    """
    Scans the input directory and randomly selects N video files or frame sequences per label,
    filtering strictly by the target_labels list if provided.
    :param input_dir: root directory containing class subfolders.
    :param samples_per_label: number of random samples to select per label.
    :param target_labels: list of allowed class strings. If None, all found classes are sampled.
    :return: list of selected filepaths or folder paths.
    """

    # Convert target list to a lowercase set for lookups:
    target_set = {lbl.lower() for lbl in target_labels} if target_labels is not None else None
    valid_vid_exts = ('.mp4', '.avi', '.mov', '.mkv')
    valid_img_exts = ('.jpg', '.jpeg', '.png', '.bmp')

    # Group all valid items by label:
    label_groups = {}
    for root, _, files in os.walk(input_dir):

        # Video file ingestion:
        for file in files:
            if file.lower().endswith(valid_vid_exts):
                full_path = os.path.join(root, file)
                lbl = extract_label_from_path(full_path, input_dir)
                if target_set is None or lbl in target_set:
                    label_groups.setdefault(lbl, []).append(("video", full_path))

        # JPG Sequence folder ingestion:
        if any(f.lower().endswith(valid_img_exts) for f in files):
            lbl = extract_label_from_path(root, input_dir)
            if target_set is None or lbl in target_set:
                label_groups.setdefault(lbl, []).append(("sequence", root))

    # Randomly sample N items from each valid label group:
    selected_samples = []
    print("\n--- Randomly Selected Test Playlist ---")
    for lbl, items in sorted(label_groups.items()):
        n = min(samples_per_label, len(items))
        chosen = random.sample(items, n)
        selected_samples.extend(chosen)
        for kind, path in chosen:
            print(f"[{lbl.upper()}] -> {os.path.basename(path)} ({kind})")

    # Randomize the playback order across different classes:
    random.shuffle(selected_samples)
    return selected_samples


def draw_bottom_top5(frame, bbox, top_k_data):
    """
    Renders the top 5 predicted actions and confidence percentages at the bottom of the bounding box.
    :param frame: annotated image frame.
    :param bbox: bounding box coordinates [x1, y1, x2, y2].
    :param top_k_data: list of (label, prob) tuples or buffering text string.
    """

    x1, y1, x2, y2 = bbox
    frame_height, frame_width = frame.shape[:2]
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

    # Calculate background block height for 5 stacked lines of text:
    total_bg_height = (len(top_k_data) * line_height) + 6
    max_tw = 0
    for label, prob in top_k_data:
        text = f"{label}: {prob * 100:.1f}%"
        (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        if tw > max_tw:
            max_tw = tw

    # Draw solid black background block below the bottom border (y2):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x1) + max_tw + 10, int(y1) + total_bg_height), (0, 0, 0), cv2.FILLED)

    # Draw each action label stacked vertically downwards:
    current_y = int(y1) + line_height - 3
    for i, (label, prob) in enumerate(top_k_data):

        # Color coding is green for #1 prediction, white for runners-up:
        color = (0, 255, 0) if i == 0 else (255, 255, 255)
        text = f"{label}: {prob * 100:.1f}%"
        cv2.putText(frame, text, (int(x1) + 5, current_y), font, font_scale, color, thickness)
        current_y += line_height


def run_test_pipeline(input_dir="input", samples_per_label=2, target_labels=None, port=8080):
    """
    Main execution loop. Samples random videos, initializes server and models, and streams annotated frames.
    :param input_dir: input directory path.
    :param samples_per_label: number of random samples to select per label.
    :param target_labels: list of allowed class strings. If None, all found classes are sampled.
    :param port: port number.
    """

    # Gather playlist:
    playlist = gather_random_samples(input_dir, samples_per_label, target_labels)
    if not playlist:
        print(f"Error: No valid video files or JPG sequences found in input directory {input_dir}.")
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
    for clip_idx, (kind, path) in enumerate(playlist, 1):
        print(f"\n[Clip {clip_idx}/{len(playlist)}] Streaming: {os.path.basename(path)}...")

        # Gather frame generators:
        if kind == "video":
            cap = cv2.VideoCapture(path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        else:
            valid_img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            img_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_img_exts)])
            frames = [cv2.imread(f) for f in img_files if cv2.imread(f) is not None]

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
                    # Pass the is_last_frame flag to force Edge Padding on short clips:
                    top5_data = predictor.predict_top_k(
                        track_id,
                        keypoints[i],
                        bboxes[i],
                        k=5,
                        is_last_frame=is_final_frame,
                        min_frames=10
                    )

                    # Draw Top-5 overlay at the bottom border of the bounding box:
                    draw_bottom_top5(res_annotated, bboxes[i], top5_data)

            # Push frame to web server buffer (~30FPS):
            annotated_buffer.set_frame(res_annotated)
            time.sleep(0.033)

            # When we reach the final frame, pause for 2.5 seconds so the user can read the answer:
            if is_final_frame:
                print("End of clip reached -> Holding final prediction frame on browser screen for 2.5s...")
                time.sleep(2.5)

    print("\nTest playlist complete! Exiting...")


if __name__ == "__main__":
    run_test_pipeline(input_dir="input/ava_kinetics/videos", samples_per_label=5, target_labels=TARGET_LABELS, port=8080)
