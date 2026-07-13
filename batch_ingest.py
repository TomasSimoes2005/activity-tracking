import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from src.dataset_writer import DatasetWriter

# Selected YOLO model:
YOLO_MODEL = "yolo11n-pose.pt"

# Set concurrency level:
MAX_WORKERS = 8


def extract_label_from_path(folder_path, root_dir):
    """
    Extracts the top-level action label from the clip's folder path.
    :param folder_path: full path of the folder containing the jpg frames.
    :param root_dir: root input directory.
    :return: cleaned action label.
    """

    try:
        # Resolve absolute paths:
        abs_folder = Path(folder_path).resolve()
        abs_root = Path(root_dir).resolve()

        # Get the path relative to the root directory:
        rel_path = abs_folder.relative_to(abs_root)

        # The first part of the relative path is ALWAYS the class folder:
        if len(rel_path.parts) > 0:
            return rel_path.parts[0].lower()
        else:
            return abs_folder.name.lower()

    except ValueError:
        return Path(folder_path).parent.name.lower()


def _worker_process_subset(worker_id, clip_dirs_subset, root_dir, temp_csv_path):
    """
    Worker function executed by each thread. Processes an assigned subset of video folders and writes to an isolated, thread-specific CSV file to prevent lock contention.
    :param worker_id: integer ID of the thread worker.
    :param clip_dirs_subset: list of folder paths assigned to this worker.
    :param root_dir: root input directory.
    :param temp_csv_path: path to the worker's isolated output CSV file.
    """

    # Disable verbose YOLO logging for this thread:
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # Each worker initializes its own model instance in thread memory:
    print(f"[Worker {worker_id}] Initializing {YOLO_MODEL} (Assigned {len(clip_dirs_subset)} folders)...")
    model = YOLO(YOLO_MODEL)

    # Each worker writes exclusively to its own temporary CSV file:
    writer = DatasetWriter(filename=temp_csv_path, window_size=30)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # For each assigned clip folder:
    for folder in clip_dirs_subset:

        # Extract label:
        label = extract_label_from_path(folder, root_dir)

        # Get chronologically sorted image frames:
        image_files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(valid_extensions)
        ])

        # Process frames:
        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            # Run ByteTrack inference:
            results = model.track(source=frame, show=False, tracker="bytetrack.yaml", persist=True, verbose=False)

            # If results are valid:
            if results[0].boxes is not None and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints = results[0].keypoints.xy.cpu().numpy()
                bboxes = results[0].boxes.xyxy.cpu().numpy()

                # Save keypoints to the worker's private buffer:
                for i, track_id in enumerate(track_ids):
                    writer.process_and_save(label, track_id, keypoints[i], bboxes[i])

        # Pad short clips (>= 10 frames) up to 30 frames and flush to disk:
        writer.pad_and_flush(label, min_frames=10)

        # Reset ByteTrack state for the next folder:
        model.predictor.trackers[0].reset()

    print(f"[Worker {worker_id}] Completed processing all assigned folders!")


def process_frame_folders_parallel(input_dir="input", output_csv="output/dataset.csv", target_labels=None):
    """
    Dispatches frame folders across multiple worker threads for maximum throughput, then merges all thread-specific CSV files into a unified master dataset.
    :param input_dir: root directory containing class subfolders of image sequences.
    :param output_csv: path to the final merged output CSV file.
    :param target_labels: list of desired class labels.
    """

    # Check if input directory exists:
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    # Find all directories containing sequential images:
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    clip_dirs = []
    for root, _, files in os.walk(input_dir):
        if any(f.lower().endswith(valid_extensions) for f in files):
            clip_dirs.append(root)

    # Only search desired labels:
    if target_labels is not None:
        print(f"Reduced clips from {len(clip_dirs)} to ", end='')

        # Convert target_labels to a lowercase set for lookups:
        target_set = {lbl.lower() for lbl in target_labels}

        # Keep only folders whose extracted class label matches our targets:
        clip_dirs = [
            clip for clip in clip_dirs
            if extract_label_from_path(clip, input_dir) in target_set
        ]
        print(len(clip_dirs))

    # If no frame directories were found:
    if not clip_dirs:
        print(f"No image sequences found in {input_dir}{" using the labels " + target_labels if target_labels is not None else ""}.")
        return

    # Sort folders to ensure deterministic distribution across workers:
    clip_dirs = sorted(clip_dirs)
    total_clips = len(clip_dirs)
    print(f"Found {total_clips} sequence folders. Preparing parallel execution across {MAX_WORKERS} workers...")

    # Partition the folder list into nearly equal chunks for each worker:
    chunk_size = (total_clips + MAX_WORKERS - 1) // MAX_WORKERS
    chunks = [clip_dirs[i:i + chunk_size] for i in range(0, total_clips, chunk_size)]

    # Prepare temporary CSV file paths:
    temp_csv_files = [f"{output_csv}.part{i}" for i in range(len(chunks))]

    # Ensure output directory exists:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Launch worker threads:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            # Submit worker task to thread pool:
            futures.append(executor.submit(_worker_process_subset, i, chunk, input_dir, temp_csv_files[i]))

        # Wait for all threads to finish:
        for future in futures:
            future.result()

    # Write header and concatenate rows from all temporary files:
    print("\nAll workers finished! Merging temporary thread files into master CSV...")
    with open(output_csv, mode='w', newline='') as master_f:
        header_written = False

        for temp_file in temp_csv_files:
            if not os.path.exists(temp_file):
                continue

            with open(temp_file, mode='r', newline='') as part_f:
                lines = part_f.readlines()
                if not lines:
                    continue

                # Write header only once from the very first non-empty part file:
                if not header_written:
                    master_f.write(lines[0])
                    header_written = True

                # Append all data rows (skipping the header of each part file):
                for line in lines[1:]:
                    master_f.write(line)

            # Delete the temporary part file after merging:
            os.remove(temp_file)

    print(f"Parallel ingestion complete! Unified dataset successfully saved to: {output_csv}")


if __name__ == "__main__":
    target_labels = [
        "drink",
        "eat",
        "fall_floor",
        "run",
        "sit",
        "smoke",
        "stand",
        "walk"
    ]
    process_frame_folders_parallel(input_dir="input/hmdb51", output_csv="output/hmdb51.csv", target_labels=target_labels)
