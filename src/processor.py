import time
import cv2
import logging
from ultralytics import YOLO
from src.dataset_writer import DatasetWriter
from src.action_predictor import ActionPredictor

# Selected YOLO model:
YOLO_MODEL = "yolo11n-pose.pt"


def process_yolo(raw_buffer, annotated_buffer=None, record_label=None):
    """
    Annotates frames using YOLO and runs real-time temporal action recognition.
    :param raw_buffer: raw frame buffer.
    :param annotated_buffer: annotated frame buffer. Used only if server is active.
    :param record_label: indicates training frames. If not None then all keypoints and boxes are recorded in a training dataset using this as the label.
    """

    # Load YOLO model:
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    print(f"Loading {YOLO_MODEL} model...")
    model = YOLO(YOLO_MODEL)

    # Initialize CSV dataset writer if recording is enabled:
    writer = DatasetWriter() if record_label else None
    if record_label:
        print(f"RECORDING TRAINING DATA. Saving actions as: '{record_label}'")

    # Initialize ONNX Action Recognition engine if NOT in recording mode:
    predictor = ActionPredictor() if not record_label else None

    # Until stopped:
    while True:

        # Get next frame:
        frame = raw_buffer.get_frame_yolo()
        if frame is not None:

            # Get results from model:
            results = model.track(source=frame, show=False, tracker="bytetrack.yaml", persist=True, verbose=False)

            # Generate base annotated frame from YOLO:
            res_annotated = results[0].plot() if annotated_buffer is not None else None

            # If results are valid and people are detected:
            if results[0].boxes is not None and results[0].boxes.id is not None:

                # Extract values:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints = results[0].keypoints.xy.cpu().numpy()
                bboxes = results[0].boxes.xyxy.cpu().numpy()

                # For each tracked person:
                for i, track_id in enumerate(track_ids):

                    # If recording training data, save to CSV:
                    if writer and record_label:
                        writer.process_and_save(record_label, track_id, keypoints[i], bboxes[i])

                    # If live inference is active, predict their action:
                    elif predictor and res_annotated is not None:
                        action_text = predictor.predict(track_id, keypoints[i], bboxes[i])

                        # Extract bounding box top-left corner:
                        x1, y1, x2, y2 = bboxes[i]

                        # Draw custom Action Label overlay above the bounding box:
                        label_str = f"ID {track_id}: {action_text}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2

                        # Calculate text background size for readability:
                        (text_width, text_height), baseline = cv2.getTextSize(label_str, font, font_scale, thickness)
                        cv2.rectangle(res_annotated, (int(x1), int(y1) - text_height - 10), (int(x1) + text_width, int(y1)), (0, 0, 0), cv2.FILLED)
                        cv2.putText(res_annotated, label_str, (int(x1), int(y1) - 5), font, font_scale, (0, 255, 0), thickness)

            # If browser displaying is enabled, push annotated frame to web server buffer:
            if annotated_buffer is not None and res_annotated is not None:
                annotated_buffer.set_frame(res_annotated)

        # If no frame is available, wait:
        else:
            time.sleep(0.005)
