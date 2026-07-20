import time
import cv2
from ultralytics import YOLO
from src.action_predictor import ActionPredictor
from src.dataset_writer import DatasetWriter


def draw_bottom_top5(frame, bbox, top_k_data):
    """
    Renders the predicted simultaneous actions and confidence percentages at the bottom of the bounding box.
    :param frame: annotated image frame.
    :param bbox: bounding box coordinates [x1, y1, x2, y2].
    :param top_k_data: list of (label, prob) tuples or buffering text string.
    """

    x1, y1, _, y2 = bbox
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 18

    # Check if still buffering:
    if isinstance(top_k_data, str):
        text = top_k_data
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (int(x1), int(y2)), (int(x1) + tw + 6, int(y2) + th + 8), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, text, (int(x1) + 3, int(y2) + th + 3), font, font_scale, (0, 255, 255), thickness)
        return

    # Calculate background block height:
    total_bg_height = (len(top_k_data) * line_height) + 6
    max_tw = 0
    for label, prob in top_k_data:
        text = f"{label}: {prob * 100:.1f}%"
        (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        if tw > max_tw:
            max_tw = tw
    
    # Draw background rectangle:
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x1) + max_tw + 10, int(y1) + total_bg_height), (0, 0, 0), cv2.FILLED)

    # Draw text lines:
    current_y = int(y1) + line_height - 3
    for i, (label, prob) in enumerate(top_k_data):
        color = (0, 255, 0) if i == 0 else (255, 255, 255)
        text = f"{label}: {prob * 100:.1f}%"
        cv2.putText(frame, text, (int(x1) + 5, current_y), font, font_scale, color, thickness)
        current_y += line_height


def process_yolo(raw_buffer, annotated_buffer, record_label=None, custom_thresholds=None):
    """
    Continuously pulls raw frames, runs YOLO tracking, and predicts multi-label actions.
    If record_label is provided, it switches to Data Collection Mode and bypasses prediction.
    :param raw_buffer: FrameBuffer instance containing raw camera frames.
    :param annotated_buffer: FrameBuffer instance to store annotated frames.
    :param record_label: string label for recording training data.
    :param custom_thresholds: dictionary mapping class label strings to specific optimal float thresholds.
    """
    
    # Initialize YOLO model:
    # NOTE: Swap this for 'yolo11n-pose.hef' when deploying the Hailo-8 compiler on the Raspberry Pi
    model = YOLO("yolo11n-pose.pt")
    
    predictor = None
    writer = None

    # Mode Selection: Live Inference vs. Data Recording
    if record_label:
        print(f"\n[Data Collection Mode] Recording physical actions for label: {record_label.upper()}")
        # Initializes the 46-feature CSV layout defined in your DatasetWriter class:
        writer = DatasetWriter(filename=f"output/live_{record_label.lower()}.csv")
    else:
        print("\n[Live Inference Mode] Initializing ONNX Engine...")
        predictor = ActionPredictor(model_path="output/model.onnx", label_map_path="output/label_map.json")

    # Processing loop:
    while True:
        
        # Grab latest frame from the camera buffer:
        frame = raw_buffer.get_frame_yolo()
        if frame is None:
            time.sleep(0.01)
            continue

        # Run ByteTrack Pose Estimation:
        results = model.track(source=frame, persist=True, show=False, verbose=False)
        res_annotated = results[0].plot()

        # Check if people are detected:
        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.xy.cpu().numpy()
            bboxes = results[0].boxes.xyxy.cpu().numpy()

            # Process each tracked person:
            for i, track_id in enumerate(track_ids):
                if writer is not None:
                    # Feed exact skeletal coordinates directly to your DatasetWriter:
                    writer.process_and_save(record_label.lower(), track_id, keypoints[i], bboxes[i])
                    
                    # Visual Feedback for Recording:
                    cv2.putText(res_annotated, f"RECORDING: {record_label.upper()}", 
                                (int(bboxes[i][0]), int(bboxes[i][1]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                elif predictor is not None:
                    # Predict multi-label actions using custom thresholds:
                    top_data = predictor.predict_multi_label(
                        track_id,
                        keypoints[i],
                        bboxes[i],
                        default_threshold=0.40,
                        custom_thresholds=custom_thresholds,
                        is_last_frame=False
                    )

                    # Draw the labels onto the frame:
                    draw_bottom_top5(res_annotated, bboxes[i], top_data)

        # Push annotated frame to the web server buffer:
        if annotated_buffer is not None:
            annotated_buffer.set_frame(res_annotated)
