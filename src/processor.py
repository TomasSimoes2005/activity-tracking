import time
import cv2
import numpy as np
from ultralytics import YOLO
from src.action_predictor import ActionPredictor
from src.dataset_writer import DatasetWriter
from src.roi_classifier import ROIClassifier
from src.shared import extract_features, crop_interaction_roi


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
    Continuously pulls raw frames, runs YOLO tracking, and predicts multi-label actions with ROI refinement.
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
    roi_classifier = None
    writer = None

    # Mode Selection: Live Inference vs. Data Recording
    if record_label:
        print(f"\n[Data Collection Mode] Recording physical actions for label: {record_label.upper()}")
        writer = DatasetWriter(filename=f"output/live_{record_label.lower()}.csv")
    else:

        # Initialize inference ONNX temporal classifier:
        print("\n[Live Inference Mode] Initializing ONNX Engine...")
        predictor = ActionPredictor(model_path="output/model.onnx", label_map_path="output/label_map.json")
        
        # Initialize secondary gated ROI vision classifier:
        print("[Live Inference Mode] Initializing ONNX ROI Vision Classifier...")
        roi_classifier = ROIClassifier(model_path="output/roi_classifier.onnx", label_map_path="output/roi_label_map.json")

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

                    # 1. Pull raw temporal probabilities:
                    temp_result, temp_map = predictor.get_raw_probabilities(track_id, keypoints[i], bboxes[i])

                    if isinstance(temp_result, str):
                        draw_bottom_top5(res_annotated, bboxes[i], temp_result)
                        continue
                    elif temp_result is None:
                        continue

                    fused_probs = temp_result.copy()
                    label_to_idx = {v: k for k, v in temp_map.items()}

                    # 2. Check wrist-to-nose proximity for ROI vision triggering:
                    feats = extract_features(keypoints[i], bboxes[i])
                    lw_to_nose, rw_to_nose = feats[34], feats[35]
                    is_wrist_near_face = (0.0 < lw_to_nose < 1.5) or (0.0 < rw_to_nose < 1.5)

                    if is_wrist_near_face and roi_classifier is not None:
                        roi_patch = crop_interaction_roi(frame, keypoints[i], bboxes[i], padding=40)
                        roi_probs, roi_map = roi_classifier.get_patch_probabilities(roi_patch)

                        if roi_probs is not None:
                            roi_label_to_idx = {v: k for k, v in roi_map.items()}
                            
                            # ALPHA BLEND: 30% Skeleton / 70% Vision for Oral Interactions
                            for oral_cls in ["DRINK", "EAT", "SMOKE"]:
                                if oral_cls in label_to_idx and oral_cls in roi_label_to_idx:
                                    t_idx = label_to_idx[oral_cls]
                                    r_idx = roi_label_to_idx[oral_cls]
                                    fused_probs[t_idx] = (0.30 * fused_probs[t_idx]) + (0.70 * roi_probs[r_idx])

                            # ALPHA BLEND: 75% Skeleton / 25% Vision for Cellphone (occlusion resistant)
                            if "CELLPHONE" in label_to_idx and "CELLPHONE" in roi_label_to_idx:
                                t_idx = label_to_idx["CELLPHONE"]
                                r_idx = roi_label_to_idx["CELLPHONE"]
                                fused_probs[t_idx] = (0.75 * fused_probs[t_idx]) + (0.25 * roi_probs[r_idx])

                            # GATING: If vision sees an Empty Hand (>65%), suppress false interaction triggers
                            if "EMPTY_HAND" in roi_label_to_idx and roi_probs[roi_label_to_idx["EMPTY_HAND"]] > 0.65:
                                for action_cls in ["DRINK", "EAT", "SMOKE", "CELLPHONE"]:
                                    if action_cls in label_to_idx:
                                        fused_probs[label_to_idx[action_cls]] *= 0.25

                    # 3. REALITY GATE: Enforce Mutually Exclusive Postures & Interactions
                    posture_cluster = ["SIT", "STAND", "LIE_SLEEP", "FALL_FLOOR"]
                    posture_indices = [label_to_idx[c] for c in posture_cluster if c in label_to_idx]
                    if posture_indices:
                        best_post_idx = posture_indices[np.argmax([fused_probs[idx] for idx in posture_indices])]
                        for idx in posture_indices:
                            if idx != best_post_idx: fused_probs[idx] = 0.001

                    oral_cluster = ["EAT", "DRINK", "SMOKE"]
                    oral_indices = [label_to_idx[c] for c in oral_cluster if c in label_to_idx]
                    if oral_indices:
                        best_oral_idx = oral_indices[np.argmax([fused_probs[idx] for idx in oral_indices])]
                        for idx in oral_indices:
                            if idx != best_oral_idx: fused_probs[idx] = 0.001

                    # 4. Apply Dynamic Decision Thresholds:
                    top_data = []
                    for idx, prob in enumerate(fused_probs):
                        label_str = temp_map[idx]
                        thresh = custom_thresholds.get(label_str, 0.40) if custom_thresholds else 0.40
                        if prob >= thresh:
                            top_data.append((label_str, float(prob)))

                    top_data.sort(key=lambda x: x[1], reverse=True)
                    if not top_data:
                        best_idx = int(np.argmax(fused_probs))
                        top_data = [(temp_map[best_idx], float(fused_probs[best_idx]))]

                    draw_bottom_top5(res_annotated, bboxes[i], top_data)

        # Push annotated frame to the web server buffer:
        if annotated_buffer is not None:
            annotated_buffer.set_frame(res_annotated)
