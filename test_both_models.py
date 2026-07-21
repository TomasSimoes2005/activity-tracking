import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from ultralytics import YOLO
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, accuracy_score
from src.action_model import ActionDataset, ActionHybridNet
from src.shared import WINDOW_SIZE, NUM_FEATURES, extract_features, crop_interaction_roi, should_trigger_roi_crop

# File paths and configuration constants:
TEMPORAL_BATCH_SIZE = 32
TEMPORAL_CSV_PATH = "output/ava_dataset.csv"
TEMPORAL_MODEL_PATH = "output/model.pth"
TEMPORAL_LABEL_MAP_PATH = "output/label_map.json"

ROI_BATCH_SIZE = 32
ROI_DATA_DIR = "dataset/roi_train"
ROI_MODEL_PATH = "output/roi_classifier.pth"
ROI_LABEL_MAP_PATH = "output/roi_label_map.json"

VIDEO_DIR = "input/ava_kinetics/videos"
CSV_ANNOTATION_LIST = ["kinetics_train_v1.0.csv", "kinetics_val_v1.0.csv", "ava_train_v2.2.csv", "ava_val_v2.2.csv"]

# Shared interaction classes evaluated across both models in deployment:
SHARED_INTERACTION_CLASSES = ["drink", "eat", "smoke"]


def compute_iou(box_a, box_b):
    """
    Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    :param box_a: first bounding box array [x1, y1, x2, y2].
    :param box_b: second bounding box array [x1, y1, x2, y2].
    :return: float IoU value between 0.0 and 1.0.
    """

    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    intersection_area = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    if intersection_area == 0.0:
        return 0.0

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = float(box_a_area + box_b_area - intersection_area)

    return intersection_area / union_area if union_area > 0.0 else 0.0


def test_temporal_model(device):
    """
    Evaluates the trained Temporal Action Model (1D-CNN + BiGRU) on the unseen validation split of the AVA dataset.
    Presents validation loss, binary accuracy, Mean Average Precision (mAP), and per-class statistics.
    :param device: torch hardware execution device (cpu or cuda).
    :return: dictionary containing temporal validation metrics and per-class performance stats.
    """

    if not os.path.exists(TEMPORAL_CSV_PATH) or not os.path.exists(TEMPORAL_MODEL_PATH):
        print(f"Warning: Temporal dataset or model checkpoint not found. Skipping Stage 1.")
        return {}

    full_dataset = ActionDataset(TEMPORAL_CSV_PATH, is_training=False)
    num_classes = len(full_dataset.label_map)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_indices = random_split(
        range(len(full_dataset)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_dataset = torch.utils.data.Subset(
        ActionDataset(TEMPORAL_CSV_PATH, label_map=full_dataset.label_map, is_training=False),
        val_indices
    )
    val_loader = DataLoader(val_dataset, batch_size=TEMPORAL_BATCH_SIZE, shuffle=False)

    model = ActionHybridNet(
        input_size=NUM_FEATURES,
        hidden_size=64,
        num_layers=2,
        num_classes=num_classes
    ).to(device)
    model.load_state_dict(torch.load(TEMPORAL_MODEL_PATH, map_location=device))
    model.eval()

    pos_counts = np.maximum(full_dataset.labels.sum(axis=0), 1.0)
    dampened_weights = np.sqrt((len(full_dataset) - pos_counts) / pos_counts)
    pos_weight = torch.tensor(dampened_weights, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    val_loss = 0.0
    correct_val = 0
    total_val = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * sequences.size(0)

            probs = torch.sigmoid(outputs)
            predicted_binary = (probs > 0.5).float()
            total_val += labels.numel()
            correct_val += (predicted_binary == labels).sum().item()

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = (correct_val / total_val) * 100.0
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)
    map_score = float(average_precision_score(y_true.astype(np.int32), y_pred, average='macro') * 100.0) if len(
        y_true) > 0 else 0.0

    print("\n" + "=" * 65)
    print("STAGE 1: TEMPORAL ACTION MODEL STATISTICS (1D-CNN + BiGRU)")
    print("=" * 65)
    print(f"Validation Sequences Evaluated: {len(val_dataset)}")
    print(f"Overall Validation Loss       : {epoch_val_loss:.4f}")
    print(f"Overall Binary Accuracy       : {epoch_val_acc:.2f}%")
    print(f"Mean Average Precision (mAP)  : {map_score:.2f}%\n")
    print("Per-Class Performance (at Decision Threshold 0.50):")
    print("-" * 65)
    print(f"{'CLASS NAME':<14} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'SUPPORT':<8}")
    print("-" * 65)

    per_class_stats = {}
    for label_str, idx in sorted(full_dataset.label_map.items(), key=lambda item: item[1]):
        class_true = y_true[:, idx]
        class_pred = (y_pred[:, idx] >= 0.50).astype(np.int32)
        prec, rec, f1, _ = precision_recall_fscore_support(class_true, class_pred, average='binary', zero_division=0)
        support = int(np.sum(class_true))
        per_class_stats[label_str] = {
            "precision": prec * 100.0,
            "recall": rec * 100.0,
            "f1": f1 * 100.0,
            "support": support
        }
        print(
            f"{label_str.upper():<14} | {prec * 100:6.2f}%    | {rec * 100:6.2f}%    | {f1 * 100:6.2f}%    | {support:<8}")
    print("-" * 65)

    return per_class_stats


def test_roi_model(device):
    """
    Evaluates the trained Spatial ROI Vision Classifier (MobileNetV3-Small) on unseen validation image crops.
    Presents validation loss, Top-1 multi-class accuracy, and per-class precision, recall, and F1-scores.
    :param device: torch hardware execution device (cpu or cuda).
    :return: dictionary containing spatial ROI per-class performance stats.
    """

    if not os.path.exists(ROI_DATA_DIR) or not os.path.exists(ROI_MODEL_PATH):
        print(f"Warning: ROI image dataset or model checkpoint not found. Skipping Stage 2.")
        return {}

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_full_dataset = datasets.ImageFolder(root=ROI_DATA_DIR, transform=val_transform)
    total_samples = len(val_full_dataset)
    indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42)).tolist()
    train_size = int(0.8 * total_samples)
    val_dataset = torch.utils.data.Subset(val_full_dataset, indices[train_size:])
    val_loader = DataLoader(val_dataset, batch_size=ROI_BATCH_SIZE, shuffle=False, num_workers=0)

    num_classes = len(val_full_dataset.classes)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[2] = nn.Dropout(p=0.5)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(ROI_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = (correct_val / total_val) * 100.0

    print("\n" + "=" * 65)
    print("STAGE 2: SPATIAL ROI VISION CLASSIFIER STATISTICS (MobileNetV3)")
    print("=" * 65)
    print(f"Validation Crops Evaluated    : {len(val_dataset)}")
    print(f"Overall Validation Loss       : {epoch_val_loss:.4f}")
    print(f"Overall Top-1 Accuracy        : {epoch_val_acc:.2f}%\n")
    print("Per-Class Performance:")
    print("-" * 65)
    print(f"{'CLASS NAME':<14} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'SUPPORT':<8}")
    print("-" * 65)

    idx_to_class = {idx: cls_name for cls_name, idx in val_full_dataset.class_to_idx.items()}
    per_class_stats = {}

    for idx in sorted(idx_to_class.keys()):
        cls_name = idx_to_class[idx]
        class_true = (np.array(all_targets) == idx).astype(np.int32)
        class_pred = (np.array(all_preds) == idx).astype(np.int32)
        prec, rec, f1, _ = precision_recall_fscore_support(class_true, class_pred, average='binary', zero_division=0)
        support = int(np.sum(class_true))
        per_class_stats[cls_name] = {
            "precision": prec * 100.0,
            "recall": rec * 100.0,
            "f1": f1 * 100.0,
            "support": support
        }
        print(
            f"{cls_name.upper():<14} | {prec * 100:6.2f}%    | {rec * 100:6.2f}%    | {f1 * 100:6.2f}%    | {support:<8}")
    print("-" * 65)

    return per_class_stats


def evaluate_deployed_interaction_fusion(temporal_stats, roi_stats):
    """
    Simulates and evaluates deployed two-stage inference routing across shared hand-to-face interaction classes.
    Compares skeletal temporal F1 against spatial visual F1 to establish automated routing strategies.
    :param temporal_stats: dictionary containing per-class metrics from the temporal action model test.
    :param roi_stats: dictionary containing per-class metrics from the spatial ROI classifier test.
    """

    print("\n" + "=" * 70)
    print("STAGE 3: DEPLOYED TWO-STAGE FUSION ANALYSIS (Shared Interaction Classes)")
    print("=" * 70)
    print("In deployment, the 1D-CNN+BiGRU monitors continuous posture and motion over 30-frame windows,")
    print("while the MobileNetV3 classifier acts as a visual specialist for hand-to-face interaction crops.")
    print("Below is the comparative performance breakdown and deployed routing strategy on shared classes:\n")

    print("-" * 75)
    print(f"{'ACTION CLASS':<12} | {'TEMPORAL F1':<14} | {'ROI VISION F1':<14} | {'RECOMMENDED DEPLOYED STRATEGY':<28}")
    print("-" * 75)

    for cls_name in SHARED_INTERACTION_CLASSES:
        temp_f1 = temporal_stats.get(cls_name, {}).get("f1", 0.0)
        roi_f1 = roi_stats.get(cls_name, {}).get("f1", 0.0)

        if roi_f1 > temp_f1 + 5.0:
            strategy = "Prioritize ROI Vision Crop"
        elif temp_f1 > roi_f1 + 5.0:
            strategy = "Prioritize Skeleton Sequence"
        else:
            strategy = "Ensemble Gate / Average Prob"

        print(f"{cls_name.upper():<12} | {temp_f1:6.2f}%        | {roi_f1:6.2f}%        | {strategy:<28}")
    print("-" * 75)


def test_end_to_end_fused_pipeline(device, video_dir=VIDEO_DIR, csv_list=CSV_ANNOTATION_LIST):
    """
    Executes Stage 4 with Mutually Exclusive Posture Gating, Alpha-Blended Late Fusion,
    and Dynamic Per-Class Decision Thresholds to maximize global accuracy and mAP.
    """
    print("\n" + "=" * 83)
    print("STAGE 4 (OPTIMIZED): END-TO-END FUSED PIPELINE EVALUATION")
    print("=" * 83)

    if not os.path.exists(video_dir):
        print(f"Error: Video directory '{video_dir}' not found. Skipping Stage 4!")
        return

    valid_vid_exts = ('.mp4', '.webm', '.mkv', '.avi')
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(valid_vid_exts)]
    if len(video_files) == 0:
        print("No video files discovered. Exiting Stage 4.")
        return

    video_id_to_path = {os.path.splitext(f)[0]: os.path.join(video_dir, f) for f in video_files}

    col_names = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"]
    df_list = [pd.read_csv(p, header=None, names=col_names, low_memory=False).dropna(subset=["action_id"]) for p in
               csv_list if os.path.exists(p)]
    if not df_list: return

    df_all = pd.concat(df_list, ignore_index=True)
    df_all["action_id"] = df_all["action_id"].astype(int)
    df_matched = df_all[df_all["video_id"].isin(video_id_to_path.keys())].copy()

    # Load mappings & checkpoints:
    with open(TEMPORAL_LABEL_MAP_PATH, "r") as f:
        temporal_label_map = json.load(f)
    with open(ROI_LABEL_MAP_PATH, "r") as f:
        roi_label_map = json.load(f)
    num_temporal_classes = len(temporal_label_map)
    idx_to_roi = {v: k for k, v in roi_label_map.items()}

    temporal_model = ActionHybridNet(input_size=NUM_FEATURES, hidden_size=64, num_layers=2,
                                     num_classes=num_temporal_classes).to(device)
    temporal_model.load_state_dict(torch.load(TEMPORAL_MODEL_PATH, map_location=device))
    temporal_model.eval()

    roi_model = models.mobilenet_v3_small(weights=None)
    roi_model.classifier[2] = nn.Dropout(p=0.5)
    roi_model.classifier[3] = nn.Linear(roi_model.classifier[3].in_features, len(roi_label_map))
    roi_model.load_state_dict(torch.load(ROI_MODEL_PATH, map_location=device))
    roi_model = roi_model.to(device)
    roi_model.eval()

    roi_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    yolo_pose = YOLO("yolo11n-pose.pt")

    target_ava_id_map = {5: "fall_floor", 8: "lie_sleep", 11: "sit", 12: "stand", 27: "drink", 29: "eat", 54: "smoke",
                         57: "cellphone"}

    # 1. DYNAMIC PER-CLASS DECISION THRESHOLDS (Loaded from optimization scans):
    dynamic_thresholds = {
        "cellphone": 0.54,
        "drink": 0.52,
        "eat": 0.49,
        "fall_floor": 0.49,
        "lie_sleep": 0.39,
        "sit": 0.46,
        "smoke": 0.42,
        "stand": 0.39
    }

    # 2. MUTUALLY EXCLUSIVE POSTURE INDICES:
    posture_classes = ["sit", "stand", "lie_sleep", "fall_floor"]
    posture_indices = [temporal_label_map[c] for c in posture_classes if c in temporal_label_map]

    grouped_seqs = df_matched.groupby(["video_id", "timestamp"])
    print(f"Executing Inference across {len(grouped_seqs)} unique sequences...\n")

    y_true_list, y_fused_prob_list = [], []
    c = 0

    for (video_id, timestamp), group in grouped_seqs:
        c += 1
        if c % 100 == 0:
            print(f"{c}/{len(grouped_seqs)} done...")
        cap = cv2.VideoCapture(video_id_to_path[video_id])
        if not cap.isOpened(): continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        target_vector = np.zeros(num_temporal_classes, dtype=np.float32)
        gt_box = None
        for _, row in group.iterrows():
            cls_str = target_ava_id_map.get(int(row["action_id"]))
            if cls_str in temporal_label_map: target_vector[temporal_label_map[cls_str]] = 1.0
            if gt_box is None: gt_box = [row["x1"] * width, row["y1"] * height, row["x2"] * width, row["y2"] * height]

        center_frame = int(float(timestamp) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, center_frame - (WINDOW_SIZE // 2)))

        feature_buffer, center_frame_patch, center_wrist_near_face = [], None, False

        for f_idx in range(WINDOW_SIZE):
            ret, frame = cap.read()
            if not ret: break
            results = yolo_pose(source=frame, show=False, verbose=False)
            best_iou, best_kpts, best_box = 0.0, None, None

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                kpts_batch = results[0].keypoints.xy.cpu().numpy()
                boxes_batch = results[0].boxes.xyxy.cpu().numpy()
                for idx, det_box in enumerate(boxes_batch):
                    iou = compute_iou(det_box, gt_box)
                    if iou > best_iou: best_iou, best_kpts, best_box = iou, kpts_batch[idx], det_box

            if best_iou > 0.15 and best_kpts is not None:
                feats = extract_features(best_kpts, best_box)
                feature_buffer.append(feats)
                if f_idx == WINDOW_SIZE // 2:
                    # Apply Two-Zone Scale-Invariant Gate (Face & Chest/Texting)
                    should_crop, zone_type = should_trigger_roi_crop(best_kpts, best_box)
                    if should_crop:
                        center_wrist_near_face = True  # Triggers downstream ROI alpha-blending
                        roi_crop = crop_interaction_roi(frame, best_kpts, best_box, padding=40, zone_type=zone_type)
                        if roi_crop is not None and roi_crop.size > 0:
                            center_frame_patch = cv2.resize(roi_crop, (224, 224))
            else:
                feature_buffer.append(feature_buffer[-1] if feature_buffer else [0.0] * NUM_FEATURES)
        cap.release()

        if len(feature_buffer) < 10: continue
        while len(feature_buffer) < WINDOW_SIZE: feature_buffer.append(feature_buffer[-1])

        # A. Temporal Inference:
        window_offsets = [-5, 0, 5]
        temporal_prob_preds = []

        for offset in window_offsets:
            start_idx = max(0, min(len(feature_buffer) - WINDOW_SIZE, (WINDOW_SIZE // 2) + offset - (WINDOW_SIZE // 2)))
            slice_buf = feature_buffer[start_idx: start_idx + WINDOW_SIZE]

            while len(slice_buf) < WINDOW_SIZE:
                slice_buf.append(slice_buf[-1])

            seq_tensor = torch.tensor(np.array(slice_buf, dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(temporal_model(seq_tensor)).cpu().numpy()[0]
                temporal_prob_preds.append(probs)

        # Average the distributions across the 3 consensus windows:
        fused_probs = np.mean(temporal_prob_preds, axis=0)

        # B. Reality Gate: Force Mutually Exclusive Postures (Zero out losing physical states)
        if posture_indices:
            best_posture_idx = posture_indices[np.argmax([fused_probs[i] for i in posture_indices])]
            for idx in posture_indices:
                if idx != best_posture_idx:
                    fused_probs[idx] = 0.001  # Suppress impossible simultaneous postures

        # C. Alpha-Blended ROI Vision Fusion:
        if center_wrist_near_face and center_frame_patch is not None:
            pil_img = transforms.ToPILImage()(cv2.cvtColor(center_frame_patch, cv2.COLOR_BGR2RGB))
            img_tensor = roi_transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                roi_probs = torch.softmax(roi_model(img_tensor), dim=1).cpu().numpy()[0]
            roi_prob_map = {idx_to_roi[i].lower(): roi_probs[i] for i in range(len(roi_label_map))}

            # Blend: 70% Vision / 30% Skeleton for EAT, DRINK, SMOKE
            for interaction_cls in ["drink", "eat", "smoke"]:
                if interaction_cls in temporal_label_map and interaction_cls in roi_prob_map:
                    idx = temporal_label_map[interaction_cls]
                    fused_probs[idx] = (0.30 * fused_probs[idx]) + (0.70 * roi_prob_map[interaction_cls])

            # Blend: 75% Skeleton / 25% Vision for CELLPHONE (due to visual occlusion)
            if "cellphone" in temporal_label_map and "cellphone" in roi_prob_map:
                idx = temporal_label_map["cellphone"]
                fused_probs[idx] = (0.75 * fused_probs[idx]) + (0.25 * roi_prob_map["cellphone"])

            # Empty Hand Gating Suppression:
            if roi_prob_map.get("empty_hand", 0.0) > 0.65:
                for interaction_cls in ["drink", "eat", "smoke", "cellphone"]:
                    if interaction_cls in temporal_label_map:
                        fused_probs[temporal_label_map[interaction_cls]] *= 0.25

        y_true_list.append(target_vector)
        y_fused_prob_list.append(fused_probs)

    # Calculate Metrics using Dynamic Thresholds:
    y_true_matrix = np.vstack(y_true_list)
    y_fused_matrix = np.vstack(y_fused_prob_list)

    y_bin_pred = np.zeros_like(y_fused_matrix, dtype=np.int32)
    for cls_name, idx in temporal_label_map.items():
        thresh = dynamic_thresholds.get(cls_name.lower(), 0.40)
        y_bin_pred[:, idx] = (y_fused_matrix[:, idx] >= thresh).astype(np.int32)

    fused_map = float(average_precision_score(y_true_matrix.astype(np.int32), y_fused_matrix, average='macro') * 100.0)
    exact_match_acc = accuracy_score(y_true_matrix.astype(np.int32), y_bin_pred) * 100.0

    print("\n" + "=" * 83)
    print("STAGE 4: FUSED PIPELINE PERFORMANCE SUMMARY")
    print("=" * 83)
    print(f"Total Video Sequences Evaluated: {len(y_true_matrix)}")
    print(f"Exact Match Ratio (Subset Acc) : {exact_match_acc:.2f}% (Surged via Mutually Exclusive Gating)")
    print(f"Mean Average Precision (mAP)   : {fused_map:.2f}%\n")
    print("Per-Class Optimized Performance (Dynamic Thresholding & Alpha Blending):")
    print("-" * 83)
    print(f"{'CLASS NAME':<14} | {'THRESHOLD':<10} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 83)

    for label_str, idx in sorted(temporal_label_map.items(), key=lambda item: item[1]):
        class_true = y_true_matrix[:, idx]
        class_pred = y_bin_pred[:, idx]
        prec, rec, f1, _ = precision_recall_fscore_support(class_true, class_pred, average='binary', zero_division=0)
        acc = accuracy_score(class_true, class_pred) * 100.0
        thresh_val = dynamic_thresholds.get(label_str.lower(), 0.40)
        print(
            f"{label_str.upper():<14} | {thresh_val:<10.2f} | {acc:6.2f}%    | {prec * 100:6.2f}%    | {rec * 100:6.2f}%    | {f1 * 100:6.2f}%")
    print("-" * 83)


if __name__ == "__main__":
    active_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Global Evaluation Suite initialized on hardware device: [{active_device}]")

    # Execute all evaluation stages sequentially:
    temporal_metrics = test_temporal_model(active_device)
    roi_metrics = test_roi_model(active_device)

    if temporal_metrics and roi_metrics:
        evaluate_deployed_interaction_fusion(temporal_metrics, roi_metrics)

    # Execute Stage 4 end-to-end fused evaluation across local AVA-Kinetics videos:
    test_end_to_end_fused_pipeline(active_device)

    print("\nGlobal Evaluation Complete!")
