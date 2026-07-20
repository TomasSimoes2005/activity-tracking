import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from src.action_model import ActionDataset, ActionHybridNet
from src.shared import WINDOW_SIZE, NUM_FEATURES
from sklearn.metrics import precision_recall_fscore_support, average_precision_score


TEMPORAL_BATCH_SIZE = 32
TEMPORAL_LR = 0.001
TEMPORAL_EPOCHS = 60
CSV_PATH = "output/ava_dataset.csv"
TEMPORAL_MODEL_SAVE = "output/model.pth"
TEMPORAL_ONNX_SAVE = "output/model.onnx"
TEMPORAL_LABEL_MAP = "output/label_map.json"
DIAGNOSTIC_SAVE_PATH = "output/diagnostic_report.json"
ROI_BATCH_SIZE = 32
ROI_LR = 0.0003
ROI_WEIGHT_DECAY = 0.01
ROI_EPOCHS = 15
ROI_DATA_DIR = "dataset/roi_train"
ROI_MODEL_SAVE = "output/roi_classifier.pth"
ROI_ONNX_SAVE = "output/roi_classifier.onnx"
ROI_LABEL_MAP = "output/roi_label_map.json"


def scan_optimal_thresholds(y_true, y_pred_probs, label_map):
    """
    Scans decision thresholds from 0.10 to 0.90 independently for each class to find the exact boundary that maximizes F1-Score.
    """

    optimal_thresholds, optimal_metrics = {}, {}
    threshold_range = np.linspace(0.10, 0.90, 81)

    print("\n--- Running Automated Per-Class Threshold Optimization ---")
    for label_str, idx in sorted(label_map.items(), key=lambda item: item[1]):
        best_f1, best_thresh, best_prec, best_rec = 0.0, 0.50, 0.0, 0.0
        class_true, class_probs = y_true[:, idx], y_pred_probs[:, idx]

        for thresh in threshold_range:
            class_pred = (class_probs >= thresh).astype(np.int32)
            prec, rec, f1, _ = precision_recall_fscore_support(class_true, class_pred, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh, best_prec, best_rec = f1, thresh, prec, rec

        optimal_thresholds[label_str.upper()] = float(best_thresh)
        optimal_metrics[label_str.upper()] = {
            "threshold": float(best_thresh), "precision": float(best_prec * 100.0),
            "recall": float(best_rec * 100.0), "f1_score": float(best_f1 * 100.0),
            "support": int(np.sum(class_true))
        }
        print(f"[{label_str.upper():<11}] -> Optimal Thresh: {best_thresh:.2f} | Prec: {best_prec*100:5.1f}% | Rec: {best_rec*100:5.1f}% | F1: {best_f1*100:5.1f}%")

    print("-" * 60)
    return optimal_thresholds, optimal_metrics


def extract_error_autopsy(y_true, y_pred_probs, label_map, top_k=5):
    """
    Identifies and extracts the worst False Positives and False Negatives for each class to diagnose feature blindspots.
    """

    autopsy_report = {}
    for label_str, idx in sorted(label_map.items(), key=lambda item: item[1]):
        class_true, class_probs = y_true[:, idx], y_pred_probs[:, idx]

        fp_indices = np.where(class_true == 0.0)[0]
        fp_sorted = fp_indices[np.argsort(class_probs[fp_indices])[::-1]][:top_k]
        fp_data = [{"sample_idx": int(i), "confidence": float(class_probs[i] * 100.0)} for i in fp_sorted if class_probs[i] > 0.30]

        fn_indices = np.where(class_true == 1.0)[0]
        fn_sorted = fn_indices[np.argsort(class_probs[fn_indices])][:top_k]
        fn_data = [{"sample_idx": int(i), "confidence": float(class_probs[i] * 100.0)} for i in fn_sorted if class_probs[i] < 0.70]

        autopsy_report[label_str.upper()] = {"top_false_positives": fp_data, "top_false_negatives": fn_data}
    return autopsy_report


def train_temporal_diagnostic_engine(device):
    """
    Executes the multi-label training pipeline, tracks gradient telemetry, optimizes decision thresholds,
    and returns comprehensive telemetry dictionaries for the JSON diagnostic report.
    """
    print("\n" + "="*60)
    print("STAGE 1: DIAGNOSTIC TRAINING FOR TEMPORAL ACTION MODEL")
    print("="*60)

    if not os.path.exists(CSV_PATH):
        print(f"Error: Dataset not found at {CSV_PATH}. Run ingestion first!")
        return None

    full_dataset = ActionDataset(CSV_PATH, is_training=False)
    num_classes = len(full_dataset.label_map)

    os.makedirs(os.path.dirname(TEMPORAL_MODEL_SAVE), exist_ok=True)
    with open(TEMPORAL_LABEL_MAP, "w") as f:
        json.dump(full_dataset.label_map, f, indent=4)

    pos_counts = np.maximum(full_dataset.labels.sum(axis=0), 1.0)
    dampened_weights = np.sqrt((len(full_dataset) - pos_counts) / pos_counts)
    pos_weight = torch.tensor(dampened_weights, dtype=torch.float32).to(device)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

    train_loader = DataLoader(torch.utils.data.Subset(ActionDataset(CSV_PATH, label_map=full_dataset.label_map, is_training=True), train_indices), batch_size=TEMPORAL_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(ActionDataset(CSV_PATH, label_map=full_dataset.label_map, is_training=False), val_indices), batch_size=TEMPORAL_BATCH_SIZE, shuffle=False)

    model = ActionHybridNet(input_size=NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TEMPORAL_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TEMPORAL_EPOCHS, eta_min=1e-5)

    best_val_loss = float('inf')
    telemetry_history = []

    for epoch in range(TEMPORAL_EPOCHS):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        epoch_grad_norms = []

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()

            total_norm = (sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)) ** 0.5
            epoch_grad_norms.append(total_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * sequences.size(0)
            predicted_binary = (torch.sigmoid(outputs.data) > 0.5).float()
            total_train += labels.numel()
            correct_train += (predicted_binary == labels).sum().item()

        scheduler.step()
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = (correct_train / total_train) * 100.0
        avg_grad_norm = float(np.mean(epoch_grad_norms))

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                val_loss += criterion(outputs, labels).item() * sequences.size(0)
                predicted_binary = (torch.sigmoid(outputs.data) > 0.5).float()
                total_val += labels.numel()
                correct_val += (predicted_binary == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = (correct_val / total_val) * 100.0

        telemetry_history.append({
            "epoch": epoch + 1, "train_loss": float(epoch_train_loss), "val_loss": float(epoch_val_loss),
            "train_bin_acc": float(epoch_train_acc), "val_bin_acc": float(epoch_val_acc),
            "grad_norm": float(avg_grad_norm), "lr": float(scheduler.get_last_lr()[0])
        })
        print(f"Epoch [{epoch + 1:02d}/{TEMPORAL_EPOCHS:02d}] | Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.2f}% | GradNorm: {avg_grad_norm:.2f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), TEMPORAL_MODEL_SAVE)

    print("\nExporting Temporal model to ONNX format...")
    model.load_state_dict(torch.load(TEMPORAL_MODEL_SAVE))
    model.eval()
    torch.onnx.export(model, torch.randn(1, WINDOW_SIZE, NUM_FEATURES, device=device), TEMPORAL_ONNX_SAVE, export_params=True, opset_version=18, do_constant_folding=True, input_names=['input_sequence'], output_names=['action_logits'], dynamic_axes={'input_sequence': {0: 'batch_size'}, 'action_logits': {0: 'batch_size'}})

    # Evaluate metrics:
    all_val_targets, all_val_preds = [], []
    with torch.no_grad():
        for sequences, labels in val_loader:
            all_val_preds.append(torch.sigmoid(model(sequences.to(device))).cpu().numpy())
            all_val_targets.append(labels.numpy())

    y_true_matrix, y_pred_matrix = np.vstack(all_val_targets), np.vstack(all_val_preds)
    global_map = float(average_precision_score(y_true_matrix.astype(np.int32), y_pred_matrix, average='macro') * 100.0) if len(y_true_matrix) > 0 else 0.0
    opt_thresholds, opt_metrics = scan_optimal_thresholds(y_true_matrix, y_pred_matrix, full_dataset.label_map)
    error_autopsy = extract_error_autopsy(y_true_matrix, y_pred_matrix, full_dataset.label_map, top_k=5)
    macro_f1_opt = float(np.mean([m["f1_score"] for m in opt_metrics.values()]))

    return {
        "summary_metrics": {"optimized_macro_f1": macro_f1_opt, "mean_average_precision_map": global_map, "best_validation_loss": float(best_val_loss), "total_validation_samples": len(val_loader.dataset)},
        "optimal_class_thresholds": opt_thresholds, "per_class_performance": opt_metrics, "error_autopsy_worst_cases": error_autopsy, "telemetry_history": telemetry_history
    }


def train_roi_diagnostic_engine(device):
    """
    Executes the training pipeline for the ROI Vision Classifier and returns telemetry for the JSON diagnostic report.
    """
    print("\n" + "="*60)
    print("STAGE 2: DIAGNOSTIC TRAINING FOR SPATIAL ROI VISION CLASSIFIER")
    print("="*60)

    if not os.path.exists(ROI_DATA_DIR):
        print(f"Warning: ROI dataset directory '{ROI_DATA_DIR}' not found. Skipping Stage 2!")
        return None

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.05, 0.20), value=0, inplace=False)
    ])
    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_full_dataset = datasets.ImageFolder(root=ROI_DATA_DIR, transform=train_transform)
    val_full_dataset = datasets.ImageFolder(root=ROI_DATA_DIR, transform=val_transform)
    num_classes, total_samples = len(train_full_dataset.classes), len(train_full_dataset)

    os.makedirs(os.path.dirname(ROI_MODEL_SAVE), exist_ok=True)
    with open(ROI_LABEL_MAP, "w") as f:
        json.dump(train_full_dataset.class_to_idx, f, indent=4)

    indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42)).tolist()
    train_size = int(0.8 * total_samples)
    train_loader = DataLoader(torch.utils.data.Subset(train_full_dataset, indices[:train_size]), batch_size=ROI_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(torch.utils.data.Subset(val_full_dataset, indices[train_size:]), batch_size=ROI_BATCH_SIZE, shuffle=False, num_workers=0)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    for param in model.features[:-1].parameters(): param.requires_grad = False
    for param in model.features[-1].parameters(): param.requires_grad = True

    model.classifier[2] = nn.Dropout(p=0.5)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=ROI_LR, weight_decay=ROI_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ROI_EPOCHS, eta_min=1e-6)

    best_val_loss, best_val_acc = float('inf'), 0.0
    telemetry_history = []

    for epoch in range(ROI_EPOCHS):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        scheduler.step()
        epoch_train_loss, epoch_train_acc = train_loss / len(train_loader.dataset), (correct_train / total_train) * 100.0

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss, epoch_val_acc = val_loss / len(val_loader.dataset), (correct_val / total_val) * 100.0
        telemetry_history.append({"epoch": epoch + 1, "train_loss": float(epoch_train_loss), "val_loss": float(epoch_val_loss), "train_acc": float(epoch_train_acc), "val_acc": float(epoch_val_acc)})
        print(f"Epoch [{epoch + 1:02d}/{ROI_EPOCHS:02d}] | Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.2f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss, best_val_acc = epoch_val_loss, epoch_val_acc
            torch.save(model.state_dict(), ROI_MODEL_SAVE)

    print("\nExporting ROI model to ONNX format...")
    model.load_state_dict(torch.load(ROI_MODEL_SAVE))
    model.eval()
    torch.onnx.export(model, torch.randn(1, 3, 224, 224, device=device), ROI_ONNX_SAVE, export_params=True, opset_version=18, do_constant_folding=True, input_names=['input_patch'], output_names=['class_logits'], dynamic_axes={'input_patch': {0: 'batch_size'}, 'class_logits': {0: 'batch_size'}})

    return {
        "summary_metrics": {"best_validation_loss": float(best_val_loss), "best_validation_accuracy": float(best_val_acc), "total_dataset_samples": total_samples, "num_classes": num_classes},
        "class_mapping": train_full_dataset.class_to_idx,
        "telemetry_history": telemetry_history
    }


if __name__ == "__main__":
    active_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Global Diagnostic Suite initialized on hardware device: [{active_device}]")
    
    # Run both training pipelines:
    temporal_diagnostics = train_temporal_diagnostic_engine(active_device)
    roi_diagnostics = train_roi_diagnostic_engine(active_device)

    # Assemble unified diagnostic JSON:
    unified_report = {
        "temporal_action_model_diagnostics": temporal_diagnostics,
        "spatial_roi_classifier_diagnostics": roi_diagnostics
    }

    with open(DIAGNOSTIC_SAVE_PATH, "w") as f:
        json.dump(unified_report, f, indent=4)
