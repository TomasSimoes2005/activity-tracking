import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.action_model import ActionDataset, ActionHybridNet
from src.shared import WINDOW_SIZE, NUM_FEATURES
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

# Hyperparameters:
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 60
CSV_PATH = "output/ava_dataset.csv"
MODEL_SAVE_PATH = "output/model.pth"
ONNX_SAVE_PATH = "output/model.onnx"
LABEL_MAP_PATH = "output/label_map.json"
DIAGNOSTIC_SAVE_PATH = "output/diagnostic_report.json"


def scan_optimal_thresholds(y_true, y_pred_probs, label_map):
    """
    Scans decision thresholds from 0.10 to 0.90 independently for each class to find the exact boundary that maximizes F1-Score.
    :param y_true: ground truth binary matrix of shape [N, num_classes].
    :param y_pred_probs: predicted sigmoid probabilities matrix of shape [N, num_classes].
    :param label_map: dictionary mapping string labels to integer IDs.
    :return: tuple of (dictionary of optimal thresholds per class, dictionary of resulting metrics).
    """

    optimal_thresholds = {}
    optimal_metrics = {}
    threshold_range = np.linspace(0.10, 0.90, 81)

    print("\n--- Running Automated Per-Class Threshold Optimization ---")
    for label_str, idx in sorted(label_map.items(), key=lambda item: item[1]):
        best_f1 = 0.0
        best_thresh = 0.50
        best_prec = 0.0
        best_rec = 0.0

        class_true = y_true[:, idx]
        class_probs = y_pred_probs[:, idx]

        # Scan all candidate thresholds for this specific action class:
        for thresh in threshold_range:
            class_pred = (class_probs >= thresh).astype(np.int32)
            prec, rec, f1, _ = precision_recall_fscore_support(class_true, class_pred, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_prec = prec
                best_rec = rec

        optimal_thresholds[label_str.upper()] = float(best_thresh)
        optimal_metrics[label_str.upper()] = {
            "threshold": float(best_thresh),
            "precision": float(best_prec * 100.0),
            "recall": float(best_rec * 100.0),
            "f1_score": float(best_f1 * 100.0),
            "support": int(np.sum(class_true))
        }
        print(f"[{label_str.upper():<11}] -> Optimal Thresh: {best_thresh:.2f} | Prec: {best_prec*100:5.1f}% | Rec: {best_rec*100:5.1f}% | F1: {best_f1*100:5.1f}%")

    print("-" * 60)
    return optimal_thresholds, optimal_metrics


def extract_error_autopsy(y_true, y_pred_probs, label_map, top_k=5):
    """
    Identifies and extracts the worst False Positives and False Negatives for each class to diagnose feature blindspots.
    :param y_true: ground truth binary matrix of shape [N, num_classes].
    :param y_pred_probs: predicted sigmoid probabilities matrix of shape [N, num_classes].
    :param label_map: dictionary mapping string labels to integer IDs.
    :param top_k: number of extreme error samples to extract per class.
    :return: dictionary mapping each class to its worst error profiles.
    """

    autopsy_report = {}
    for label_str, idx in sorted(label_map.items(), key=lambda item: item[1]):
        class_true = y_true[:, idx]
        class_probs = y_pred_probs[:, idx]

        # False Positives: Ground truth is 0, but predicted probability is extremely high:
        fp_indices = np.where(class_true == 0.0)[0]
        fp_sorted = fp_indices[np.argsort(class_probs[fp_indices])[::-1]][:top_k]
        fp_data = [{"sample_idx": int(i), "confidence": float(class_probs[i] * 100.0)} for i in fp_sorted if class_probs[i] > 0.30]

        # False Negatives: Ground truth is 1, but predicted probability is extremely low:
        fn_indices = np.where(class_true == 1.0)[0]
        fn_sorted = fn_indices[np.argsort(class_probs[fn_indices])][:top_k]
        fn_data = [{"sample_idx": int(i), "confidence": float(class_probs[i] * 100.0)} for i in fn_sorted if class_probs[i] < 0.70]

        autopsy_report[label_str.upper()] = {
            "top_false_positives": fp_data,
            "top_false_negatives": fn_data
        }

    return autopsy_report


def train_and_diagnose():
    """
    Executes the multi-label training pipeline, tracks gradient telemetry, optimizes decision thresholds,
    and exports a comprehensive JSON diagnostic report for architectural and feature debugging.
    """

    # Check device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: [{device}]")

    # Base dataset check to extract class mappings:
    if not os.path.exists(CSV_PATH):
        print(f"Error: Dataset not found at {CSV_PATH}. Run ingestion first!")
        return
    full_dataset = ActionDataset(CSV_PATH, is_training=False)
    num_classes = len(full_dataset.label_map)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(full_dataset.label_map, f, indent=4)
    print(f"Saved label mapping to: {LABEL_MAP_PATH}")

    # Calculate dampened square-root pos_weight to balance precision and recall:
    pos_counts = full_dataset.labels.sum(axis=0)
    total_samples = len(full_dataset)
    pos_counts = np.maximum(pos_counts, 1.0)
    raw_weights = (total_samples - pos_counts) / pos_counts
    dampened_weights = np.sqrt(raw_weights)
    pos_weight = torch.tensor(dampened_weights, dtype=torch.float32).to(device)

    print("\n--- Dampened Positive Class Weights (Sqrt) ---")
    for name, idx in full_dataset.label_map.items():
        print(f"[{name.upper():<11}]: {pos_weight[idx]:.2f}x penalty for misses")
    print("-" * 46)

    # 80/20 Train-Validation Split using indices:
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

    train_dataset = torch.utils.data.Subset(
        ActionDataset(CSV_PATH, label_map=full_dataset.label_map, is_training=True), 
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        ActionDataset(CSV_PATH, label_map=full_dataset.label_map, is_training=False), 
        val_indices
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Hybrid Net, Weighted BCE Loss, and Optimizer:
    model = ActionHybridNet(input_size=NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_loss = float('inf')
    telemetry_history = []
    print("\nStarting Multi-Label Diagnostic Training Loop...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_grad_norms = []

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()

            # Track gradient norm before clipping to diagnose gradient explosion/vanishing:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            epoch_grad_norms.append(total_norm)

            # Gradient clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * sequences.size(0)
            predicted_binary = (torch.sigmoid(outputs.data) > 0.5).float()
            total_train += labels.numel()
            correct_train += (predicted_binary == labels).sum().item()

        scheduler.step()
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = (correct_train / total_train) * 100.0
        avg_grad_norm = float(np.mean(epoch_grad_norms))

        # Validation pass:
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * sequences.size(0)
                predicted_binary = (torch.sigmoid(outputs.data) > 0.5).float()
                total_val += labels.numel()
                correct_val += (predicted_binary == labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = (correct_val / total_val) * 100.0

        # Log epoch telemetry:
        telemetry_history.append({
            "epoch": epoch + 1,
            "train_loss": float(epoch_train_loss),
            "val_loss": float(epoch_val_loss),
            "train_bin_acc": float(epoch_train_acc),
            "val_bin_acc": float(epoch_val_acc),
            "grad_norm": float(avg_grad_norm),
            "lr": float(scheduler.get_last_lr()[0])
        })

        print(f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.2f}% | "
              f"GradNorm: {avg_grad_norm:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint based on lowest validation loss:
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nTraining Complete! Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

    # Export ONNX model:
    print("\nExporting trained model to ONNX format...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    dummy_input = torch.randn(1, WINDOW_SIZE, NUM_FEATURES, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_SAVE_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_sequence'],
        output_names=['action_logits'],
        dynamic_axes={'input_sequence': {0: 'batch_size'}, 'action_logits': {0: 'batch_size'}}
    )
    print(f"ONNX Model successfully exported to: {ONNX_SAVE_PATH}")

    # Run Deep Diagnostic Evaluation:
    print("\nExecuting Deep Diagnostic Evaluation on Validation Dataset...")
    all_val_targets = []
    all_val_preds = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_val_preds.append(probs)
            all_val_targets.append(labels.numpy())

    y_true_matrix = np.vstack(all_val_targets)
    y_pred_matrix = np.vstack(all_val_preds)

    # Calculate global mAP:
    try:
        global_map = float(average_precision_score(y_true_matrix.astype(np.int32), y_pred_matrix, average='macro') * 100.0)
    except ValueError:
        global_map = 0.0

    # 1. Scan for optimal per-class decision thresholds:
    opt_thresholds, opt_metrics = scan_optimal_thresholds(y_true_matrix, y_pred_matrix, full_dataset.label_map)

    # 2. Perform error autopsy:
    error_autopsy = extract_error_autopsy(y_true_matrix, y_pred_matrix, full_dataset.label_map, top_k=5)

    # Calculate macro F1 using optimal thresholds:
    macro_f1_opt = float(np.mean([m["f1_score"] for m in opt_metrics.values()]))

    print(f"\nFINAL OPTIMIZED MACRO F1-SCORE : {macro_f1_opt:.2f}% (Using Per-Class Thresholds)")
    print(f"FINAL MEAN AVERAGE PRECISION : {global_map:.2f}%\n")

    # Assemble unified diagnostic report:
    diagnostic_report = {
        "summary_metrics": {
            "optimized_macro_f1": macro_f1_opt,
            "mean_average_precision_map": global_map,
            "best_validation_loss": float(best_val_loss),
            "total_validation_samples": len(val_dataset),
            "num_features": NUM_FEATURES,
            "window_size": WINDOW_SIZE
        },
        "optimal_class_thresholds": opt_thresholds,
        "per_class_performance": opt_metrics,
        "error_autopsy_worst_cases": error_autopsy,
        "telemetry_history": telemetry_history
    }

    with open(DIAGNOSTIC_SAVE_PATH, "w") as f:
        json.dump(diagnostic_report, f, indent=4)
    print(f"Comprehensive Diagnostic Report saved to -> {DIAGNOSTIC_SAVE_PATH}")


if __name__ == "__main__":
    train_and_diagnose()
