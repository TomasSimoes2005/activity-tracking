import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.action_model import ActionDataset, ActionHybridNet
from src.shared import WINDOW_SIZE, NUM_FEATURES
from src.metrics import evaluate_multilabel_metrics

# Hyperparameters:
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 60
CSV_PATH = "output/ava_dataset.csv"
MODEL_SAVE_PATH = "output/model.pth"
ONNX_SAVE_PATH = "output/model.onnx"
LABEL_MAP_PATH = "output/label_map.json"


def train():
    """
    Executes the training and validation pipeline using the Hybrid 1D-CNN + BiGRU architecture.
    Saves the best model checkpoint and exports it to ONNX format for live inference.
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

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(full_dataset.label_map, f, indent=4)
    print(f"Saved label mapping to: {LABEL_MAP_PATH}")

    # 80/20 Train-Validation Split using indices:
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

    # Instantiate distinct datasets: Training gets augmentation, Validation is clean:
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

    # Initialize Hybrid Net:
    model = ActionHybridNet(input_size=NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=num_classes).to(device)

    # Calculate class counts:
    pos_counts = full_dataset.labels.sum(axis=0)
    total_samples = len(full_dataset)
    pos_counts = np.maximum(pos_counts, 1.0)

    # Get class weights:
    raw_weights = (total_samples - pos_counts) / pos_counts
    dampened_weights = np.sqrt(raw_weights)
    pos_weight = torch.tensor(dampened_weights, dtype=torch.float32).to(device)
    print("\n--- Dampened Positive Class Weights (Sqrt) ---")
    for name, idx in full_dataset.label_map.items():
        print(f"[{name.upper()}]: {pos_weight[idx]:.2f}x penalty for misses")
    print("-" * 46)

    # Pass pos_weight directly into BCEWithLogitsLoss:
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Cosine Annealing smoothly decays learning rate to prevent plateauing:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_loss = float('inf')
    print("\nStarting Multi-Label Training Loop...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            # Forward pass:
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward pass and optimization:
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping prevents gradient explosion during sharp 1D convolutions:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track metrics using a 0.5 probability threshold:
            train_loss += loss.item() * sequences.size(0)
            predicted_binary = (torch.sigmoid(outputs.data) > 0.5).float()
            total_train += labels.numel()
            correct_train += (predicted_binary == labels).sum().item()

        scheduler.step()

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = (correct_train / total_train) * 100.0

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

        print(f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Bin Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Bin Acc: {epoch_val_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model checkpoint based on lowest validation loss:
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nTraining Complete! Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

    # ONNX export:
    print("\nExporting trained model to ONNX format...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    # Create a dummy input tensor matching our sliding window shape [1 batch, WINDOW_SIZE frames, NUM_FEATURES features]:
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

    print("\nRunning Full Metric Evaluation on Best Saved Checkpoint...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    all_val_targets = []
    all_val_preds = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            
            # Apply sigmoid to extract raw probabilities:
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_val_preds.append(probs)
            all_val_targets.append(labels.numpy())

    # Concatenate batches into full dataset matrices:
    y_true_matrix = np.vstack(all_val_targets)
    y_pred_matrix = np.vstack(all_val_preds)

    # Calculate and print all precision, recall, mAP, and F-scores:
    evaluate_multilabel_metrics(
        y_true=y_true_matrix,
        y_pred_probs=y_pred_matrix,
        label_map=full_dataset.label_map,
        threshold=0.35
    )


if __name__ == "__main__":
    train()
