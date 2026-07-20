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
from src.metrics import evaluate_multilabel_metrics

# Temporal Keypoint Model (1D-CNN + BiGRU)
TEMPORAL_BATCH_SIZE = 32
TEMPORAL_LR = 0.001
TEMPORAL_EPOCHS = 60
CSV_PATH = "output/ava_dataset.csv"
TEMPORAL_MODEL_SAVE = "output/model.pth"
TEMPORAL_ONNX_SAVE = "output/model.onnx"
TEMPORAL_LABEL_MAP = "output/label_map.json"

# Spatial Vision Classifier (MobileNetV3-Small)
ROI_BATCH_SIZE = 32
ROI_LR = 0.0003
ROI_WEIGHT_DECAY = 0.01
ROI_EPOCHS = 15
ROI_DATA_DIR = "dataset/roi_train"
ROI_MODEL_SAVE = "output/roi_classifier.pth"
ROI_ONNX_SAVE = "output/roi_classifier.onnx"
ROI_LABEL_MAP = "output/roi_label_map.json"


def train_temporal_engine(device):
    """
    Executes the training and validation pipeline using the Hybrid 1D-CNN + BiGRU architecture.
    Saves the best model checkpoint and exports it to ONNX format for live inference.
    """

    print("\n" + "="*60)
    print("STAGE 1: TRAINING TEMPORAL ACTION MODEL (1D-CNN + BiGRU)")
    print("="*60)

    if not os.path.exists(CSV_PATH):
        print(f"Error: Dataset not found at {CSV_PATH}. Run ingestion first!")
        return

    full_dataset = ActionDataset(CSV_PATH, is_training=False)
    num_classes = len(full_dataset.label_map)

    os.makedirs(os.path.dirname(TEMPORAL_MODEL_SAVE), exist_ok=True)
    with open(TEMPORAL_LABEL_MAP, "w") as f:
        json.dump(full_dataset.label_map, f, indent=4)
    print(f"Saved temporal label mapping to: {TEMPORAL_LABEL_MAP}")

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

    train_loader = DataLoader(train_dataset, batch_size=TEMPORAL_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TEMPORAL_BATCH_SIZE, shuffle=False)

    model = ActionHybridNet(input_size=NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=num_classes).to(device)

    pos_counts = np.maximum(full_dataset.labels.sum(axis=0), 1.0)
    dampened_weights = np.sqrt((len(full_dataset) - pos_counts) / pos_counts)
    pos_weight = torch.tensor(dampened_weights, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TEMPORAL_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TEMPORAL_EPOCHS, eta_min=1e-5)

    best_val_loss = float('inf')
    print("\nStarting Temporal Multi-Label Training Loop...")

    for epoch in range(TEMPORAL_EPOCHS):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * sequences.size(0)
            predicted_binary = (torch.sigmoid(outputs.data) > 0.5).float()
            total_train += labels.numel()
            correct_train += (predicted_binary == labels).sum().item()

        scheduler.step()
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = (correct_train / total_train) * 100.0

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

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

        print(f"Epoch [{epoch + 1:02d}/{TEMPORAL_EPOCHS:02d}] | Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), TEMPORAL_MODEL_SAVE)

    print(f"\nTemporal Training Complete! Best Validation Loss: {best_val_loss:.4f}")
    print("\nExporting Temporal model to ONNX...")
    model.load_state_dict(torch.load(TEMPORAL_MODEL_SAVE))
    model.eval()

    dummy_input = torch.randn(1, WINDOW_SIZE, NUM_FEATURES, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        TEMPORAL_ONNX_SAVE,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_sequence'],
        output_names=['action_logits'],
        dynamic_axes={'input_sequence': {0: 'batch_size'}, 'action_logits': {0: 'batch_size'}}
    )
    print(f"ONNX Temporal Model saved to -> {TEMPORAL_ONNX_SAVE}")

    # Evaluate final metrics:
    all_val_targets, all_val_preds = [], []
    with torch.no_grad():
        for sequences, labels in val_loader:
            probs = torch.sigmoid(model(sequences.to(device))).cpu().numpy()
            all_val_preds.append(probs)
            all_val_targets.append(labels.numpy())

    evaluate_multilabel_metrics(
        y_true=np.vstack(all_val_targets),
        y_pred_probs=np.vstack(all_val_preds),
        label_map=full_dataset.label_map,
        threshold=0.35
    )


def train_roi_engine(device):
    """
    Executes the training and validation pipeline for the Spatial ROI Vision Classifier (MobileNetV3).
    Uses fast Micro-Unfreezing (Block 12 only) and Cutout regularization to prevent memorization.
    """

    print("\n" + "="*60)
    print("STAGE 2: TRAINING SPATIAL ROI VISION CLASSIFIER (MobileNetV3)")
    print("="*60)

    if not os.path.exists(ROI_DATA_DIR):
        print(f"Warning: ROI dataset directory '{ROI_DATA_DIR}' not found. Skipping Stage 2!")
        return

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.05, 0.20), value=0, inplace=False)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_full_dataset = datasets.ImageFolder(root=ROI_DATA_DIR, transform=train_transform)
    val_full_dataset = datasets.ImageFolder(root=ROI_DATA_DIR, transform=val_transform)

    num_classes = len(train_full_dataset.classes)
    total_samples = len(train_full_dataset)

    os.makedirs(os.path.dirname(ROI_MODEL_SAVE), exist_ok=True)
    with open(ROI_LABEL_MAP, "w") as f:
        json.dump(train_full_dataset.class_to_idx, f, indent=4)
    print(f"Saved ROI label mapping to: {ROI_LABEL_MAP}")

    indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42)).tolist()
    train_size = int(0.8 * total_samples)
    train_dataset = torch.utils.data.Subset(train_full_dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(val_full_dataset, indices[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=ROI_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=ROI_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Lock Blocks 0-11, Unfreeze Block 12 only:
    for param in model.features[:-1].parameters():
        param.requires_grad = False
    for param in model.features[-1].parameters():
        param.requires_grad = True

    model.classifier[2] = nn.Dropout(p=0.5)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=ROI_LR, weight_decay=ROI_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ROI_EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    print("\nStarting Fast Micro-Unfreeze ROI Training Loop...")

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
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = (correct_train / total_train) * 100.0

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = (correct_val / total_val) * 100.0

        print(f"Epoch [{epoch + 1:02d}/{ROI_EPOCHS:02d}] | Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), ROI_MODEL_SAVE)

    print(f"\nROI Training Complete! Best Validation Loss: {best_val_loss:.4f}")
    print("\nExporting ROI model to ONNX...")
    model.load_state_dict(torch.load(ROI_MODEL_SAVE))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        ROI_ONNX_SAVE,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_patch'],
        output_names=['class_logits'],
        dynamic_axes={'input_patch': {0: 'batch_size'}, 'class_logits': {0: 'batch_size'}}
    )
    print(f"ONNX ROI Model saved to -> {ROI_ONNX_SAVE}")


if __name__ == "__main__":
    active_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Global Training Suite initialized on hardware device: [{active_device}]")
    
    # Execute dual-stage training:
    train_temporal_engine(active_device)
    train_roi_engine(active_device)
