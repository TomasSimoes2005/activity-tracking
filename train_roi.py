import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Hyperparameters:
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.01
EPOCHS = 15
DATA_DIR = "dataset/roi_train"
MODEL_SAVE_PATH = "output/roi_classifier.pth"
ONNX_SAVE_PATH = "output/roi_classifier.onnx"
LABEL_MAP_PATH = "output/roi_label_map.json"


def train_and_export_roi():
    """
    Executes an 80/20 train-validation fine-tuning loop for MobileNetV3-Small on cropped interaction patches.
    Uses Micro-Unfreezing (Block 12 only), Heavy Dropout, and Random Erasing (Cutout) to maximize generalization.
    """

    # Check hardware device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training ROI Classifier on device: [{device}]")

    # Verify dataset directory exists:
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found. Run harvest_roi_dataset.py first!")
        return

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Randomly drops a black box over 5% to 20% of the image to stop background memorization:
        transforms.RandomErasing(p=0.5, scale=(0.05, 0.20), value=0, inplace=False)
    ])

    # Validation transformations (clean scaling and normalization only):
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full datasets in memory with separated transform pipelines:
    train_full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    val_full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)

    num_classes = len(train_full_dataset.classes)
    total_samples = len(train_full_dataset)
    print(f"Dataset assembled! Found {total_samples} total images across {num_classes} classes: {train_full_dataset.class_to_idx}")

    # Export clean class-to-index label mapping for the ONNX inference engine:
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(train_full_dataset.class_to_idx, f, indent=4)
    print(f"Saved ROI label mapping to: {LABEL_MAP_PATH}")

    # 80/20 Train-Validation Split using deterministic indices:
    indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42)).tolist()
    train_size = int(0.8 * total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Instantiate subsets using identical index slices:
    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices)

    # In-process data loading for maximum CPU speed:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Data Split -> Training Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}")

    # Initialize pretrained MobileNetV3-Small:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Slashes CPU training time and prevents background memorization:
    for param in model.features[:-1].parameters():
        param.requires_grad = False
    for param in model.features[-1].parameters():
        param.requires_grad = True
    print("Locked Blocks 0-11. Unfroze only final semantic block (Block 12)...")

    model.classifier[2] = nn.Dropout(p=0.5)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # Loss with label smoothing to curb overconfidence on blurry crops:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Pass only unfrozen parameters to the optimizer:
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    print("\nStarting Fast Micro-Unfreeze ROI Training Loop...")

    # Training loop:
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to maintain stability:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        scheduler.step()
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation pass:
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

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

        print(f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint based on lowest validation loss:
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nTraining Complete! Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

    # Export best checkpoint to ONNX format:
    print("\nLoading best weights and exporting to ONNX format...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_SAVE_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_patch'],
        output_names=['class_logits'],
        dynamic_axes={'input_patch': {0: 'batch_size'}, 'class_logits': {0: 'batch_size'}}
    )
    print(f"ONNX ROI Model successfully exported to: {ONNX_SAVE_PATH}")


if __name__ == "__main__":
    train_and_export_roi()