import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict
from src.shared import WINDOW_SIZE, NUM_FEATURES


class ActionDataset(Dataset):
    """
    PyTorch Dataset for loading skeleton keypoint sequences from CSV with advanced non-linear temporal augmentation.
    Aggregates simultaneous action annotations into multi-hot binary vectors for multi-label classification.
    """

    def __init__(self, csv_file, label_map=None, is_training=False):
        """
        Constructor. Loads CSV data, aggregates simultaneous actions, and encodes text labels into multi-hot vectors.
        :param csv_file: path to the dataset CSV file.
        :param label_map: dictionary mapping string labels to integer IDs. If None, it is generated automatically.
        :param is_training: boolean flag to enable anti-overfitting data augmentation during training passes.
        """

        self.is_training = is_training
        self.data = pd.read_csv(csv_file)

        raw_labels = self.data.iloc[:, 0].values
        raw_features = self.data.iloc[:, 1:].values.astype(np.float32)

        row_sums = np.sum(np.abs(raw_features), axis=1)
        valid_indices = row_sums > 1e-5

        filtered_labels = raw_labels[valid_indices]
        filtered_features = raw_features[valid_indices]

        dropped_count = len(raw_labels) - len(filtered_labels)
        if dropped_count > 0:
            print(f"Dataset Sanitizer: Dropped {dropped_count} empty/invalid skeleton sequences out of {len(raw_labels)}.")

        if label_map is None:
            unique_labels = sorted(list(set(filtered_labels)))
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_map = label_map

        # Aggregate simultaneous action labels for identical timestamp windows:
        sequence_groups = defaultdict(lambda: {"features": None, "labels": set()})
        for lbl, feats in zip(filtered_labels, filtered_features):
            if lbl in self.label_map:
                feat_hash = feats.tobytes()
                sequence_groups[feat_hash]["features"] = feats
                sequence_groups[feat_hash]["labels"].add(self.label_map[lbl])

        unique_features = []
        multi_hot_labels = []
        for group in sequence_groups.values():
            unique_features.append(group["features"])
            multi_hot = np.zeros(len(self.label_map), dtype=np.float32)
            for idx in group["labels"]:
                multi_hot[idx] = 1.0
            multi_hot_labels.append(multi_hot)

        self.features = np.array(unique_features, dtype=np.float32)
        self.labels = np.array(multi_hot_labels, dtype=np.float32)
        print(f"Dataset Sanitizer: Assembled {len(self.features)} unique multi-label sequences across {len(self.label_map)} classes.")

    def __len__(self):
        return len(self.labels)

    def _augment_sequence(self, sequence):
        """
        Applies random kinematic jitter, temporal elasticity (time warping), and anatomical mirroring.
        :param sequence: input sequence array of shape [WINDOW_SIZE, NUM_FEATURES].
        :return: augmented sequence array of shape [WINDOW_SIZE, NUM_FEATURES].
        """

        scale_factor = np.random.uniform(0.95, 1.05)
        sequence = sequence * scale_factor
        noise = np.random.normal(0.0, 0.005, sequence.shape).astype(np.float32)
        sequence += noise

        # Stretches and compresses the action sequence to make the network robust to different motion speeds:
        if np.random.rand() < 0.3:
            orig_indices = np.arange(WINDOW_SIZE)
            warp_magnitude = np.random.uniform(1.1, 2.0)
            
            # Create a non-linear curve to warp the time axis:
            warp_curve = orig_indices + np.sin(orig_indices * np.pi / WINDOW_SIZE) * warp_magnitude
            warp_curve = np.clip(warp_curve, 0, WINDOW_SIZE - 1)
            warp_curve = np.sort(warp_curve)

            warped_seq = np.zeros_like(sequence)
            for i in range(NUM_FEATURES):
                warped_seq[:, i] = np.interp(orig_indices, warp_curve, sequence[:, i])
            sequence = warped_seq

        # Anatomical Horizontal Mirroring:
        if np.random.rand() < 0.5:
            mirrored = sequence.copy()
            mirrored[:, 0:34:2] = -mirrored[:, 0:34:2]
            swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            for left_idx, right_idx in swap_pairs:
                lx, ly = left_idx * 2, (left_idx * 2) + 1
                rx, ry = right_idx * 2, (right_idx * 2) + 1
                mirrored[:, [lx, rx]] = mirrored[:, [rx, lx]]
                mirrored[:, [ly, ry]] = mirrored[:, [ry, ly]]
            mirrored[:, [34, 35]] = mirrored[:, [35, 34]]
            mirrored[:, [36, 37]] = mirrored[:, [37, 36]]
            mirrored[:, [42, 43]] = mirrored[:, [43, 42]]
            mirrored[:, [44, 45]] = mirrored[:, [45, 44]]
            sequence = mirrored

        if np.random.rand() < 0.2:
            temporal_mask = np.random.rand(WINDOW_SIZE) > 0.05
            sequence = sequence * temporal_mask[:, np.newaxis]

        if np.random.rand() < 0.2:
            spatial_mask = np.random.rand(NUM_FEATURES) > 0.02
            sequence = sequence * spatial_mask[np.newaxis, :]

        return sequence

    def __getitem__(self, idx):
        sequence = self.features[idx].reshape(WINDOW_SIZE, NUM_FEATURES)
        if self.is_training:
            sequence = self._augment_sequence(sequence)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)


class TemporalAttention(nn.Module):
    """
    Lightweight Temporal Self-Attention layer. Automatically weights critical movement frames over static background frames.
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, gru_output):
        attn_weights = torch.softmax(self.attention(gru_output), dim=1)
        context = torch.sum(gru_output * attn_weights, dim=1)
        return context


class ActionHybridNet(nn.Module):
    """
    Hybrid Kinematic 1D-CNN + Bidirectional GRU + Temporal Attention network for action recognition.
    Safely calculates and masks kinematic derivatives to prevent YOLO outlier spikes.
    """

    def __init__(self, input_size=NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=10, dropout=0.3):
        super(ActionHybridNet, self).__init__()

        # Temporal Smoother to kill minor YOLO keypoint jitter:
        self.smoother = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)

        self.conv1d = nn.Conv1d(in_channels=input_size * 2, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(dropout)

        self.bigru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.attn = TemporalAttention(hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass with YOLO Outlier Rejection.
        """
        # 1. Create a binary validity mask. Missing YOLO points manifest as exactly 0.0
        # This mask prevents dropped frames from generating fake velocity spikes.
        valid_mask = (torch.abs(x) > 1e-5).float()

        # 2. Smooth the raw sequence:
        x_smooth = self.smoother(x.transpose(1, 2)).transpose(1, 2)

        # 3. Calculate raw velocity: v(t) = p(t) - p(t-1)
        velocity = x_smooth[:, 1:, :] - x_smooth[:, :-1, :]
        first_frame_vel = torch.zeros_like(x_smooth[:, :1, :])
        velocity = torch.cat([first_frame_vel, velocity], dim=1)

        # 4. ZERO-MASKING: Flatten artificial velocity spikes caused by dropped keypoints:
        velocity = velocity * valid_mask

        # Concatenate positions and clean velocities -> Shape: [batch, WINDOW_SIZE, NUM_FEATURES * 2]:
        x_dynamic = torch.cat([x, velocity], dim=-1)

        x_in = x_dynamic.transpose(1, 2)
        x_in = self.conv1d(x_in)
        x_in = self.bn1(x_in)
        x_in = self.relu(x_in)
        x_in = self.dropout_cnn(x_in)

        x_in = x_in.transpose(1, 2)

        out, _ = self.bigru(x_in)
        out = self.attn(out)

        out = self.fc1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)

        return out
