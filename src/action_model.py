import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import defaultdict
from src.shared import WINDOW_SIZE, NUM_FEATURES


class ActionDataset(Dataset):
    """
    PyTorch Dataset for loading skeleton keypoint sequences from CSV with on-the-fly kinematic augmentation.
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
        """
        Returns the total number of sequence samples in the dataset.
        :return: total number of sequence samples.
        """
        return len(self.labels)

    def _augment_sequence(self, sequence):
        """
        Applies random kinematic jitter, amplitude scaling, spatial/temporal dropout, and anatomical horizontal mirroring.
        :param sequence: input sequence array of shape [WINDOW_SIZE, NUM_FEATURES].
        :return: augmented sequence array of shape [WINDOW_SIZE, NUM_FEATURES].
        """

        scale_factor = np.random.uniform(0.90, 1.10)
        sequence = sequence * scale_factor

        noise = np.random.normal(0.0, 0.01, sequence.shape).astype(np.float32)
        sequence += noise

        # 50% Chance for Anatomical Horizontal Mirroring (Instantly doubles training diversity for hand actions!):
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
            sequence = mirrored

        if np.random.rand() < 0.5:
            temporal_mask = np.random.rand(WINDOW_SIZE) > 0.10
            sequence = sequence * temporal_mask[:, np.newaxis]

        if np.random.rand() < 0.5:
            spatial_mask = np.random.rand(NUM_FEATURES) > 0.05
            sequence = sequence * spatial_mask[np.newaxis, :]

        return sequence

    def __getitem__(self, idx):
        """
        Retrieves, reshapes, and optionally augments a single sequence sample.
        :param idx: sample index.
        :return: tuple of (sequence_tensor, label_tensor).
        """
        sequence = self.features[idx].reshape(WINDOW_SIZE, NUM_FEATURES)
        if self.is_training:
            sequence = self._augment_sequence(sequence)

        # Return float32 targets for BCEWithLogitsLoss:
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
        """
        :param gru_output: tensor of shape [batch_size, sequence_length, hidden_size * 2].
        :return: context vector of shape [batch_size, hidden_size * 2].
        """
        attn_weights = torch.softmax(self.attention(gru_output), dim=1)
        context = torch.sum(gru_output * attn_weights, dim=1)
        return context


class ActionHybridNet(nn.Module):
    """
    Hybrid Kinematic 1D-CNN + Bidirectional GRU + Temporal Attention network for action recognition.
    Automatically calculates 1st-order velocity derivatives on-the-fly during the forward pass.
    """

    def __init__(self, input_size=NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=10, dropout=0.3):
        """
        Constructor.
        :param input_size: number of features per frame.
        :param hidden_size: internal GRU and convolutional channel hidden dimension.
        :param num_layers: number of stacked Bidirectional GRU layers.
        :param num_classes: number of output action categories.
        :param dropout: dropout probability to prevent overfitting.
        """

        super(ActionHybridNet, self).__init__()

        # Multiply input_size by 2 because we concatenate raw coordinates with calculated frame-to-frame velocity:
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

        # Temporal Attention Mechanism replaces static end-frame slicing:
        self.attn = TemporalAttention(hidden_size)

        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass with native kinematic derivative calculation.
        :param x: input tensor of shape [batch_size, WINDOW_SIZE, NUM_FEATURES].
        :return: class logits tensor of shape [batch_size, num_classes].
        """

        # 1. Calculate 1st-order temporal velocity natively on GPU: v(t) = p(t) - p(t-1)
        velocity = x[:, 1:, :] - x[:, :-1, :]
        first_frame_vel = torch.zeros_like(x[:, :1, :])
        velocity = torch.cat([first_frame_vel, velocity], dim=1)

        # Concatenate positions and velocities -> Shape: [batch_size, WINDOW_SIZE, NUM_FEATURES * 2]:
        x_dynamic = torch.cat([x, velocity], dim=-1)

        # Conv1d expects shape [batch_size, channels, sequence_length]:
        x_in = x_dynamic.transpose(1, 2)
        x_in = self.conv1d(x_in)
        x_in = self.bn1(x_in)
        x_in = self.relu(x_in)
        x_in = self.dropout_cnn(x_in)

        # Restore shape for GRU [batch_size, sequence_length, channels]:
        x_in = x_in.transpose(1, 2)

        # Pass through Bidirectional GRU:
        out, _ = self.bigru(x_in)

        # Apply Temporal Attention to dynamically weight the most active movement frames:
        out = self.attn(out)

        # Classification head:
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)

        return out
