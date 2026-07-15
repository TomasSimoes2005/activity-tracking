import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from src.shared import WINDOW_SIZE, NUM_FEATURES


class ActionDataset(Dataset):
    """
    PyTorch Dataset for loading skeleton keypoint sequences from CSV with on-the-fly kinematic augmentation.
    """

    def __init__(self, csv_file, label_map=None, is_training=False):
        """
        Constructor. Loads CSV data and encodes text labels into integers.
        :param csv_file: path to the dataset CSV file.
        :param label_map: dictionary mapping string labels to integer IDs. If None, it is generated automatically.
        :param is_training: boolean flag to enable anti-overfitting data augmentation during training passes.
        """

        # Save args:
        self.is_training = is_training

        # Load CSV:
        self.data = pd.read_csv(csv_file)

        # Extract raw text labels and numerical features:
        raw_labels = self.data.iloc[:, 0].values
        raw_features = self.data.iloc[:, 1:].values.astype(np.float32)

        # If a row is purely 0.0, its sum will be 0 (we only keep rows with actual movement/posture):
        row_sums = np.sum(np.abs(raw_features), axis=1)
        valid_indices = row_sums > 1e-5

        # Apply the filter to drop the dead rows from memory:
        filtered_labels = raw_labels[valid_indices]
        self.features = raw_features[valid_indices]

        # Calculate how many garbage rows were removed:
        dropped_count = len(raw_labels) - len(filtered_labels)
        if dropped_count > 0:
            print(f"Dataset Sanitizer: Dropped {dropped_count} empty/invalid skeleton sequences out of {len(raw_labels)}, {len(filtered_labels)} remaining.")

        # Create or use existing label mapping:
        if label_map is None:
            unique_labels = sorted(list(set(filtered_labels)))
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_map = label_map

        # Encode labels to integers:
        self.labels = np.array([self.label_map[lbl] for lbl in filtered_labels], dtype=np.int64)

    def __len__(self):
        """
        Returns the total number of sequence samples in the dataset.
        :return: total number of sequence samples.
        """

        return len(self.labels)

    def _augment_sequence(self, sequence):
        """
        Applies random kinematic jitter, amplitude scaling, and spatial/temporal dropout to aggressively prevent overfitting on small/medium datasets.
        :param sequence: input sequence array of shape [WINDOW_SIZE, NUM_FEATURES].
        :return: augmented sequence array of shape [WINDOW_SIZE, NUM_FEATURES].
        """

        # Random amplitude scaling (simulates different body sizes and camera distances):
        scale_factor = np.random.uniform(0.90, 1.10)
        sequence = sequence * scale_factor

        # Random coordinate jitter (simulates ByteTrack bounding box and sensor noise):
        noise = np.random.normal(0.0, 0.01, sequence.shape).astype(np.float32)
        sequence += noise

        # Temporal masking / frame dropout (50% chance to apply):
        if np.random.rand() < 0.5:
            temporal_mask = np.random.rand(WINDOW_SIZE) > 0.10
            sequence = sequence * temporal_mask[:, np.newaxis]

        # Spatial masking / keypoint dropout (50% chance to apply):
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

        # Reshape flattened features into [WINDOW_SIZE frames, NUM_FEATURES coordinates]:
        sequence = self.features[idx].reshape(WINDOW_SIZE, NUM_FEATURES)

        # Apply augmentation only during training passes:
        if self.is_training:
            sequence = self._augment_sequence(sequence)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


class ActionHybridNet(nn.Module):
    """
    Hybrid 1D-CNN + Bidirectional GRU network for temporal action recognition.
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

        # 1D Temporal Convolution (Extracts sharp velocity/acceleration transitions across neighboring frames):
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(dropout)

        # Bidirectional GRU (Scans forward and backward over extracted motion maps):
        self.bigru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fully connected classification head (hidden_size * 2 due to bidirectional concatenation):
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass.
        :param x: input tensor of shape [batch_size, WINDOW_SIZE, NUM_FEATURES].
        :return: class logits tensor of shape [batch_size, num_classes].
        """

        # Conv1d expects shape [batch_size, channels, sequence_length]:
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout_cnn(x)

        # Restore shape for GRU [batch_size, sequence_length, channels]:
        x = x.transpose(1, 2)

        # Pass through Bidirectional GRU:
        out, _ = self.bigru(x)

        # Extract both forward and backward final temporal states:
        out = out[:, -1, :]

        # Classification head:
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)

        return out
