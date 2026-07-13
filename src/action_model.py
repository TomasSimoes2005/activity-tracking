import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from src.shared import WINDOW_SIZE, NUM_FEATURES


class ActionDataset(Dataset):
    """
    PyTorch Dataset for loading WINDOW_SIZE-frame skeleton keypoint sequences from CSV.
    """

    def __init__(self, csv_file, label_map=None):
        """
        Constructor. Loads CSV data and encodes text labels into integers.
        :param csv_file: path to the dataset CSV file.
        :param label_map: dictionary mapping string labels to integer IDs. If None, it is generated automatically.
        """

        # Load CSV:
        self.data = pd.read_csv(csv_file)

        # Extract raw text labels and numerical features:
        raw_labels = self.data.iloc[:, 0].values
        self.features = self.data.iloc[:, 1:].values.astype(np.float32)

        # Create or use existing label mapping:
        if label_map is None:
            unique_labels = sorted(list(set(raw_labels)))
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_map = label_map

        # Encode labels to integers:
        self.labels = np.array([self.label_map[lbl] for lbl in raw_labels], dtype=np.int64)

    def __len__(self):
        """
        :return: total number of sequence samples.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves and reshapes a single sequence sample.
        :param idx: sample index.
        :return: tuple of (sequence_tensor, label_tensor).
        """

        # Reshape flattened 1020 features into [WINDOW_SIZE frames, NUM_FEATURES coordinates]:
        sequence = self.features[idx].reshape(WINDOW_SIZE, NUM_FEATURES)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


class ActionGRU(nn.Module):
    """
    Lightweight 3-layer Gated Recurrent Unit (GRU) for temporal action recognition.
    """

    def __init__(self, input_size=NUM_FEATURES, hidden_size=64, num_layers=3, num_classes=10, dropout=0.2):
        """
        Constructor.
        :param input_size: number of features per frame.
        :param hidden_size: internal GRU hidden state dimension.
        :param num_layers: number of stacked GRU layers.
        :param num_classes: number of output action categories.
        :param dropout: dropout probability to prevent overfitting.
        """
        super(ActionGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stacked GRU layers:
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fully connected classification heads:
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        """
        Forward pass.
        :param x: input tensor of shape [batch_size, WINDOW_SIZE, NUM_FEATURES].
        :return: class logits of shape [batch_size, num_classes].
        """

        # Initialize hidden state with zeros:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass through GRU (out shape: [batch_size, WINDOW_SIZE, hidden_size]):
        out, _ = self.gru(x, h0)

        # Decode only the final time step (out[:, -1, :]):
        out = out[:, -1, :]

        # Pass through classification head:
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
