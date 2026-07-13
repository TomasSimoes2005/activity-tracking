import csv
import os
from collections import defaultdict, deque
from src.shared import WINDOW_SIZE, extract_features


class DatasetWriter:
    """
    Class for writing WINDOW_SIZE-frame temporal keypoint sequences into a CSV file for time-series action models.
    """

    def __init__(self, filename="output/dataset.csv", window_size=WINDOW_SIZE):
        """
        Constructor. Initializes CSV file with headers if not already present.
        :param filename: name of the CSV file.
        :param window_size: size of the temporal keypoint sequences.
        """

        # Save args:
        self.filename = filename
        self.window_size = window_size
        self.track_buffers = defaultdict(lambda: deque(maxlen=window_size))  # Maps track_id to window

        # Create header if it doesn't exist yet:
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                header = ["label"]
                for f_idx in range(self.window_size):
                    for k_idx in range(17):
                        header.extend([f"f{f_idx}_p{k_idx}_x", f"f{f_idx}_p{k_idx}_y"])
                    header.extend([
                        f"f{f_idx}_lw_dist", f"f{f_idx}_rw_dist",
                        f"f{f_idx}_lw_elev", f"f{f_idx}_rw_elev",
                        f"f{f_idx}_prof_ratio", f"f{f_idx}_head_tilt",
                        f"f{f_idx}_inter_wrist", f"f{f_idx}_chest_dist"
                    ])
                writer.writerow(header)

    def _extract_enriched_features(self, keypoints, bbox):
        """
        Calls shared.py's extract_features (compatibility function).
        """

        return extract_features(keypoints, bbox)

    def process_and_save(self, label, track_id, keypoints, bbox):
        """
        Normalizes frame keypoints, appends to the tracked person's rolling buffer and saves a sequence row once the buffer reaches window_size.
        :param label: label of the activity.
        :param track_id: id of the person.
        :param keypoints: keypoint data.
        :param bbox: bounding box data.
        """

        # Get normalized keypoints:
        norm_kpts = self._extract_enriched_features(keypoints, bbox)
        self.track_buffers[track_id].append(norm_kpts)

        # If the buffer reached window size:
        if len(self.track_buffers[track_id]) == self.window_size:

            # Construct row:
            sequence_row = [label]
            for frame_data in self.track_buffers[track_id]:
                sequence_row.extend(frame_data)

            # Write data:
            with open(self.filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(sequence_row)

    def pad_and_flush(self, label, min_frames=10):
        """
        Pads incomplete sequences using Edge Padding (repeating the last known pose), saves them to CSV, and clears the memory buffers.
        :param label: label of the activity.
        :param min_frames: minimum required frames to justify padding and saving.
        """

        # Check all tracked persons in memory:
        for track_id, buffer in list(self.track_buffers.items()):

            # If sequence is shorter than WINDOW_SIZE frames but longer than our minimum quality threshold:
            if min_frames <= len(buffer) < self.window_size:

                # Grab the final posture observed for this person:
                last_frame_pose = buffer[-1]

                # Edge pad: repeat the final posture until we hit window_size (WINDOW_SIZE):
                while len(buffer) < self.window_size:
                    buffer.append(last_frame_pose)

                # Construct row:
                sequence_row = [label]
                for frame_data in buffer:
                    sequence_row.extend(frame_data)

                # Write padded sequence to file:
                with open(self.filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(sequence_row)

        # Clear buffers after flushing:
        self.track_buffers.clear()
