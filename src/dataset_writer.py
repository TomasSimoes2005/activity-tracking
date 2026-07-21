import csv
import os
from collections import defaultdict, deque
from src.shared import WINDOW_SIZE, extract_features


class DatasetWriter:
    """
    Class for writing temporal keypoint sequences into a CSV file for time-series action models.
    """

    def __init__(self, filename="output/hmdb51.csv", window_size=WINDOW_SIZE):
        """
        Constructor. Initializes CSV file with headers if not already present.
        :param filename: name of the CSV file.
        :param window_size: size of the temporal keypoint sequences.
        """

        # Save args:
        self.filename = filename
        self.window_size = window_size
        self.track_buffers = defaultdict(lambda: deque(maxlen=window_size))  # Maps track_id to window

        # Ensure output directory exists to prevent FileNotFoundError:
        output_dir = os.path.dirname(self.filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Create header if it doesn't exist yet:
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                header = ["label"]
                for f_idx in range(self.window_size):
                    # 34 normalized coordinate features:
                    for k_idx in range(17):
                        header.extend([f"f{f_idx}_p{k_idx}_x", f"f{f_idx}_p{k_idx}_y"])
                    # 8 orientation and kinematic features:
                    header.extend([
                        f"f{f_idx}_lw_dist", f"f{f_idx}_rw_dist",
                        f"f{f_idx}_lw_elev", f"f{f_idx}_rw_elev",
                        f"f{f_idx}_prof_ratio", f"f{f_idx}_head_tilt",
                        f"f{f_idx}_inter_wrist", f"f{f_idx}_chest_dist",
                        f"f{f_idx}_lw_lear", f"f{f_idx}_rw_rear",
                        f"f{f_idx}_lw_rear", f"f{f_idx}_rw_lear"
                    ])
                writer.writerow(header)

    def _extract_enriched_features(self, keypoints, bbox):
        """
        Extracts 46 features per frame: 34 anatomically normalized coordinates + 12 orientation/kinematic metrics.
        :param keypoints: keypoint data.
        :param bbox: bounding box data.
        :return: enriched feature list of length 46.
        """

        return extract_features(keypoints, bbox)

    def process_and_save(self, label, track_id, keypoints, bbox):
        """
        Extracts enriched features per frame, appends to the buffer, and saves a sequence row once window_size is reached.
        :param label: label of the activity.
        :param track_id: id of the person.
        :param keypoints: keypoint data.
        :param bbox: bounding box data.
        """

        # Get enriched features:
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
        Pads incomplete sequences using Edge Padding, saves them to CSV, and clears the buffers.
        :param label: label of the activity.
        :param min_frames: minimum required frames to justify padding and saving.
        """

        # Check all tracked persons in memory:
        for track_id, buffer in list(self.track_buffers.items()):

            # If sequence is shorter than window_size but longer than our minimum quality threshold:
            if min_frames <= len(buffer) < self.window_size:

                # Grab the final posture observed for this person:
                last_frame_pose = buffer[-1]

                # Edge pad: repeat the final posture until we hit window_size:
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
