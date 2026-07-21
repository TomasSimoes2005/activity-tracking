import os
import json
import numpy as np
from collections import defaultdict, deque
from src.shared import WINDOW_SIZE, NUM_FEATURES, extract_features

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class ActionPredictor:
    """
    Class for real-time temporal action recognition using an exported ONNX model.
    """

    def __init__(self, model_path="output/model.onnx", label_map_path="output/label_map.json", window_size=WINDOW_SIZE):
        """
        Constructor. Loads the ONNX runtime session and label mapping.
        :param model_path: path to the exported ONNX model file.
        :param label_map_path: path to the JSON file mapping class names to integer IDs.
        :param window_size: size of the temporal keypoint sequence required for inference.
        """

        # Save args:
        self.window_size = window_size
        self.track_buffers = defaultdict(lambda: deque(maxlen=window_size))
        self.track_labels = {}  # Caches the latest predicted action string per track_id

        # Verify ONNX Runtime installation:
        if ort is None:
            print("Error: `onnxruntime` is not installed. Please run: pip install onnxruntime")
            self.session = None
            return

        # Check if model files exist:
        if not os.path.exists(model_path) or not os.path.exists(label_map_path):
            print(f"Warning: Model ({model_path}) or Label Map ({label_map_path}) not found. Live inference disabled.")
            self.session = None
            return

        # Load label mapping and invert it (ID -> Label String):
        with open(label_map_path, "r") as f:
            label_map = json.load(f)
        self.idx_to_label = {int(idx): label.upper() for label, idx in label_map.items()}

        # Initialize ONNX Runtime Inference Session:
        print(f"Loading ONNX Action Recognition model from: {model_path}...")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        print("Action Recognition engine online!")

    def _extract_enriched_features(self, keypoints, bbox):
        """
        Extracts 46 features per frame: 34 anatomically normalized coordinates + 12 orientation/kinematic metrics.
        :param keypoints: keypoint data.
        :param bbox: bounding box data.
        :return: enriched feature list of length 46.
        """

        return extract_features(keypoints, bbox)
    
    def get_raw_probabilities(self, track_id, keypoints, bbox, is_last_frame=False, min_frames=10):
            """
            Returns the raw Sigmoid probability vector and class map for advanced blending math.
            """

            if self.session is None:
                return None, None

            norm_kpts = self._extract_enriched_features(keypoints, bbox)
            self.track_buffers[track_id].append(norm_kpts)

            if is_last_frame and min_frames <= len(self.track_buffers[track_id]) < self.window_size:
                last_pose = self.track_buffers[track_id][-1]
                while len(self.track_buffers[track_id]) < self.window_size:
                    self.track_buffers[track_id].append(last_pose)

            if len(self.track_buffers[track_id]) == self.window_size:
                input_seq = np.array(self.track_buffers[track_id], dtype=np.float32).reshape(1, self.window_size, NUM_FEATURES)
                logits = self.session.run(None, {self.input_name: input_seq})[0][0]
                probabilities = 1.0 / (1.0 + np.exp(-logits))
                return probabilities, self.idx_to_label

            return f"BUFFERING ({len(self.track_buffers[track_id])}/{self.window_size})", None

    def predict(self, track_id, keypoints, bbox):
        """
        Normalizes frame keypoints, updates the person's rolling buffer, and runs ONNX inference once a full window is accumulated.
        :param track_id: id of the person.
        :param keypoints: keypoint data.
        :param bbox: bounding box data.
        :return: string label of the predicted action (or buffering status).
        """

        # If engine failed to initialize:
        if self.session is None:
            return "NO MODEL"

        # Get enriched features and append to memory buffer:
        norm_kpts = self._extract_enriched_features(keypoints, bbox)
        self.track_buffers[track_id].append(norm_kpts)

        # If we have accumulated a full window of 30 frames:
        if len(self.track_buffers[track_id]) == self.window_size:

            # Format input array to shape [1 batch, WINDOW_SIZE frames, NUM_FEATURES coordinates]:
            input_sequence = np.array(self.track_buffers[track_id], dtype=np.float32).reshape(1, self.window_size, NUM_FEATURES)

            # Run inference:
            logits = self.session.run(None, {self.input_name: input_sequence})[0]

            # Get class ID with highest confidence:
            pred_idx = np.argmax(logits, axis=1)[0]

            # Cache and return the corresponding text label:
            action_label = self.idx_to_label.get(pred_idx, "UNKNOWN")
            self.track_labels[track_id] = action_label
            return action_label

        # If buffer is still filling up (< 30 frames), return cached label or buffering status:
        return self.track_labels.get(track_id, f"BUFFERING ({len(self.track_buffers[track_id])}/{self.window_size})")

    def predict_top_k(self, track_id, keypoints, bbox, k=5, is_last_frame=False, min_frames=10):
        """
        Normalizes frame keypoints, updates the rolling buffer, and returns the top K action predictions using Sigmoid probabilities.
        If is_last_frame is True and the buffer has between min_frames and window_size, it applies Edge Padding.
        :param track_id: id of the person.
        :param keypoints: keypoint data.
        :param bbox: bounding box data.
        :param k: number of top predictions to return.
        :param is_last_frame: boolean flag indicating if this is the final frame of the video/clip.
        :param min_frames: minimum required frames to justify padding and predicting.
        :return: list of tuples [(label_str, probability_float), ...] or buffering status string.
        """

        # If engine failed to initialize:
        if self.session is None:
            return "NO MODEL"

        # Get enriched features and append to memory buffer:
        norm_kpts = self._extract_enriched_features(keypoints, bbox)
        self.track_buffers[track_id].append(norm_kpts)

        # If this is the end of the clip, and we don't have 30 frames yet, but have at least min_frames:
        if is_last_frame and min_frames <= len(self.track_buffers[track_id]) < self.window_size:
            last_pose = self.track_buffers[track_id][-1]
            while len(self.track_buffers[track_id]) < self.window_size:
                self.track_buffers[track_id].append(last_pose)

        # If we have accumulated a full window of 30 frames (either naturally or via Edge Padding):
        if len(self.track_buffers[track_id]) == self.window_size:

            # Format input array to shape [1 batch, WINDOW_SIZE frames, NUM_FEATURES coordinates]:
            input_sequence = np.array(self.track_buffers[track_id], dtype=np.float32).reshape(1, self.window_size, NUM_FEATURES)

            # Run ONNX inference:
            logits = self.session.run(None, {self.input_name: input_sequence})[0][0]

            # Compute independent Sigmoid probabilities for multi-label inference:
            probabilities = 1.0 / (1.0 + np.exp(-logits))

            # Get indices of the top K highest probabilities (sorted descending):
            top_k_indices = np.argsort(probabilities)[::-1][:k]

            # Construct list of (Label, Probability) tuples:
            top_k_results = [
                (self.idx_to_label.get(idx, "UNKNOWN"), float(probabilities[idx]))
                for idx in top_k_indices
            ]

            # Cache and return results:
            self.track_labels[track_id] = top_k_results
            return top_k_results

        # If buffer is still filling up (< 30 frames), return cached list or buffering string:
        return self.track_labels.get(track_id, f"BUFFERING ({len(self.track_buffers[track_id])}/{self.window_size})")

    def predict_multi_label(self, track_id, keypoints, bbox, default_threshold=0.40, custom_thresholds=None, is_last_frame=False, min_frames=10):
        """
        Normalizes frame keypoints, updates rolling buffer, and returns ALL simultaneous actions exceeding per-class thresholds.
        :param track_id: id of the person.
        :param keypoints: keypoint data.
        :param bbox: bounding box data.
        :param default_threshold: fallback minimum sigmoid probability if custom threshold is missing.
        :param custom_thresholds: dictionary mapping class label strings to specific optimal float thresholds.
        :param is_last_frame: boolean flag indicating if this is the final frame of the video/clip.
        :param min_frames: minimum required frames to justify padding and predicting.
        :return: list of tuples [(label_str, probability_float), ...] sorted by confidence, or buffering status string.
        """

        if self.session is None:
            return "NO MODEL"

        norm_kpts = self._extract_enriched_features(keypoints, bbox)
        self.track_buffers[track_id].append(norm_kpts)

        if is_last_frame and min_frames <= len(self.track_buffers[track_id]) < self.window_size:
            last_pose = self.track_buffers[track_id][-1]
            while len(self.track_buffers[track_id]) < self.window_size:
                self.track_buffers[track_id].append(last_pose)

        if len(self.track_buffers[track_id]) == self.window_size:
            input_sequence = np.array(self.track_buffers[track_id], dtype=np.float32).reshape(1, self.window_size, NUM_FEATURES)
            logits = self.session.run(None, {self.input_name: input_sequence})[0][0]

            probabilities = 1.0 / (1.0 + np.exp(-logits))

            active_actions = []
            for idx, prob in enumerate(probabilities):
                label_str = self.idx_to_label.get(idx, "UNKNOWN")
                
                # Check if we have an optimal custom threshold for this specific class:
                thresh = default_threshold
                if custom_thresholds and label_str in custom_thresholds:
                    thresh = custom_thresholds[label_str]

                if prob >= thresh:
                    active_actions.append((label_str, float(prob)))

            active_actions.sort(key=lambda x: x[1], reverse=True)

            if not active_actions:
                best_idx = np.argmax(probabilities)
                active_actions = [(self.idx_to_label.get(best_idx, "UNKNOWN"), float(probabilities[best_idx]))]

            self.track_labels[track_id] = active_actions
            return active_actions

        return self.track_labels.get(track_id, f"BUFFERING ({len(self.track_buffers[track_id])}/{self.window_size})")
