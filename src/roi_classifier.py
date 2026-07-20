import os
import cv2
import json
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class ROIClassifier:
    """
    Class for classifying object interaction inside cropped hand-to-face bounding boxes using an exported ONNX vision model.
    """

    def __init__(self, model_path="output/roi_classifier.onnx", label_map_path="output/roi_label_map.json", input_size=(224, 224)):
        """
        Constructor. Loads the ONNX runtime session and image label mapping for ROI patches.
        :param model_path: path to the exported ONNX vision model file.
        :param label_map_path: path to the JSON file mapping ROI class names to integer IDs.
        :param input_size: tuple representing the required spatial dimensions (width, height) for the vision model.
        """

        # Save args:
        self.input_size = input_size
        self.label_map = {}

        # Verify ONNX Runtime installation:
        if ort is None:
            print("Error: `onnxruntime` is not installed. Please run: pip install onnxruntime")
            self.session = None
            return

        # Check if model files exist:
        if not os.path.exists(model_path) or not os.path.exists(label_map_path):
            print(f"Warning: ROI Model ({model_path}) or Label Map ({label_map_path}) not found. ROI refinement disabled.")
            self.session = None
            return

        # Load label mapping and invert it (ID -> Label String):
        with open(label_map_path, "r") as f:
            label_map = json.load(f)
        self.idx_to_label = {int(idx): label.upper() for label, idx in label_map.items()}

        # Initialize ONNX Runtime Inference Session:
        print(f"Loading ONNX ROI Classifier model from: {model_path}...")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        print("ROI Classifier engine online!")

    def _preprocess_image(self, roi_patch):
        """
        Resizes and normalizes the raw image patch to match standard ImageNet input distributions.
        :param roi_patch: raw OpenCV BGR image array of the cropped bounding box.
        :return: preprocessed numpy array of shape [1, 3, height, width] ready for ONNX inference.
        """

        # Convert BGR to RGB:
        rgb_patch = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2RGB)

        # Resize to model input dimensions:
        resized_patch = cv2.resize(rgb_patch, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Scale pixel values to [0.0, 1.0]:
        img_data = resized_patch.astype(np.float32) / 255.0

        # Apply standard ImageNet mean and standard deviation normalization:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (img_data - mean) / std

        # Transpose from [H, W, C] to [C, H, W] and add batch dimension:
        tensor_data = np.transpose(norm_img, (2, 0, 1))
        return np.expand_dims(tensor_data, axis=0)

    def classify_patch(self, roi_patch, threshold=0.45):
        """
        Runs the vision model on the cropped image patch to identify objects like cigarettes, phones, or drinks.
        :param roi_patch: raw OpenCV BGR image array of the cropped bounding box.
        :param threshold: float minimum confidence required to return a valid class prediction.
        :return: tuple of (label_str, probability_float) or None if below threshold or uninitialized.
        """

        # If engine failed to initialize or patch is empty:
        if self.session is None or roi_patch is None or roi_patch.size == 0:
            return None

        # Preprocess the image crop:
        input_tensor = self._preprocess_image(roi_patch)

        # Run ONNX inference:
        logits = self.session.run(None, {self.input_name: input_tensor})[0][0]

        # Compute Softmax probabilities for mutually exclusive object classes:
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # Get top prediction:
        best_idx = int(np.argmax(probabilities))
        best_prob = float(probabilities[best_idx])

        # Verify confidence threshold:
        if best_prob >= threshold:
            label_str = self.idx_to_label.get(best_idx, "UNKNOWN")
            return (label_str, best_prob)

        return None
