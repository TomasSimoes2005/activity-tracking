## 1. Feature Engineering & Spatial Geometry

* **Convert Absolute Pixel Padding to Scale-Invariant Padding:** In `crop_interaction_roi`, the cropping bounding box uses a hardcoded `padding = 40` pixels. Across different camera resolutions (e.g., 4K vs. 360p) or varying subject distances, 40 absolute pixels will either capture irrelevant background or crop out the object entirely. **Test:** Replace absolute pixel padding with a dynamic scalar proportional to the physiological reference scale (`torso_length` or shoulder width).


* **Stabilize Profile-View Anatomical Scaling:** In `extract_features`, the primary normalization scale is shoulder width (`rs_x - ls_x`). When a person turns sideways into a profile view, the 2D projected shoulder width collapses toward zero, forcing an abrupt jump to the bounding box diagonal fallback. **Test:** Implement a blended physiological scale (e.g., $\frac{\text{torso\_length} + \text{shoulder\_width}}{2}$) that remains stable across 360-degree bodily rotations.


* **Incorporate Second-Derivative Kinematics (Acceleration):** `ActionHybridNet` safely computes instantaneous velocity $v(t) = p(t) - p(t-1)$ and concatenates it with raw positions. **Test:** Compute and concatenate second-derivative acceleration $a(t) = v(t) - v(t-1)$ to give the network explicit signal signatures for sudden, explosive movements like `FALL_FLOOR` or rapid reaching.

---

## 2. Multi-Modal Fusion & Decision Gating

* **Hardcoded Alpha Blending:** Uses fixed scalars (30/70 for oral actions; 75/25 for cellphone). Fixed weights cannot adapt to environmental context (e.g., poor lighting degrading vision vs. occluded wrists degrading pose). **Test:** Replace static scalars with a lightweight learned gating layer (like logistic regression or a 2-layer MLP) trained on validation out-of-fold probability vectors to dynamically weight modalities based on confidence scores.

* **Reality Gate Zeroing:** Forces non-dominant mutually exclusive physical postures to `0.001`. Can cause severe classification flickering between consecutive frames during transitional states (e.g., standing up from a chair). **Test:** Implement temporal hysteresis thresholding or apply an Exponential Moving Average (EMA) to probability vectors prior to mutually exclusive winner-take-all suppression.

* **Empty Hand Suppression:** Multiplies interaction class probabilities by `0.25` if `EMPTY_HAND` $> 0.65$. A brief 1-frame occlusion or motion blur on a cigarette or cup can instantly suppress true positive action ongoing in the temporal buffer. **Test:** Require temporal persistence (e.g., `EMPTY_HAND` confidence must exceed threshold for 3+ consecutive frames) before applying the `0.25` penalty multiplicative decay.

---

## 3. Neural Network Architecture & Augmentation

* **Evaluate Time-Warping Kinematic Distortion:** Sinusoidal time-warping augmentation in `_augment_sequence` non-linearly stretches and compresses the 30-frame window. Because `ActionHybridNet` calculates velocity dynamically during the forward pass, aggressive time-warping creates physically impossible velocity spikes. **Test:** Cap the maximum time-warp magnitude or apply a post-warp velocity normalization step to ensure augmented training sequences retain realistic human motion profiles.


* **Benchmark Modern Edge Vision Backbones:** ROI classifier uses MobileNetV3-Small with fine-tuning restricted to blocks 10, 11, and 12. **Test:** Benchmark newer edge-optimized vision architectures like MobileNetV4, EfficientNet-Lite, or EdgeNeXt. Additionally, test progressive layer unfreezing schedules rather than statically locking blocks 0 through 9.


* **Test Graph Convolutional Networks (ST-GCN):** Flattening 17 anatomical keypoints into a 34-channel 1D-CNN discards the inherent physical connectivity of the human skeleton. **Test:** Compare `ActionHybridNet` against a Spatial-Temporal Graph Convolutional Network (ST-GCN), which passes messages along physical bone edges (e.g., wrist to elbow to shoulder), significantly improving posture disambiguation.

---

## 4. Systems Engineering & Edge Pipeline Performance

* **Eliminate Disk I/O Bottlenecks in Data Recording:** In `DatasetWriter.process_and_save`, the file is opened, written to, and closed row-by-row (`open(..., mode='a')`) inside the live processing loop. In a high-FPS video stream, continuous file system I/O creates severe thread locking and frame dropping. **Test:** Implement an in-memory buffer array inside `DatasetWriter` that caches rows and flushes to disk asynchronously in batches (e.g., every 200 frames or upon track termination).


* **Optimize Consensus Window Overhead:** In `test_end_to_end_fused_pipeline`, temporal inference is evaluated across 3 consensus window offsets (`[-5, 0, 5]`) and averaged. While this boosts mAP, it triples the temporal inference compute load per sequence. **Test:** Measure whether a single centered window evaluation combined with temporal probability smoothing matches the mAP of the 3-offset consensus scan, freeing up critical CPU cycles on edge hardware.


* **Validate Face-to-Wrist Proximity False Alarms:** The ROI vision trigger relies on `dist_to_nose < 0.60`. **Test:** Evaluate false-trigger rates during non-interaction facial gestures (e.g., scratching the nose, resting chin on hand, wiping sweat). May need to train an explicit `BODY_GROOMING` or `RESTING_HEAD` negative class within the ROI dataset to prevent the vision model from forcing false-positive interaction blendings.
