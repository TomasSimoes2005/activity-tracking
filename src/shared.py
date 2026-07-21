import threading
import numpy as np

# Total number of temporal frames per action sequence:
WINDOW_SIZE = 30

# Total number of extracted features per frame:
NUM_FEATURES = 46


def should_trigger_roi_crop(kpts, box, thresh_face=0.60, thresh_chest=0.45):
    """
    Determines if an ROI crop should be triggered using scale-invariant anatomical distances.
    Returns: (boolean should_crop, string zone_type)
    """
    # 1. Calculate Physiological Reference Scale (Torso Length)
    mid_shoulder = (kpts[5][:2] + kpts[6][:2]) / 2.0
    mid_hip = (kpts[11][:2] + kpts[12][:2]) / 2.0
    torso_length = np.linalg.norm(mid_shoulder - mid_hip)
    
    # Fallback to bounding box height if torso length is degenerate (severe occlusion)
    if torso_length < 1e-3:
        torso_length = max(1.0, box[3] - box[1]) * 0.4

    # 2. Extract active keypoints
    nose = kpts[0][:2]
    wrists = [kpts[9][:2], kpts[10][:2]]
    wrist_confs = [kpts[9][2], kpts[10][2]] if kpts.shape[1] > 2 else [1.0, 1.0]

    for wrist, conf in zip(wrists, wrist_confs):
        if conf < 0.3:  # Skip low-confidence wrist detections
            continue
            
        # Head/Face Gate (Phone calls, Drinking, Eating, Smoking)
        dist_to_nose = np.linalg.norm(wrist - nose) / torso_length
        if dist_to_nose < thresh_face:
            return True, "face"

        # Chest/Texting Gate (Texting, browsing phone)
        dist_to_chest = np.linalg.norm(wrist - mid_shoulder) / torso_length
        is_below_shoulders = wrist[1] >= mid_shoulder[1] - (0.1 * torso_length)
        is_above_hips = wrist[1] <= mid_hip[1] + (0.2 * torso_length)
        
        if (dist_to_chest < thresh_chest) and is_below_shoulders and is_above_hips:
            return True, "chest"

    return False, None


def crop_interaction_roi(frame, keypoints, bbox, padding=40, zone_type="face"):
    """
    Crops a square region enclosing the interaction area based on the triggered anatomical zone.
    """

    height, width = frame.shape[:2]
    lw_x, lw_y = keypoints[9][:2]
    rw_x, rw_y = keypoints[10][:2]

    # Determine active wrist:
    active_wrist = None
    if lw_y > 0 and rw_y > 0:
        active_wrist = (lw_x, lw_y) if lw_y < rw_y else (rw_x, rw_y)
    elif lw_y > 0:
        active_wrist = (lw_x, lw_y)
    elif rw_y > 0:
        active_wrist = (rw_x, rw_y)

    if active_wrist is None:
        return None

    # Define crop anchor based on zone type
    if zone_type == "chest":
        # Anchor between shoulders and active wrist for texting/lap usage
        ls_x, ls_y = keypoints[5][:2]
        rs_x, rs_y = keypoints[6][:2]
        anchor_x = (ls_x + rs_x) / 2.0 if (ls_x > 0 and rs_x > 0) else active_wrist[0]
        anchor_y = (ls_y + rs_y) / 2.0 if (ls_y > 0 and rs_y > 0) else active_wrist[1]
    else:
        # Default Zone 1: Anchor between nose and active wrist
        nose_x, nose_y = keypoints[0][:2]
        if nose_x <= 0 or nose_y <= 0:
            return None
        anchor_x, anchor_y = nose_x, nose_y

    # Calculate bounding box encompassing the anchor and active wrist
    min_x = max(0, int(min(anchor_x, active_wrist[0]) - padding))
    max_x = min(width, int(max(anchor_x, active_wrist[0]) + padding))
    min_y = max(0, int(min(anchor_y, active_wrist[1]) - padding))
    max_y = min(height, int(max(anchor_y, active_wrist[1]) + padding))

    if max_x > min_x and max_y > min_y:
        return frame[min_y:max_y, min_x:max_x]

    return None


def extract_features(keypoints, bbox):
    """
    Extracts 46 features per frame: 34 anatomically normalized coordinates + 12 orientation/kinematic metrics.
    :param keypoints: keypoint data.
    :param bbox: bounding box data.
    :return: enriched feature list of length 46.
    """

    # Extract values:
    x1, y1, x2, y2 = bbox
    bbox_width = max(x2 - x1, 1.0)
    bbox_height = max(y2 - y1, 1.0)

    # Anatomical anchor (shoulder width):
    ls_x, ls_y = keypoints[5]  # Left shoulder
    rs_x, rs_y = keypoints[6]  # Right shoulder
    if ls_x > 0 and rs_x > 0:
        anatomical_scale = np.sqrt((rs_x - ls_x) ** 2 + (rs_y - ls_y) ** 2)
        anatomical_scale = max(anatomical_scale, bbox_width * 0.1)
    else:
        anatomical_scale = np.sqrt(bbox_width ** 2 + bbox_height ** 2)

    # Normalize 17 keypoints (34 features):
    norm_kpts = []
    for x, y in keypoints:
        if x == 0 and y == 0:
            norm_kpts.extend([0.0, 0.0])
        else:
            nx = (x - x1) / anatomical_scale
            ny = (y - y1) / anatomical_scale
            norm_kpts.extend([nx, ny])

    # Keypoint references for kinematics:
    nose_x, nose_y = keypoints[0]
    le_x, le_y = keypoints[1]  # Left eye
    re_x, re_y = keypoints[2]  # Right eye
    lear_x, lear_y = keypoints[3] # Left ear
    rear_x, rear_y = keypoints[4] # Right ear
    lw_x, lw_y = keypoints[9]  # Left wrist
    rw_x, rw_y = keypoints[10]  # Right wrist

    # Wrist-to-nose Euclidean Distance:
    lw_to_nose = np.sqrt((lw_x - nose_x) ** 2 + (lw_y - nose_y) ** 2) / anatomical_scale if (lw_x > 0 and nose_x > 0) else 0.0
    rw_to_nose = np.sqrt((rw_x - nose_x) ** 2 + (rw_y - nose_y) ** 2) / anatomical_scale if (rw_x > 0 and nose_x > 0) else 0.0

    # Ear distances:
    lw_to_lear = np.sqrt((lw_x - lear_x) ** 2 + (lw_y - lear_y) ** 2) / anatomical_scale if (lw_x > 0 and lear_x > 0) else 0.0
    rw_to_rear = np.sqrt((rw_x - rear_x) ** 2 + (rw_y - rear_y) ** 2) / anatomical_scale if (rw_x > 0 and rear_x > 0) else 0.0
    lw_to_rear = np.sqrt((lw_x - rear_x) ** 2 + (lw_y - rear_y) ** 2) / anatomical_scale if (lw_x > 0 and rear_x > 0) else 0.0
    rw_to_lear = np.sqrt((rw_x - lear_x) ** 2 + (rw_y - lear_y) ** 2) / anatomical_scale if (rw_x > 0 and lear_x > 0) else 0.0

    # Wrist vertical elevation relative to shoulders:
    avg_shoulder_y = (ls_y + rs_y) / 2.0 if (ls_y > 0 and rs_y > 0) else y1
    lw_elevation = (lw_y - avg_shoulder_y) / anatomical_scale if lw_y > 0 else 0.0
    rw_elevation = (rw_y - avg_shoulder_y) / anatomical_scale if rw_y > 0 else 0.0

    # Facial orientation / profile ratio:
    if le_x > 0 and re_x > 0:
        profile_ratio = abs(re_x - le_x) / anatomical_scale
    else:
        profile_ratio = 0.0

    # Head tilt / neck extension:
    avg_eye_y = (le_y + re_y) / 2.0 if (le_y > 0 and re_y > 0) else nose_y
    if avg_eye_y > 0 and avg_shoulder_y > 0:
        head_tilt = (avg_shoulder_y - avg_eye_y) / anatomical_scale
    else:
        head_tilt = 0.5

    # Inter-wrist Euclidean Distance:
    if lw_x > 0 and rw_x > 0:
        inter_wrist_dist = np.sqrt((rw_x - lw_x) ** 2 + (rw_y - lw_y) ** 2) / anatomical_scale
    else:
        inter_wrist_dist = 0.0

    # Active wrist to mid-chest anchor:
    mid_chest_x = (ls_x + rs_x) / 2.0 if (ls_x > 0 and rs_x > 0) else (x1 + x2) / 2.0
    mid_chest_y = avg_shoulder_y + (anatomical_scale * 0.5)
    lw_to_chest = np.sqrt((lw_x - mid_chest_x) ** 2 + (lw_y - mid_chest_y) ** 2) if lw_x > 0 else 999.0
    rw_to_chest = np.sqrt((rw_x - mid_chest_x) ** 2 + (rw_y - mid_chest_y) ** 2) if rw_x > 0 else 999.0
    min_wrist_to_chest = min(lw_to_chest, rw_to_chest) / anatomical_scale if min(lw_to_chest, rw_to_chest) < 999.0 else 0.0

    return norm_kpts + [
        lw_to_nose, rw_to_nose, lw_elevation, rw_elevation,
        profile_ratio, head_tilt, inter_wrist_dist, min_wrist_to_chest,
        lw_to_lear, rw_to_rear, lw_to_rear, rw_to_lear
    ]


class FrameBuffer:
    """
    Class used to store and return information about a frame.
    """

    def __init__(self):
        """
        Initializes the frame buffer with a lock.
        """

        self.frame = None
        self.lock = threading.Lock()

    def set_frame(self, frame):
        """
        Stores the new frame, overwriting the previous one.
        :param frame: the frame to store.
        """

        with self.lock:
            self.frame = frame

    def get_frame_yolo(self):
        """
        :return: the current frame.
        """

        with self.lock:
            f = self.frame
            self.frame = None
            return f

    def get_frame_server(self):
        """
        Returns the current frame but keeps it.
        :return: the current frame.
        """

        with self.lock:
            return self.frame
