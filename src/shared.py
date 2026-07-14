import threading
import numpy as np

# Total number of temporal frames per action sequence:
WINDOW_SIZE = 30

# Total number of extracted features per frame:
NUM_FEATURES = 42


def extract_features(keypoints, bbox):
    """
    Extracts 42 features per frame: 34 anatomically normalized coordinates + 8 orientation/kinematic metrics.
    :param keypoints: keypoint data.
    :param bbox: bounding box data.
    :return: enriched feature list of length 42.
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
    lw_x, lw_y = keypoints[9]  # Left wrist
    rw_x, rw_y = keypoints[10]  # Right wrist

    # Wrist-to-nose Euclidean Distance:
    lw_to_nose = np.sqrt((lw_x - nose_x) ** 2 + (lw_y - nose_y) ** 2) / anatomical_scale if (
                lw_x > 0 and nose_x > 0) else 0.0
    rw_to_nose = np.sqrt((rw_x - nose_x) ** 2 + (rw_y - nose_y) ** 2) / anatomical_scale if (
                rw_x > 0 and nose_x > 0) else 0.0

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
        profile_ratio, head_tilt, inter_wrist_dist, min_wrist_to_chest
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
