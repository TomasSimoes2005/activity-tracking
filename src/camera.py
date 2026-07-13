import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import platform
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np

# Get OS and Arch information
OS_TYPE = sys.platform
ARCH_TYPE = platform.machine()

# Check which CLI camera tools exist in the system's PATH:
HAS_RPICAM = shutil.which("rpicam-vid") is not None
HAS_LIBCAMERA = shutil.which("libcamera-vid") is not None

# Set a global flag for our primary camera backend:
if HAS_RPICAM:
    CAMERA_BACKEND = "rpicam-vid"
elif HAS_LIBCAMERA:
    CAMERA_BACKEND = "libcamera-vid"
else:
    CAMERA_BACKEND = "opencv"

# Print information on screen:
print(f"System Detected: {OS_TYPE} ({ARCH_TYPE})")
print(f"Camera Backend Selected: [{CAMERA_BACKEND}]")


def _capture_via_pipe(raw_buffer, cam_name):
    """
    MJPEG stream pipe for extracting frames.
    :param raw_buffer: raw buffer for frame data.
    :param cam_name: camera backend name.
    """

    # Set pipe args:
    cmd = [
        cam_name, "-t", "0", "--inline",
        "--width", "640", "--height", "480",
        "--framerate", "30", "--codec", "mjpeg", "-o", "-"
    ]

    # Create pipe and bytes buffer:
    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=4096)
    bytes_buffer = b""

    try:

        # Until stopped:
        while True:

            # Get next chunk:
            chunk = pipe.stdout.read(4096)
            if not chunk:
                break

            # Append data to bytes buffer:
            bytes_buffer += chunk

            # Locate JPEG frame boundaries:
            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9')
            if a != -1 and b != -1:

                # Trim data:
                jpg = bytes_buffer[a:b + 2]
                bytes_buffer = bytes_buffer[b + 2:]

                # Decode frame:
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                # Set frame in the raw buffer:
                if frame is not None:
                    raw_buffer.set_frame(frame)
    finally:

        # Clear the pipe:
        pipe.terminate()


def _capture_via_opencv(raw_buffer, camera_index=0):
    """
    Standard frame capture for PCs, Laptops, or generic USB Webcams.
    :param raw_buffer: raw buffer for frame data.
    :param camera_index: camera backend index.
    """

    # Get video capture instance:
    cap = cv2.VideoCapture(camera_index)

    # Request standard resolution and framerate from hardware:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # If there was an error opening the video capture:
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}.")
        return

    try:

        # Until stopped:
        while True:

            # Get next frame:
            ret, frame = cap.read()

            # If no frame was given, wait:
            if not ret:
                time.sleep(0.01)
                continue

            # Set the frame in the raw buffer:
            raw_buffer.set_frame(frame)
    finally:

        # Close the video capture:
        cap.release()


def _capture_from_file(raw_buffer, source_path):
    """
    Iterates through a directory of videos for frame capture.
    :param raw_buffer: raw buffer for frame data.
    :param source_path: path of the directory that contains the video file(s) or path for a single file.
    """

    # Gather all video files from the given directory:
    video_files = []
    if os.path.isdir(source_path):
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(root, file))

    # Otherwise if single video:
    else:
        video_files.append(source_path)

    # For each video:
    print(f"Found {len(video_files)} video file(s) for batch processing.")
    for filepath in video_files:

        # Try to open video capture:
        print(f"Processing video: {filepath}")
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            print(f"Warning: Could not open {filepath}")
            continue

        # While there are frames to read:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Push to buffer and yield slightly to let YOLO consume the frame:
            raw_buffer.set_frame(frame)
            time.sleep(0.001)

        # Close video capture:
        cap.release()
    print("Batch video processing complete!")


def capture_frames(raw_buffer, source=0):
    """
    Unified entry point. Automatically routes to the fastest available hardware backend.
    :param raw_buffer: raw buffer for frame data.
    :param source: video source (either camera index or filepath).
    """

    # If source is a file path:
    if isinstance(source, str) and (os.path.exists(source) or os.path.isdir(source)):
        print(f"Starting batch ingestion from path: `{source}`...")
        _capture_from_file(raw_buffer, source)

    # If the camera backend is rpicam or libcamera:
    elif CAMERA_BACKEND in ["rpicam-vid", "libcamera-vid"]:
        print(f"Starting pipe via `{CAMERA_BACKEND}`...")
        _capture_via_pipe(raw_buffer, CAMERA_BACKEND)

    # Otherwise:
    else:
        cam_idx = int(source) if isinstance(source, str) and source.isdigit() else source
        print(f"Starting pipe via OpenCV VideoCapture (Index: {cam_idx})...")
        _capture_via_opencv(raw_buffer, cam_idx)
