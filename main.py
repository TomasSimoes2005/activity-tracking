"""
=============================================================================
LIVE TEMPORAL ACTION RECOGNITION PIPELINE (MULTI-LABEL)
=============================================================================
Architecture & Data Flow:

1. DATA CAPTURE (Thread 1 - Camera): 
   - Continuously reads frames from a webcam, video file, or HTTP stream.
   - Pushes raw frames into a thread-safe sliding FrameBuffer.

2. SPATIAL VISION (Thread 2 - YOLO/Hailo-8):
   - Model: YOLOv11-Pose + ByteTrack.
   - Task: Extracts bounding boxes and 17 COCO skeletal keypoints per person.
   - Hardware: Designed to be offloaded to the Hailo-8 NPU for 30+ FPS.

3. FEATURE ENGINEERING & MEMORY (Action Predictor):
   - Normalizes keypoints into 46 scale-invariant features.
   - Calculates specific geometry (like wrist-to-ear distance to isolate cellphone usage).
   - Maintains a rolling 30-frame memory buffer (~1 second) for each Track ID.

4. TEMPORAL INFERENCE (ONNX Runtime):
   - Model: Custom ActionHybridNet (1D-CNN + BiGRU + Temporal Attention).
   - Pre-processing: Applies 1D Smoothing and calculates instantaneous velocity (dy/dt).
   - Task: Scans the 30-frame temporal window to classify chronological motion.

5. MULTI-LABEL CLASSIFICATION (Threshold Optimizer):
   - Applies independent, mathematically optimized thresholds per class.
   - Allows the pipeline to recognize simultaneous actions (like Sitting AND Drinking).

6. OUTPUT STREAMING (Thread 3 - Local Server):
   - Draws Top-5 concurrent actions onto the annotated frame.
   - Serves the live feed to a local lightweight HTTP server (http://127.0.0.1:8080).
=============================================================================

=============================================================================
EXECUTION GUIDE & COMMAND LINE ARGUMENTS
=============================================================================
Basic Usage:
    python main.py [options]

Command Line Arguments:
    --source      Specifies the input video feed. 
                  Can be a webcam USB index (e.g., "0"), a direct path to an 
                  .mp4 file, or a directory of videos. 
                  (Default: "0" - Primary Webcam)
                  
    --headless    Runs the pipeline in the background without launching the 
                  local HTTP server or allocating memory to visual rendering. 
                  Ideal for remote headless servers or pure data-collection.
                  (Default: False)
                  
    --record      Switches the pipeline from 'Live Inference Mode' into 
                  'Data Collection Mode'. Instead of predicting actions, 
                  it extracts your skeletal geometry and saves it to a CSV 
                  file named after the label you provide.
                  (Default: None)

Example Commands:
    1. Standard Live Webcam Inference (View at http://127.0.0.1:8080):
       python main.py

    2. Run Inference on a Pre-Recorded Video:
       python main.py --source input/videos/test_video.mp4

    3. Record New Synthetic Training Data the "cellphone" Class:
       python main.py --record cellphone

    4. Run Inference on a Pre-Recorded Video on a Headless Server (No Web UI):
       python main.py --source input/videos/test_video.mp4 --headless
=============================================================================
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import threading
import signal
import time
from src.shared import FrameBuffer
from src.camera import capture_frames
from src.processor import process_yolo
from src.server import run_server

# Shutdown flag:
stop_event = threading.Event()


def custom_handler(signum, _):
    """
    Custom signal handler that triggers the shutdown event.
    :param signum: integer representing the signal number.
    """

    try:
        signame = signal.Signals(signum).name
    except ValueError:
        signame = str(signum)
    print(f"Caught {signame} signal! Initiating graceful shutdown...")
    stop_event.set()


def main():
    """
    Starts the YOLO pipeline.
    """

    # Parse arguments:
    parser = argparse.ArgumentParser(description="YOLO pipeline")
    parser.add_argument('--source', type=str, default="0", help="Camera index (0), video path, or directory of videos.")
    parser.add_argument('--headless', action='store_true', help="Deactivates HTTP server.")
    parser.add_argument('--record', type=str, help="Records data in a training CSV with chosen label.")
    args = parser.parse_args()

    # Initialize buffer(s):
    raw_buffer = FrameBuffer()
    annotated_buffer = None if args.headless else FrameBuffer()

    # Register signal handlers:
    signal.signal(signal.SIGINT, custom_handler)
    signal.signal(signal.SIGTERM, custom_handler)

    # Start camera thread:
    t_cam = threading.Thread(target=capture_frames, args=(raw_buffer, args.source), daemon=True)
    t_cam.start()
    print("Camera thread started.")

    # Define optimal class thresholds:
    optimal_thresholds = {
        "CELLPHONE": 0.30,
        "DRINK": 0.56,
        "EAT": 0.35,
        "FALL_FLOOR": 0.55,
        "LIE_SLEEP": 0.48,
        "SIT": 0.37,
        "SMOKE": 0.40,
        "STAND": 0.48
    }

    # Start yolo thread:
    t_yolo = threading.Thread(target=process_yolo, args=(raw_buffer, annotated_buffer, args.record, optimal_thresholds), daemon=True)
    t_yolo.start()
    print("YOLO thread started.")

    # Start server thread:
    t_server = None
    if not args.headless:
        t_server = threading.Thread(target=run_server, args=(annotated_buffer,), daemon=True)
        t_server.start()
        print("Server thread started.")

    print("Pipeline started. Press Ctrl+C to stop.")

    # Wait until stopped:
    try:
        while not stop_event.is_set():
            time.sleep(1.0)
    except Exception as e:
        print(f"Caught exception: {e}")
        stop_event.set()
    print("Shutting down pipeline...")

    # Join camera thread:
    t_cam.join(timeout=2.0)
    print("Camera thread stopped.")

    # Join YOLO thread:
    t_yolo.join(timeout=2.0)
    print("YOLO thread stopped.")

    # Join server thread:
    if not args.headless:
        t_server.join(timeout=2.0)
        print("Server thread stopped.")

    print("Exiting...")


if __name__ == "__main__":
    main()
