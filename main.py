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


def custom_handler(signum, frame):
    """
    Custom signal handler that triggers the shutdown event.
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

    # Start yolo thread:
    t_yolo = threading.Thread(target=process_yolo, args=(raw_buffer, annotated_buffer, args.record), daemon=True)
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
