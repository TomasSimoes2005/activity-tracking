import cv2
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class StreamingHandler(BaseHTTPRequestHandler):
    """
    Class for the streaming handler for displaying frames in the browser.
    Inherits from BaseHTTPRequestHandler.
    """

    def do_GET(self):
        """
        GET method for displaying frames in the browser.
        """

        # Only display in root path:
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            # Until stopped:
            while True:

                # Get next frame:
                frame = self.server.frame_buffer.get_frame_server()

                # Wait if there are no frames:
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Get frame bytes:
                _, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                frame_bytes = encoded_img.tobytes()

                # Write to browser:
                try:
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')

                # Stop if client closed the browser or stopped script:
                except (ConnectionResetError, BrokenPipeError):
                    break

                # Wait:
                time.sleep(0.03)


class VideoServer(HTTPServer):
    """
    Class for serving frames in the browser.
    Inherits from HTTPServer.
    """

    def __init__(self, server_address, handler_class, frame_buffer):
        """
        Constructor. Starts the server.
        :param server_address: server address.
        :param handler_class: handler class.
        :param frame_buffer: frame buffer.
        """

        super().__init__(server_address, handler_class)
        self.frame_buffer = frame_buffer


def run_server(frame_buffer, port=8080):
    """
    Runs the server.
    :param frame_buffer: frame buffer.
    :param port: server port.
    """

    # Init server:
    server = VideoServer(('0.0.0.0', port), StreamingHandler, frame_buffer)
    print(f"Server online at: http://127.0.0.1:{port}")
    server.serve_forever()
