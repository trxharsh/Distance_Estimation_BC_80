import cv2
import threading
import time


class CameraStream:
    """
    Thread-safe camera capture wrapper.
    Continuously reads frames in a background thread so the Flask
    streaming endpoint never blocks waiting for a new frame.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._frame       = None
        self._lock        = threading.Lock()
        self._running     = True

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"[CameraStream] WARNING: could not open camera index {camera_index}")

        # Keep buffer minimal to avoid stale frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while self._running:
            if not self.cap.isOpened():
                time.sleep(0.5)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            with self._lock:
                self._frame = frame.copy()

    def get_frame(self):
        """Return the most recent frame, or None if none is available yet."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        self._running = False
        self.cap.release()
