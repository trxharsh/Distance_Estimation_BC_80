import socket
import threading
import time


class BC80Controller:
    """
    VISCA-over-DVIP controller for Datavideo BC-80 HD Block Camera.

    Protocol:
      - Physical:  Ethernet, TCP port 5002
      - Framing:   2-byte big-endian length header + VISCA payload
                   Packet Length = len(VISCA payload) + 2
    """

    RECONNECT_DELAY = 3   # seconds between reconnect attempts
    RECV_TIMEOUT    = 2   # socket receive timeout

    def __init__(self, ip: str, port: int = 5002, auto_reconnect: bool = True):
        self.ip             = ip
        self.port           = port
        self.auto_reconnect = auto_reconnect
        self._lock          = threading.Lock()   # protect socket from concurrent threads
        self._sock          = None
        self._connected     = False
        self.last_zoom_position = 0
        self._connect()

    # ------------------------------------------------------------------
    # CONNECTION MANAGEMENT
    # ------------------------------------------------------------------

    def _connect(self) -> bool:
        """Open TCP connection to the camera. Returns True on success."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.RECV_TIMEOUT)
            sock.connect((self.ip, self.port))
            self._sock      = sock
            self._connected = True
            print(f"[BC80] Connected → {self.ip}:{self.port}")
            return True
        except OSError as exc:
            print(f"[BC80] Connection failed: {exc}")
            self._connected = False
            return False

    def _reconnect(self):
        """Try to re-establish the connection (called after a send/recv failure)."""
        self._connected = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        if self.auto_reconnect:
            print(f"[BC80] Reconnecting in {self.RECONNECT_DELAY}s …")
            time.sleep(self.RECONNECT_DELAY)
            self._connect()

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # LOW-LEVEL SEND
    # ------------------------------------------------------------------

    def send_visca(self, visca: bytes) -> bytes | None:
        """
        Wrap a raw VISCA command in the DVIP 2-byte length header and send it.

        Header:  [len_high, len_low]  where  len = len(visca) + 2
        The +2 accounts for the two header bytes themselves (DVIP spec §3.1).
        """
        payload_len = len(visca) + 2          # total packet length per spec
        packet = bytes([
            (payload_len >> 8) & 0xFF,        # high byte (usually 0x00)
            payload_len & 0xFF,               # low byte
        ]) + visca

        with self._lock:
            if not self._connected:
                print("[BC80] Not connected — skipping command.")
                return None
            try:
                self._sock.sendall(packet)
                response = self._sock.recv(1024)
                return response
            except OSError as exc:
                print(f"[BC80] Send/recv error: {exc}")
                self._reconnect()
                return None

    # ------------------------------------------------------------------
    # ZOOM  (VISCA 81 01 04 07 pp FF)
    #   pp = 0x00 stop | 0x20+speed tele-in | 0x30+speed wide-out
    #   speed range: 0–7
    # ------------------------------------------------------------------

    def zoom_in(self, speed: int = 4):
        speed = max(0, min(7, speed))
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x07, 0x20 + speed, 0xFF]))

    def zoom_out(self, speed: int = 4):
        speed = max(0, min(7, speed))
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x07, 0x30 + speed, 0xFF]))

    def zoom_stop(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x07, 0x00, 0xFF]))

    # ------------------------------------------------------------------
    # FOCUS  (VISCA 81 01 04 08 pp FF)
    #   pp = 0x00 stop | 0x20+speed far | 0x30+speed near
    # ------------------------------------------------------------------

    def focus_far(self, speed: int = 4):
        speed = max(0, min(7, speed))
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x08, 0x20 + speed, 0xFF]))

    def focus_near(self, speed: int = 4):
        speed = max(0, min(7, speed))
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x08, 0x30 + speed, 0xFF]))

    def focus_stop(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x08, 0x00, 0xFF]))

    # ------------------------------------------------------------------
    # AUTO FOCUS  (VISCA 81 01 04 38 pp FF)
    #   Auto=0x02  Manual=0x03  One-push trigger=04 18 01
    # ------------------------------------------------------------------

    def autofocus_on(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x38, 0x02, 0xFF]))

    def autofocus_off(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x38, 0x03, 0xFF]))

    def one_push_focus(self):
        """Trigger one-shot autofocus (camera must be in AF mode)."""
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x18, 0x01, 0xFF]))

    # ------------------------------------------------------------------
    # IRIS  (VISCA 81 01 04 0B pp FF)
    #   Up=0x02  Down=0x03  Stop=0x00
    # ------------------------------------------------------------------

    def iris_open(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x0B, 0x02, 0xFF]))

    def iris_close(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x0B, 0x03, 0xFF]))

    def iris_stop(self):
        """Stop iris movement (was missing in original — caused broken HTML hold behaviour)."""
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x0B, 0x00, 0xFF]))

    # ------------------------------------------------------------------
    # GAIN  (VISCA 81 01 04 0C pp FF)
    #   Up=0x02  Down=0x03
    # ------------------------------------------------------------------

    def gain_up(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x0C, 0x02, 0xFF]))

    def gain_down(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x0C, 0x03, 0xFF]))

    def gain_reset(self):
        """Reset gain to default/auto."""
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x0C, 0x00, 0xFF]))

    # ------------------------------------------------------------------
    # SHUTTER  (VISCA 81 01 04 0A pp FF)
    #   Up=0x02  Down=0x03
    # ------------------------------------------------------------------

    def shutter_up(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x0A, 0x02, 0xFF]))

    def shutter_down(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x0A, 0x03, 0xFF]))

    # ------------------------------------------------------------------
    # EXPOSURE MODE  (VISCA 81 01 04 39 pp FF)
    #   Full Auto=0x00  Manual=0x03  Shutter-priority=0x0A  Iris-priority=0x0B
    # ------------------------------------------------------------------

    def exposure_auto(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x39, 0x00, 0xFF]))

    def exposure_manual(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x39, 0x03, 0xFF]))

    def exposure_shutter_priority(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x39, 0x0A, 0xFF]))

    def exposure_iris_priority(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x39, 0x0B, 0xFF]))

    # ------------------------------------------------------------------
    # WHITE BALANCE  (VISCA 81 01 04 35 pp FF)
    #   Auto=0x00  Indoor=0x01  Outdoor=0x02  One-push=0x03  Manual=0x05
    # ------------------------------------------------------------------

    def wb_auto(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x35, 0x00, 0xFF]))

    def wb_indoor(self):
        """3200 K indoor preset."""
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x35, 0x01, 0xFF]))

    def wb_outdoor(self):
        """5600 K outdoor preset."""
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x35, 0x02, 0xFF]))

    def wb_one_push(self):
        """One-push white balance lock (trigger)."""
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x10, 0x05, 0xFF]))

    def wb_manual(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x35, 0x05, 0xFF]))

    # ------------------------------------------------------------------
    # BACKLIGHT COMPENSATION  (VISCA 81 01 04 33 pp FF)
    #   On=0x02  Off=0x03
    # ------------------------------------------------------------------

    def backlight_on(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x33, 0x02, 0xFF]))

    def backlight_off(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x33, 0x03, 0xFF]))

    # ------------------------------------------------------------------
    # IMAGE MIRROR / FLIP  (VISCA 81 01 04 61 pp FF  /  04 66 pp FF)
    #   Mirror LR : 04 61  On=0x02 Off=0x03
    #   Mirror TB : 04 66  On=0x02 Off=0x03
    # ------------------------------------------------------------------

    def mirror_lr_on(self):
        """Horizontal flip (left-right mirror)."""
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x61, 0x02, 0xFF]))

    def mirror_lr_off(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x61, 0x03, 0xFF]))

    def mirror_tb_on(self):
        """Vertical flip (top-bottom mirror)."""
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x66, 0x02, 0xFF]))

    def mirror_tb_off(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x66, 0x03, 0xFF]))

    # ------------------------------------------------------------------
    # OSD MENU  (VISCA 81 01 06 06 pp FF)
    #   On=0x02  Off=0x03
    # ------------------------------------------------------------------

    def menu_on(self):
        self.send_visca(bytes([0x81, 0x01, 0x06, 0x06, 0x02, 0xFF]))

    def menu_off(self):
        self.send_visca(bytes([0x81, 0x01, 0x06, 0x06, 0x03, 0xFF]))

    # ------------------------------------------------------------------
    # PRESETS  (VISCA 81 01 04 3F mm pp FF)
    #   mm = 0x01 set  |  0x02 recall
    #   pp = preset number 0–9
    # ------------------------------------------------------------------

    def preset_set(self, preset_no: int):
        preset_no = max(0, min(9, preset_no))
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x3F, 0x01, preset_no, 0xFF]))

    def preset_recall(self, preset_no: int):
        preset_no = max(0, min(9, preset_no))
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x3F, 0x02, preset_no, 0xFF]))

    # ------------------------------------------------------------------
    # POWER  (VISCA 81 01 04 00 pp FF)
    #   On=0x02  Standby=0x03
    # ------------------------------------------------------------------

    def power_on(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x00, 0x02, 0xFF]))

    def power_standby(self):
        self.send_visca(bytes([0x81, 0x01, 0x04, 0x00, 0x03, 0xFF]))

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    def close(self):
        with self._lock:
            self._connected = False
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
        print("[BC80] Connection closed.")


    def get_zoom_position(self):

        response = self.send_visca(
            bytes([0x81, 0x09, 0x04, 0x47, 0xFF])
        )

        print("zoom position :", response)

        if not response:
            return None

        # Ignore ACK/Completion packets
        if b'\x90\x50' not in response:
            print("Ignoring non-zoom response")
            return self.last_zoom_position

        idx = response.find(b'\x90\x50')

        zoom_packet = response[idx:idx+7]

        if len(zoom_packet) < 7:
            return self.last_zoom_position

        p = zoom_packet[2]
        q = zoom_packet[3]
        r = zoom_packet[4]
        s = zoom_packet[5]

        zoom_position = (
            (p << 12) |
            (q << 8)  |
            (r << 4)  |
            s
        )

        self.last_zoom_position = zoom_position

        return zoom_position