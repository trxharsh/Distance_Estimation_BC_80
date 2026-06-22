"""
Lens_control.py — standalone smoke-test for BC-80 zoom via DVIP/VISCA.

BUG in original: zoom command used 0x02 as the payload byte, which is
not a valid VISCA zoom command.  The correct byte for variable-speed
tele is 0x20 + speed (0–7).  0x02 alone sits in an undefined range.

Fixed: use 0x25 (tele, speed 5) then 0x00 (stop).
"""

import socket
import time

CAMERA_IP = "192.168.100.100"
PORT      = 5002

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(3)

print("Connecting …")
sock.connect((CAMERA_IP, PORT))
print("Connected")


def send_visca(sock: socket.socket, visca: bytes) -> bytes | None:
    """
    DVIP framing: [len_hi, len_lo] + visca
    Packet Length = len(visca) + 2   (per DVIP spec §3.1)
    """
    plen   = len(visca) + 2
    packet = bytes([(plen >> 8) & 0xFF, plen & 0xFF]) + visca
    sock.sendall(packet)
    try:
        return sock.recv(1024)
    except socket.timeout:
        return None


# ── ZOOM IN (tele, speed 5) ──────────────────────────────────────────
# Original had 0x02 here — WRONG. Correct is 0x20 + speed.
visca_tele = bytes([0x81, 0x01, 0x04, 0x07, 0x25, 0xFF])
print("Sending ZOOM TELE (speed 5) …")
resp = send_visca(sock, visca_tele)
print("Response:", resp)

time.sleep(2)

# ── ZOOM STOP ────────────────────────────────────────────────────────
visca_stop = bytes([0x81, 0x01, 0x04, 0x07, 0x00, 0xFF])
print("Sending ZOOM STOP …")
resp = send_visca(sock, visca_stop)
print("Response:", resp)

sock.close()
print("Done.")
