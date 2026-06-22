import cv2
from ultralytics import YOLO
import numpy as np
import time
import math
import os
from datetime import datetime

# ======================== CONFIGURATION ========================
video_path = 1          # 0 for webcam, or "path/to/video.mp4"
SAVE_OUTPUT = True       # always save processed video
FORCE_RESOLUTION = False  # force 1920x1080 for live camera
TARGET_WIDTH  = 1920
TARGET_HEIGHT = 1080

# ================== DIMENSION SELECTION ========================
# For each class, choose which real dimension to use for ranging.
# 'height' -> vertical pixel size × vertical focal length
# 'width'  -> horizontal pixel size × horizontal focal length
MEASURE_BY = {
    'Person':     'height',
    'Lighthouse': 'height',
    'Warship':    'height',
    'Vessel':     'height',
    'Boat':       'height',
    'Buoy':       'height',
    'Helicopter': 'width'   # rotor span is the most reliable measurement
}

# --- Indian‑typical real dimensions (metres) ---
# "width" is not used for ranging of height‑measured classes,
# but we keep it for future reference.
OBJECT_DIMENSIONS = {
    # { class: {'width': beam_m, 'height': air_draft_m} }
    'Boat':       {'width': 3.5,  'height': 3.0},     # fishing boat
    'Buoy':       {'width': 1.5,  'height': 2.0},
    'Helicopter': {'width': 13.0, 'height': 4.0},     # rotor span ~ Dhruv 13.2m
    'Lighthouse': {'width': 5.0,  'height': 25.0},
    'Person':     {'width': 0.5,  'height': 1.7},
    'Vessel':     {'width': 10.0, 'height': 15.0},    # patrol boat
    'Warship':    {'width': 17.0, 'height': 35.0}    # Kolkata/Visakhapatnam class    
}
DEFAULT_WIDTH  = 15.0
DEFAULT_HEIGHT = 20.0

# --- Detection & tracking ---
CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_FACTOR = 0.7
MAX_HISTORY = 10
MIN_IOU_FOR_TRACKING = 0.4

# --- FPS smoothing ---
FPS_SMOOTH = 0.9

# ================== BC-80 WIDE‑END CAMERA MODEL ==================
f_mm = 4.3                     # no zoom
diag_fov_deg = 63.7            # diagonal FOV at wide end
diag_fov_rad = math.radians(diag_fov_deg)

sensor_diag_mm = 2 * f_mm * math.tan(diag_fov_rad / 2)
aspect = 16.0 / 9.0
sensor_height_mm = sensor_diag_mm / math.sqrt(aspect**2 + 1)
sensor_width_mm  = sensor_height_mm * aspect

hfov_rad = 2 * math.atan(sensor_width_mm / (2 * f_mm))
vfov_rad = 2 * math.atan(sensor_height_mm / (2 * f_mm))
HORIZONTAL_FOV = math.degrees(hfov_rad)
VERTICAL_FOV   = math.degrees(vfov_rad)

print(f"BC-80 Wide End: f={f_mm} mm")
print(f"HFOV: {HORIZONTAL_FOV:.2f}°  VFOV: {VERTICAL_FOV:.2f}°")

# =================== LOAD MODEL & VIDEO =========================
model = YOLO('best2.pt')
print("Available classes:", model.names)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error: Could not open video!")
    exit()

if FORCE_RESOLUTION:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual resolution: {frame_width}x{frame_height}")

center_x_frame = frame_width // 2
center_y_frame = frame_height // 2

FOCAL_LENGTH_H = frame_width  / (2 * math.tan(hfov_rad / 2))
FOCAL_LENGTH_V = frame_height / (2 * math.tan(vfov_rad / 2))
print(f"Focal length (pixels): H = {FOCAL_LENGTH_H:.1f}, V = {FOCAL_LENGTH_V:.1f}")

# ===================== AUTO‑SAVE SETUP ==========================
out = None
if SAVE_OUTPUT:
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    if isinstance(video_path, str) and video_path != '0':
        base, _ = os.path.splitext(video_path)
        out_filename = base + "_processed.avi"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = f"BC80_range_newCode_{timestamp}.avi"
    out = cv2.VideoWriter(out_filename, fourcc, fps_video, (frame_width, frame_height))
    print(f"🎥 Recording to: {out_filename}")

# ======================= OBJECT TRACKER =========================
class ObjectTracker:
    def __init__(self, max_history=MAX_HISTORY):
        self.tracks = {}
        self.next_id = 0
        self.max_history = max_history

    def update(self, current_detections):
        updated_tracks = {}
        for det in current_detections:
            x1, y1, x2, y2, conf, class_name = det
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            best_id, best_dist = None, float('inf')
            for tid, data in self.tracks.items():
                last_cx, last_cy = data['centers'][-1]
                dist = math.hypot(cx - last_cx, cy - last_cy)
                last_box = data['boxes'][-1]
                iou = self.calc_iou([x1, y1, x2, y2], last_box)
                if dist < 100 and iou > MIN_IOU_FOR_TRACKING and dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is not None:
                trk = self.tracks[best_id]
                sx1 = int(SMOOTHING_FACTOR*x1 + (1-SMOOTHING_FACTOR)*trk['boxes'][-1][0])
                sy1 = int(SMOOTHING_FACTOR*y1 + (1-SMOOTHING_FACTOR)*trk['boxes'][-1][1])
                sx2 = int(SMOOTHING_FACTOR*x2 + (1-SMOOTHING_FACTOR)*trk['boxes'][-1][2])
                sy2 = int(SMOOTHING_FACTOR*y2 + (1-SMOOTHING_FACTOR)*trk['boxes'][-1][3])
                sc  = SMOOTHING_FACTOR*conf + (1-SMOOTHING_FACTOR)*trk['confs'][-1]
                trk['boxes'].append([sx1, sy1, sx2, sy2])
                trk['centers'].append([(sx1+sx2)//2, (sy1+sy2)//2])
                trk['confs'].append(sc)
                trk['class_name'] = class_name
                trk['last_seen'] = time.time()
                if len(trk['boxes']) > self.max_history:
                    trk['boxes'].pop(0)
                    trk['centers'].pop(0)
                    trk['confs'].pop(0)
                updated_tracks[best_id] = trk
            else:
                self.next_id += 1
                updated_tracks[self.next_id] = {
                    'boxes': [[x1,y1,x2,y2]],
                    'centers': [[cx, cy]],
                    'confs': [conf],
                    'class_name': class_name,
                    'last_seen': time.time()
                }
        now = time.time()
        self.tracks = {tid: t for tid, t in updated_tracks.items() if now - t['last_seen'] < 1.0}
        return self.tracks

    @staticmethod
    def calc_iou(box1, box2):
        ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2-ix1)*(iy2-iy1)
        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        return inter / float(area1+area2-inter)

tracker = ObjectTracker()
distance_history = {}   # cm
angle_history = {}

# ===================== HELPER FUNCTIONS =========================
def estimate_distance_cm(pixel_size, real_size_m, focal):
    """Pinhole distance in centimetres"""
    if pixel_size > 0:
        return max(0.1, (real_size_m * 100.0 * focal) / pixel_size)
    return 0.0

def calculate_angle(cx, cy, fcx, fcy, f_h, f_v):
    dx = cx - fcx
    dy = cy - fcy
    angle_h = math.degrees(math.atan2(dx, f_h))
    angle_v = math.degrees(math.atan2(dy, f_v))
    return angle_h, angle_v

def get_distance_cm(class_name, pixel_w, pixel_h):
    """Returns distance in centimetres using the pre‑defined dimension (height or width)"""
    dims = OBJECT_DIMENSIONS.get(class_name, {'width': DEFAULT_WIDTH, 'height': DEFAULT_HEIGHT})
    measure_by = MEASURE_BY.get(class_name, 'height')   # default to height if not specified

    if measure_by == 'width':
        real_m = dims['width']
        pixel_size = pixel_w
        focal = FOCAL_LENGTH_H
    else:
        real_m = dims['height']
        pixel_size = pixel_h
        focal = FOCAL_LENGTH_V

    return estimate_distance_cm(pixel_size, real_m, focal)

# ========================= MAIN LOOP ============================
frame_count = 0
fps_display = 0.0
prev_time = time.time()
print("🚀 Processing... Press 'q' to quit, 'p' to pause")

while True:
    success, img = cap.read()
    if not success:
        print("✅ End of video or stream.")
        break
    frame_count += 1

    # FPS
    now = time.time()
    delta = now - prev_time
    prev_time = now
    if delta > 0:
        fps_display = FPS_SMOOTH * fps_display + (1 - FPS_SMOOTH) * (1.0 / delta)

    # Crosshair
    cv2.line(img, (center_x_frame-20, center_y_frame),
             (center_x_frame+20, center_y_frame), (0,255,255), 1)
    cv2.line(img, (center_x_frame, center_y_frame-20),
             (center_x_frame, center_y_frame+20), (0,255,255), 1)
    cv2.circle(img, (center_x_frame, center_y_frame), 5, (0,255,255), 1)

    # YOLO detection
    current_detections = []
    results = model(img, stream=True, conf=CONFIDENCE_THRESHOLD)
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            if class_name != 'Person':   # optional – keep only persons
                continue
            current_detections.append([x1,y1,x2,y2,conf,class_name])

    tracks = tracker.update(current_detections)

    for track_id, data in tracks.items():
        if not data['boxes']:
            continue
        boxes_arr = np.array(data['boxes'])
        avg_box = np.mean(boxes_arr, axis=0).astype(int)
        x1,y1,x2,y2 = avg_box
        avg_conf = np.mean(data['confs'][-3:]) if len(data['confs'])>=3 else data['confs'][-1]

        pixel_w = x2 - x1
        pixel_h = y2 - y1
        class_name = data['class_name']

        # Distance from fixed dimension (height or width)
        dist_cm = get_distance_cm(class_name, pixel_w, pixel_h)

        # Smooth distance
        if track_id in distance_history:
            dist_cm = SMOOTHING_FACTOR*dist_cm + (1-SMOOTHING_FACTOR)*distance_history[track_id]
        distance_history[track_id] = dist_cm

        dist_m = dist_cm / 100.0

        cx, cy = (x1+x2)//2, (y1+y2)//2
        angle_h, angle_v = calculate_angle(cx, cy, center_x_frame, center_y_frame,
                                           FOCAL_LENGTH_H, FOCAL_LENGTH_V)

        if track_id in angle_history:
            ph, pv = angle_history[track_id]
            angle_h = SMOOTHING_FACTOR*angle_h + (1-SMOOTHING_FACTOR)*ph
            angle_v = SMOOTHING_FACTOR*angle_v + (1-SMOOTHING_FACTOR)*pv
        angle_history[track_id] = (angle_h, angle_v)

        # Color
        if dist_cm < 50000:         # < 500 m
            color = (0,0,255)
        elif dist_cm < 200000:      # 500–2000 m
            color = (0,165,255)
        else:
            color = (0,255,0)

        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.line(img, (center_x_frame, center_y_frame), (cx, cy), color, 1)

        label = (f'{class_name} {avg_conf:.2f}: {dist_cm:.0f}cm ({dist_m:.2f}m) | '
                 f'H:{angle_h:+.1f}\u00b0 V:{angle_v:+.1f}\u00b0')
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, y1-th-12), (x1+tw+5, y1), color, -1)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.circle(img, (cx, cy), 4, color, -1)

    # Overlay info
    info1 = f'FPS: {fps_display:.1f} | Res: {frame_width}x{frame_height} | f={f_mm}mm'
    info2 = f'HFOV:{HORIZONTAL_FOV:.2f}\u00b0  VFOV:{VERTICAL_FOV:.2f}\u00b0'
    cv2.putText(img, info1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, info2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 1)

    cv2.imshow("BC-80 Maritime Ranging", img)
    if out is not None:
        out.write(img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(0)

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
print("🎉 Finished. Recording saved.")