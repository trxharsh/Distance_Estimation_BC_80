from flask import Flask, Response, render_template, jsonify
import cv2

from components.bc80_controller import BC80Controller
from components.camera_stream    import CameraStream


# =========================================
# CONFIG  — edit these before running
# =========================================

BC80_IP      = "192.168.100.100"
BC80_PORT    = 5002
CAMERA_INDEX = 1   # OpenCV capture index for the BC-80 SDI/HDMI feed


# =========================================
# INIT
# =========================================

app        = Flask(__name__)
controller = BC80Controller(BC80_IP, BC80_PORT, auto_reconnect=True)
camera     = CameraStream(CAMERA_INDEX)

# =========================================
# HELPERS
# =========================================

def _ok(msg: str = "OK"):
    return jsonify({"status": "ok", "message": msg})

def _err(msg: str):
    return jsonify({"status": "error", "message": msg}), 500


# =========================================
# VIDEO STREAM
# =========================================

def _generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buf.tobytes() +
            b'\r\n'
        )

def zoom_position_to_ratio(pos):

    return round(
        1 + (pos / 0x4000) * 29,
        1
    )

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/zoom_position')
def zoom_position():

    pos = controller.get_zoom_position()
    print("zoom_position_value =", pos)

    ratio = zoom_position_to_ratio(pos)

    return jsonify({
        "zoom_position": pos,
        "zoom_ratio": ratio
    })

@app.route('/video_feed')
def video_feed():
    return Response(
        _generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/status')
def status():
    return jsonify({"connected": controller.connected, "ip": controller.ip})


# =========================================
# ZOOM
# =========================================

@app.route('/zoom_in')
def zoom_in():
    controller.zoom_in()
    return _ok("zoom_in")

@app.route('/zoom_out')
def zoom_out():
    controller.zoom_out()
    return _ok("zoom_out")

@app.route('/zoom_stop')
def zoom_stop():
    controller.zoom_stop()
    return _ok("zoom_stop")


# =========================================
# FOCUS
# =========================================

@app.route('/focus_far')
def focus_far():
    controller.focus_far()
    return _ok("focus_far")

@app.route('/focus_near')
def focus_near():
    controller.focus_near()
    return _ok("focus_near")

@app.route('/focus_stop')
def focus_stop():
    controller.focus_stop()
    return _ok("focus_stop")

@app.route('/af_on')
def af_on():
    controller.autofocus_on()
    return _ok("af_on")

@app.route('/af_off')
def af_off():
    controller.autofocus_off()
    return _ok("af_off")

@app.route('/one_push_focus')
def one_push_focus():
    controller.one_push_focus()
    return _ok("one_push_focus")


# =========================================
# IRIS  (iris_stop was completely missing — broken hold behaviour)
# =========================================

@app.route('/iris_open')
def iris_open():
    controller.iris_open()
    return _ok("iris_open")

@app.route('/iris_close')
def iris_close():
    controller.iris_close()
    return _ok("iris_close")

@app.route('/iris_stop')
def iris_stop():
    controller.iris_stop()
    return _ok("iris_stop")


# =========================================
# GAIN
# =========================================

@app.route('/gain_up')
def gain_up():
    controller.gain_up()
    return _ok("gain_up")

@app.route('/gain_down')
def gain_down():
    controller.gain_down()
    return _ok("gain_down")

@app.route('/gain_reset')
def gain_reset():
    controller.gain_reset()
    return _ok("gain_reset")


# =========================================
# SHUTTER
# =========================================

@app.route('/shutter_up')
def shutter_up():
    controller.shutter_up()
    return _ok("shutter_up")

@app.route('/shutter_down')
def shutter_down():
    controller.shutter_down()
    return _ok("shutter_down")


# =========================================
# EXPOSURE MODE
# =========================================

@app.route('/exp_auto')
def exp_auto():
    controller.exposure_auto()
    return _ok("exposure_auto")

@app.route('/exp_manual')
def exp_manual():
    controller.exposure_manual()
    return _ok("exposure_manual")

@app.route('/exp_shutter_priority')
def exp_shutter_priority():
    controller.exposure_shutter_priority()
    return _ok("exposure_shutter_priority")

@app.route('/exp_iris_priority')
def exp_iris_priority():
    controller.exposure_iris_priority()
    return _ok("exposure_iris_priority")


# =========================================
# WHITE BALANCE
# =========================================

@app.route('/wb_auto')
def wb_auto():
    controller.wb_auto()
    return _ok("wb_auto")

@app.route('/wb_indoor')
def wb_indoor():
    controller.wb_indoor()
    return _ok("wb_indoor")

@app.route('/wb_outdoor')
def wb_outdoor():
    controller.wb_outdoor()
    return _ok("wb_outdoor")

@app.route('/wb_one_push')
def wb_one_push():
    controller.wb_one_push()
    return _ok("wb_one_push")


# =========================================
# BACKLIGHT
# =========================================

@app.route('/backlight_on')
def backlight_on():
    controller.backlight_on()
    return _ok("backlight_on")

@app.route('/backlight_off')
def backlight_off():
    controller.backlight_off()
    return _ok("backlight_off")


# =========================================
# MIRROR / FLIP
# =========================================

@app.route('/mirror_lr_on')
def mirror_lr_on():
    controller.mirror_lr_on()
    return _ok("mirror_lr_on")

@app.route('/mirror_lr_off')
def mirror_lr_off():
    controller.mirror_lr_off()
    return _ok("mirror_lr_off")

@app.route('/mirror_tb_on')
def mirror_tb_on():
    controller.mirror_tb_on()
    return _ok("mirror_tb_on")

@app.route('/mirror_tb_off')
def mirror_tb_off():
    controller.mirror_tb_off()
    return _ok("mirror_tb_off")


# =========================================
# OSD MENU
# =========================================

@app.route('/menu_on')
def menu_on():
    controller.menu_on()
    return _ok("menu_on")

@app.route('/menu_off')
def menu_off():
    controller.menu_off()
    return _ok("menu_off")


# =========================================
# PRESETS
# =========================================

@app.route('/preset_set/<int:n>')
def preset_set(n):
    if not 0 <= n <= 9:
        return _err("Preset number must be 0–9")
    controller.preset_set(n)
    return _ok(f"preset_set:{n}")

@app.route('/preset_recall/<int:n>')
def preset_recall(n):
    if not 0 <= n <= 9:
        return _err("Preset number must be 0–9")
    controller.preset_recall(n)
    return _ok(f"preset_recall:{n}")


# =========================================
# POWER
# =========================================

@app.route('/power_on')
def power_on():
    controller.power_on()
    return _ok("power_on")

@app.route('/power_standby')
def power_standby():
    controller.power_standby()
    return _ok("power_standby")


# =========================================
# MAIN
# =========================================

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        controller.close()
        camera.release()
