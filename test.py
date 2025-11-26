import cv2
import time
import threading

from blue_detector import detect_blue_object
from can_toolbox import send_message
from pin_toggle import toggle

# ----------------- Flask imports -----------------
from flask import Flask, Response, render_template_string

# Global buffer for latest JPEG frame (for Flask)
latest_jpeg = None

app = Flask(__name__)


# =======================
#   FLASK STREAM SETUP
# =======================

@app.route("/")
def index():
    html = """
    <html>
      <head>
        <title>Blue Detector Stream</title>
        <style>
          body { background:#111; color:#ddd; text-align:center; font-family:Arial; }
          img  { border:2px solid #444; margin-top:20px; width:70%; max-width:960px; }
        </style>
      </head>
      <body>
        <h2>Pi Camera – Blue Detector Live Stream</h2>
        <p>Stream provided by main.py (no second camera).</p>
        <img src="{{ url_for('video_feed') }}">
      </body>
    </html>
    """
    return render_template_string(html)


@app.route("/video_feed")
def video_feed():
    def gen():
        global latest_jpeg
        while True:
            if latest_jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    latest_jpeg +
                    b"\r\n"
                )
            time.sleep(0.03)  # ~30 fps cap
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def start_flask():
    # Run Flask in a background thread
    app.run(host="0.0.0.0", port=5000, debug=False,
            threaded=True, use_reloader=False)


# =======================
#   CAMERA / MAIN LOGIC
# =======================

# Try to import Picamera2 for Pi Camera 3 support
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

# ---------------------------------------------------------------------
# Set this to "laptop" or "pi" depending on where you're running.
# ---------------------------------------------------------------------
PLATFORM = "pi"   # << change as needed
# ---------------------------------------------------------------------


def get_config(platform: str):
    """
    Returns a config dict depending on where we're running.
    """
    if platform == "laptop":
        return {
            "backend": "opencv",
            "device_index": 0,
            "show_window": True,
            "width": 640,
            "height": 480,
        }
    elif platform == "pi":
        return {
            "backend": "picamera2",
            # usually False when using Flask stream; set True if you want X11 window too
            "show_window": False,
            # use wider FOV on Pi
            "width": 1640,
            "height": 1232,
        }
    else:
        raise ValueError(f"Unknown PLATFORM: {platform}")


def call_arduino():
    pass
    # toggle(27, 2)


def main():
    global latest_jpeg

    cfg = get_config(PLATFORM)
    backend = cfg["backend"]

    # ---------------------------------------------------------
    # 1) Initialize camera
    # ---------------------------------------------------------
    if backend == "opencv":
        cap = cv2.VideoCapture(cfg["device_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])

        if not cap.isOpened():
            print("Could not open laptop camera")
            return

        print("Running on LAPTOP | window:", cfg["show_window"])

    elif backend == "picamera2":
        if not PICAM_AVAILABLE:
            print("Picamera2 missing; install with: sudo apt install python3-picamera2")
            return

        picam = Picamera2()
        picam.configure(
            picam.create_video_configuration(
                main={"size": (cfg["width"], cfg["height"])}
            )
        )
        picam.start()

        print("Running on PI CAMERA 3 | window:", cfg["show_window"])
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")

    # ---------------------------------------------------------
    # 2) Main Loop
    # ---------------------------------------------------------
    frame_count = 0

    try:
        while True:

            # Grab frame
            if backend == "opencv":
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
            else:
                frame_rgb = picam.capture_array()
                # Convert RGB → BGR so colours are correct in OpenCV
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # 🔄 Rotate Pi camera frame BEFORE any calculations
                # so all centering/angle math is done in landscape orientation.
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                # If this is the wrong way, swap to:
                # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Run detection on the rotated frame (for Pi) or raw frame (for laptop)
            annotated, info = detect_blue_object(frame)

            if info is not None:
                x, y, w, h, cx, cy, ex_rel, ey_rel, angle_rel, distance_m  = info
                send_message(0x100, ex_rel, ey_rel)
                send_message(0x101, angle_rel, distance_m)

                print(
                    f"Center error X,Y: ({ex_rel:.3f}, {ey_rel:.3f})  "
                    f"Size: {w}x{h}, angle: {angle_rel:.3f}, distance_m {distance_m:.3f}"
                )
            else:
                print("No blue object detected.                              ", end="\r")

            # ------------ Update Flask stream frame ------------
            # Use the already-rotated annotated frame for the stream
            stream_frame = cv2.resize(
                annotated,
                (960, 720),
                interpolation=cv2.INTER_AREA,
            )
            ok, buf = cv2.imencode(".jpg", stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                latest_jpeg = buf.tobytes()
            # --------------------------------------------------

            # Optional: local OpenCV window
            if cfg["show_window"]:
                cv2.imshow("Blue Object Detection (Local)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Window closed by user.")
                    break

            # Save debug frame every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                cv2.imwrite("debug_frame.jpg", annotated)

    except KeyboardInterrupt:
        print("\nCTRL+C detected")

    finally:
        if backend == "opencv":
            cap.release()
            if cfg["show_window"]:
                cv2.destroyAllWindows()
        else:
            picam.stop()
            if cfg["show_window"]:
                cv2.destroyAllWindows()

        print("Camera released.")


# =======================
#   ENTRY POINT
# =======================

if __name__ == "__main__":
    # Start Flask stream in background
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    print("Flask stream running on http://0.0.0.0:5000")
    # Run main camera / detection loop
    main()