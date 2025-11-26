# flask_stream.py
# Live annotated stream from Pi Camera 3 using Flask (MJPEG)

from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
from blue_detector import detect_blue_object

# ----------------- Flask setup -----------------
app = Flask(__name__)

# ----------------- Camera setup ----------------
picam = Picamera2()
picam.configure(
    picam.create_video_configuration(
        main={"size": (640, 480)}  # adjust resolution if you want
    )
)
picam.start()


def gen_frames():
    """
    Generator that captures frames from the Pi camera,
    runs blue_detector, and yields JPEG bytes for MJPEG streaming.
    """
    while True:
        # Capture frame as RGB, convert to BGR for OpenCV
        frame_rgb = picam.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # --------------------------
        # ROTATE CAMERA VIEW HERE
        # --------------------------
        # Rotate 90 degrees clockwise
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
        # If you want counter-clockwise instead, use:
        # frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # --------------------------

        # Run your existing detector on the rotated frame
        annotated, info = detect_blue_object(frame_bgr)

        # Encode the annotated frame as JPEG
        success, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            continue

        frame_bytes = buffer.tobytes()

        # MJPEG frame boundary
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# ----------------- Routes -----------------
@app.route("/")
def index():
    html = """
    <html>
      <head>
        <title>Pi Blue Detector Stream</title>
        <style>
          body { background:#111; color:#ddd; font-family: Arial, sans-serif; text-align:center; }
          img { border: 2px solid #444; margin-top: 20px; }
        </style>
      </head>
      <body>
        <h2>Pi Camera 3 – Blue Detector View</h2>
        <p>Press CTRL+C in the terminal on the Pi to stop.</p>
        <img src="{{ url_for('video_feed') }}" />
      </body>
    </html>
    """
    return render_template_string(html)


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ----------------- Main -----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
