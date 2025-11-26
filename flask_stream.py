# flask_stream.py
# Live annotated stream from Pi Camera 3 using Flask (MJPEG)

from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
from blue_detector import detect_blue_object

app = Flask(__name__)

# ----------------- Camera Setup -----------------
picam = Picamera2()
picam.configure(
    picam.create_video_configuration(
        main={"size": (1280, 720)}   # USE WIDE RESOLUTION
    )
)
picam.start()


def gen_frames():
    while True:
        frame_rgb = picam.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 🔵 FIX: rotate so the frame is WIDE, not tall
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

        annotated, info = detect_blue_object(frame_bgr)

        success, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    html = """
    <html>
      <head>
        <title>Pi Blue Detector Stream</title>
        <style>
          body { background:#111; color:#ddd; text-align:center; }
          img { border: 2px solid #444; margin-top: 20px; width:90%; }
        </style>
      </head>
      <body>
        <h2>Pi Camera 3 – Blue Detector View (Wide Mode)</h2>
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
