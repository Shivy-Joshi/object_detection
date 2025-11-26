# flask_stream.py
# Live annotated stream from Pi Camera using Flask (MJPEG)

from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
from blue_detector import detect_blue_object

app = Flask(__name__)

# -------------- Camera Setup --------------
picam = Picamera2()

config = picam.create_video_configuration(
    main={
        "size": (1640, 1232),   # good FOV, lighter than full 2592x1944
        "format": "BGR888",     # directly usable by OpenCV
    },
    buffer_count=2,
)
picam.configure(config)
picam.start()
print("Camera config:", config)


def gen_frames():
    while True:
        # Grab a BGR frame directly
        frame_bgr = picam.capture_array()   # already BGR888

        # Rotate so image is landscape (adjust if wrong direction)
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

        # Run your detector
        annotated, info = detect_blue_object(frame_bgr)

        # ---- Resize for streaming (no cropping, just scaling) ----
        # Keep aspect ratio: choose a target width/height
        display = cv2.resize(
            annotated,
            (960, 720),               # target resolution
            interpolation=cv2.INTER_AREA,
        )
        # -----------------------------------------------------------

        ok, buffer = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
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
          img  { border: 2px solid #444; margin-top: 20px; width: 50%; max-width: 960px; }
        </style>
      </head>
      <body>
        <h2>Pi Camera – Landscape, Smaller, Correct Colours</h2>
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
    print("Streaming on http://0.0.0.0:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
