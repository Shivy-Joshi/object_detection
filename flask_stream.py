# flask_stream.py
# Live annotated stream from Pi Camera using Flask (MJPEG)
# Full sensor FOV (2592x1944), rotated to landscape.

from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
from blue_detector import detect_blue_object

app = Flask(__name__)

# -------------- Camera Setup --------------
picam = Picamera2()

config = picam.create_video_configuration(
    main={
        "size": (2592, 1944),   # sensor native resolution (full FOV)
        "format": "RGB888",
    },
    buffer_count=2,             # keep buffers low to save memory
)
picam.configure(config)
picam.start()
print("Camera config:", config)


def gen_frames():
    while True:
        frame_rgb = picam.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Rotate to make final image landscape
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

        # Run your detector
        annotated, info = detect_blue_object(frame_bgr)

        # ---------- RESIZE WITHOUT CROPPING ----------
        # Scale down by 50% in each direction (keeps aspect ratio)
        # Adjust fx, fy as you like (0.5 = half size, 0.33 = third, etc.)
        display = cv2.resize(
            annotated,
            None,
            fx=0.1,
            fy=0.1,
            interpolation=cv2.INTER_AREA,
        )
        # ----------------------------------------------

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
          img  { border: 2px solid #444; margin-top: 20px; width: 90%; }
        </style>
      </head>
      <body>
        <h2>Pi Camera – Full FOV, Landscape</h2>
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
