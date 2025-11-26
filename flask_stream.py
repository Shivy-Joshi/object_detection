# flask_stream.py
# FULL SENSOR, UNCROPPED, ROTATED LANDSCAPE

from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
from blue_detector import detect_blue_object

app = Flask(__name__)

# -------------- Camera Setup (FULL SENSOR, NO CROP) --------------
picam = Picamera2()

config = picam.create_video_configuration(
    main={
        "size": (4608, 2592),   # FULL 12MP SENSOR, UNCROPPED
        "format": "RGB888"      # simple format for OpenCV
    }
)

picam.configure(config)
picam.start()
print("Using full uncropped sensor:", config)


def gen_frames():
    while True:
        frame_rgb = picam.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # ROTATE the full sensor frame so it's landscape
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

        # run detector (optional)
        annotated, info = detect_blue_object(frame_bgr)

        # JPEG encode
        ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template_string("""
    <html>
      <head>
        <title>Pi Camera 3 Full Sensor Stream</title>
        <style> body { background:#111; color:#ddd; text-align:center; } img { width:90%; } </style>
      </head>
      <body>
        <h2>Pi Camera 3 – Full Uncropped Sensor (Landscape)</h2>
        <img src="{{ url_for('video_feed') }}">
      </body>
    </html>
    """)


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    print("Streaming full sensor at http://0.0.0.0:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
