# flask_stream.py
# Live annotated stream from Pi Camera 3 using Flask (MJPEG)
# Uses Picamera2's default colour pipeline (no manual format).

from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
from blue_detector import detect_blue_object

app = Flask(__name__)

# =======================
#   CAMERA CONFIG
# =======================

picam = Picamera2()

# Let Picamera2 choose the default format/colour pipeline.
# Just ask for a reasonable resolution.
config = picam.create_video_configuration(
    main={"size": (1280, 720)}   # 720p, wide, should look normal
)
picam.configure(config)
picam.start()
print("Camera configured:", config)


# =======================
#   FRAME GENERATOR
# =======================

def gen_frames():
    while True:
        # Picamera2 returns an RGB frame with default ISP / colour settings
        frame_rgb = picam.capture_array()

        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Rotate so the sideways-mounted camera becomes landscape
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

        # Run your blue detector
        annotated, info = detect_blue_object(frame_bgr)

        # Resize for streaming – NO cropping, only scaling
        display_frame = cv2.resize(
            annotated,
            (960, 720),                 # adjust if you want a different size
            interpolation=cv2.INTER_AREA
        )

        # Encode as JPEG
        ok, buffer = cv2.imencode(".jpg", display_frame,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        jpeg_bytes = buffer.tobytes()

        # MJPEG stream format
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
        )


# =======================
#        ROUTES
# =======================

@app.route("/")
def index():
    html = """
    <html>
      <head>
        <title>Pi Camera 3 – Stream</title>
        <style>
          body { background:#111; color:#ddd; text-align:center; font-family:Arial; }
          img  { border:2px solid #444; margin-top:20px; width:70%; max-width:960px; }
        </style>
      </head>
      <body>
        <h2>Pi Camera 3 – Landscape, Scaled</h2>
        <img src="{{ url_for('video_feed') }}">
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


# =======================
#         MAIN
# =======================

if __name__ == "__main__":
    print("Starting Flask stream on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
