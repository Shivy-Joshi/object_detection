# flask_stream.py
# Live annotated stream from Pi Camera 3 using Flask (MJPEG)

from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import cv2
from blue_detector import detect_blue_object

app = Flask(__name__)

# =======================
#   CAMERA CONFIG
# =======================

picam = Picamera2()

config = picam.create_video_configuration(
    main={
        "size": (1640, 1232),     # full wide FOV, moderate resolution
        "format": "RGB888",       # IMPORTANT: RGB to avoid blue skin
    },
    buffer_count=2,
)

picam.configure(config)
picam.start()

print("Camera configured:", config)


# =======================
#   FRAME GENERATOR
# =======================

def gen_frames():
    while True:
        # Capture RGB frame
        frame_rgb = picam.capture_array()

        # Convert to BGR for OpenCV (corrects color)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Rotate 90° to make camera landscape
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

        # Run your detector (returns annotated frame)
        annotated, info = detect_blue_object(frame_bgr)

        # Resize for streaming — NO cropping, only scaling
        display_frame = cv2.resize(
            annotated,
            (960, 720),                 # change size here if needed
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
        <title>Pi Camera – Landscape Stream</title>
        <style>
          body { background:#111; color:#ddd; text-align:center; font-family:Arial; }
          img  { border:2px solid #444; margin-top:20px; width:70%; max-width:960px; }
        </style>
      </head>
      <body>
        <h2>Pi Camera 3 – Landscape, Scaled, Correct Colours</h2>
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
