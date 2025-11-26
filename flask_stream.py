# flask_stream.py
from flask import Flask, Response
import cv2

app = Flask(__name__)

# This reference will be replaced by main.py
latest_frame_ref = None

def gen_frames():
    global latest_frame_ref
    while True:
        if latest_frame_ref is None:
            continue

        frame = latest_frame_ref.copy()

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/video")
def video_feed():
    return Response(gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def start_flask(shared_frame_obj):
    global latest_frame_ref
    latest_frame_ref = shared_frame_obj
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

