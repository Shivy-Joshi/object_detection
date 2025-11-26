#!/usr/bin/env python3
"""
calibrate_camera.py  (Flask version)

Distance calibration using a known white object at a fixed distance.

Flow:
1) You run:  python3 calibrate_camera.py
2) Enter the known distance (meters) in the terminal (e.g., 0.15).
3) On your laptop, open:  http://<pi-ip>:5001/
4) Place the white object at that distance, so the green box fits it well.
5) Click "Capture Calibration".
6) The script saves ref_distance_m and ref_pixel_height to
   'distance_calibration.json' and shows a confirmation page.
"""

import json
import os
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, redirect, url_for

# Try to import Picamera2
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

# ----------------- Config -----------------
CALIB_FILE = "distance_calibration.json"
PORT = 5001  # use 5001 so it doesn't clash with main.py (5000)

# Globals shared between streaming + capture
picam = None
latest_h = None           # latest detected pixel height of white object
ref_distance_m = None     # known distance you enter at startup

app = Flask(__name__)


# ------------- White object detection ------------- #
def find_white_object(frame_bgr):
    """
    Simple segmentation of a white object using HSV thresholding.
    Returns:
        bbox: (x, y, w, h) of largest white blob, or None
        vis:  frame with drawing/overlay
    """
    vis = frame_bgr.copy()

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Threshold for "white-ish": low saturation, high value
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Clean up noise a bit
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, vis

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        vis,
        f"h_pixels = {h}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return (x, y, w, h), vis


# ------------- Frame generator for MJPEG ------------ #
def gen_frames():
    global latest_h, picam, ref_distance_m

    while True:
        # Capture from Pi camera as RGB
        frame_rgb = picam.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Rotate to match main.py orientation (camera mounted sideways)
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

        # Find white object and annotate frame
        bbox, vis = find_white_object(frame_bgr)

        if bbox is not None:
            _, _, _, h = bbox
            latest_h = int(h)
            status_text = f"Current height: {h} px | Known distance: {ref_distance_m:.3f} m"
        else:
            latest_h = None
            status_text = "No white object detected | Adjust object position"

        # Put status text at top
        cv2.putText(
            vis,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Resize for streaming (no cropping, just scaling)
        vis_small = cv2.resize(vis, (960, 720), interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode(".jpg", vis_small,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# ----------------- Flask routes ----------------- #
@app.route("/")
def index():
    html = """
    <html>
      <head>
        <title>Distance Calibration</title>
        <style>
          body { background:#111; color:#ddd; font-family:Arial; text-align:center; }
          img  { border:2px solid #444; margin-top:20px; width:70%; max-width:960px; }
          button {
            margin-top: 20px; padding: 10px 20px;
            font-size: 16px; border-radius: 6px;
          }
        </style>
      </head>
      <body>
        <h2>Distance Calibration – White Object</h2>
        <p>Known distance: {{ dist }} m</p>
        <img src="{{ url_for('video_feed') }}"><br/>
        <form action="{{ url_for('capture') }}" method="post">
          <button type="submit">Capture Calibration</button>
        </form>
        <p>Make sure the green box tightly covers the white object before capturing.</p>
      </body>
    </html>
    """
    return render_template_string(html, dist=ref_distance_m)


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/capture", methods=["POST"])
def capture():
    global latest_h, ref_distance_m

    if latest_h is None:
        msg = "No white object detected when capture was pressed. Try again."
        html = f"""
        <html>
          <head><title>Calibration Failed</title></head>
          <body style="background:#111; color:#ddd; font-family:Arial; text-align:center;">
            <h2>Calibration Failed</h2>
            <p>{msg}</p>
            <p><a href="{url_for('index')}">Go back</a></p>
          </body>
        </html>
        """
        return html

    # Save model: D_est = ref_distance_m * (ref_pixel_height / h)
    calib_data = {
        "ref_distance_m": ref_distance_m,
        "ref_pixel_height": latest_h,
        "model": "D_est = ref_distance_m * (ref_pixel_height / h)",
        "note": "Valid for same white object and camera pose.",
    }

    with open(CALIB_FILE, "w") as f:
        json.dump(calib_data, f, indent=4)

    full_path = os.path.abspath(CALIB_FILE)
    html = f"""
    <html>
      <head><title>Calibration Saved</title></head>
      <body style="background:#111; color:#ddd; font-family:Arial; text-align:center;">
        <h2>Calibration Saved</h2>
        <p>ref_distance_m = {ref_distance_m:.3f} m</p>
        <p>ref_pixel_height = {latest_h} px</p>
        <p>Saved to: {full_path}</p>
        <p>You can now close this page and stop calibrate_camera.py.</p>
      </body>
    </html>
    """
    print(f"Calibration saved to {full_path}")
    print(f"  ref_distance_m   = {ref_distance_m:.3f} m")
    print(f"  ref_pixel_height = {latest_h} px")
    return html


# ----------------- Main entry ----------------- #
def main():
    global picam, ref_distance_m

    if not PICAM_AVAILABLE:
        raise RuntimeError("Picamera2 is required on the Pi for this script.")

    # Ask for known distance in meters
    while True:
        try:
            ref_distance_m = float(input("Enter known distance to white object (meters): ").strip())
            if ref_distance_m <= 0:
                print("Distance must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a numeric distance, e.g. 0.15")

    # Init Pi cam
    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": (1640, 1232)}  # same as main.py FOV
    )
    picam.configure(config)
    picam.start()
    print("Pi Camera started for calibration.")
    print(f"Open your browser at:  http://<pi-ip>:{PORT}/")

    try:
        # Run Flask server (blocking)
        app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
    finally:
        picam.stop()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
