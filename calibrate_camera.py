#!/usr/bin/env python3
"""
calibrate_camera.py

Distance calibration using a known white object at a fixed distance.

Flow:
1) You enter the known distance (in meters).
2) The Pi camera shows a live preview (rotated same as main.py).
3) Place the same white object at that distance.
4) Script automatically segments "white", finds the largest white blob,
   and shows its bounding box + pixel height.
5) Press 'c' to capture and save calibration constants to
   'distance_calibration.json'.

Later, you can estimate distance with:

    D_est ≈ ref_distance_m * (ref_pixel_height / current_pixel_height)

assuming you are observing the *same object*.
"""

import json
import os
import cv2
import numpy as np

# Try to import Picamera2 (Pi path only)
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
PLATFORM = "pi"   # this script is mainly for Pi
CALIB_FILE = "distance_calibration.json"


def init_camera():
    """
    Initialize camera: PiCamera2 if available and PLATFORM=='pi',
    otherwise fallback to OpenCV webcam.
    Returns (backend, cam)
        backend: "picamera2" or "opencv"
        cam:     Picamera2 instance or cv2.VideoCapture
    """
    if PLATFORM == "pi" and PICAM_AVAILABLE:
        picam = Picamera2()
        config = picam.create_video_configuration(
            main={"size": (1640, 1232)}  # same idea as main.py
        )
        picam.configure(config)
        picam.start()
        print("Using Pi Camera (Picamera2) for distance calibration.")
        return "picamera2", picam
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam for calibration.")
        print("Using OpenCV webcam for distance calibration.")
        return "opencv", cap


def grab_frame(backend, cam):
    """
    Grab a BGR frame from camera, rotated the same way as main.py
    on the Pi path (ROTATE_90_CLOCKWISE).
    """
    if backend == "picamera2":
        frame_rgb = cam.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # Rotate to match main.py orientation
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
        return frame_bgr
    else:
        ret, frame = cam.read()
        if not ret:
            return None
        return frame


def find_white_object(frame_bgr):
    """
    Simple segmentation of a white object using HSV thresholding.
    Returns:
        bbox: (x, y, w, h) of largest white blob, or None if not found.
        vis:  visualization frame with box drawn (for display).
    """
    vis = frame_bgr.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Threshold for "white-ish": low saturation, high value
    # You can tweak these if needed
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Morphological ops to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, vis

    # Take largest contour as the white object
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Draw visualization: box + height text
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


def main():
    # Ask user for known distance
    while True:
        try:
            ref_distance_m = float(
                input("Enter known distance to white object (meters): ").strip()
            )
            if ref_distance_m <= 0:
                print("Distance must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a numeric distance (e.g. 0.75).")

    backend, cam = init_camera()

    print("\n--- Distance Calibration ---")
    print("1) Place the known WHITE object at the entered distance.")
    print("2) Try to fill a good portion of the frame with the white object.")
    print("3) Make sure the box overlay looks reasonable.")
    print("4) Press 'c' to capture and save calibration.")
    print("5) Press 'q' to quit without saving.\n")

    ref_pixel_height = None

    try:
        while True:
            frame = grab_frame(backend, cam)
            if frame is None:
                print("Failed to grab frame.")
                break

            bbox, vis = find_white_object(frame)

            if bbox is None:
                cv2.putText(
                    vis,
                    "No white object detected.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                _, _, _, h = bbox
                cv2.putText(
                    vis,
                    "Press 'c' to calibrate, 'q' to quit.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis,
                    f"Current height: {h} px",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Distance Calibration View", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting without saving calibration.")
                break

            if key == ord('c'):
                if bbox is None:
                    print("No white object detected to calibrate on.")
                    continue
                _, _, _, h = bbox
                ref_pixel_height = int(h)
                print(f"Captured reference pixel height: {ref_pixel_height} px")
                break

    finally:
        if backend == "picamera2":
            cam.stop()
        else:
            cam.release()
        cv2.destroyAllWindows()

    if ref_pixel_height is None:
        print("No calibration data saved.")
        return

    # Simple model:
    #  D_est ≈ ref_distance_m * (ref_pixel_height / current_height_px)
    calib_data = {
        "ref_distance_m": ref_distance_m,
        "ref_pixel_height": ref_pixel_height,
        "model": "D_est = ref_distance_m * (ref_pixel_height / h)",
        "note": "Valid for the same white object and camera setup.",
    }

    with open(CALIB_FILE, "w") as f:
        json.dump(calib_data, f, indent=4)

    print("\nCalibration saved to:", os.path.abspath(CALIB_FILE))
    print(f"  ref_distance_m   = {ref_distance_m:.3f} m")
    print(f"  ref_pixel_height = {ref_pixel_height} px")
    print("\nIn your main code, if the same object has height h_px, you can estimate:")
    print("  D_est ≈ ref_distance_m * (ref_pixel_height / h_px)")


if __name__ == "__main__":
    main()
