# main.py
import cv2
import time
import threading
import numpy as np
from blue_detector import detect_blue_object
from picamera2 import Picamera2

# Flask stream
from flask_stream import start_flask

# Shared global frame for Flask
latest_frame = None

PLATFORM = "pi"


def get_config(platform: str):
    if platform == "laptop":
        return {
            "backend": "opencv",
            "device_index": 0,
            "show_window": True,
            "width": 640,
            "height": 480,
        }
    elif platform == "pi":
        return {
            "backend": "picamera2",
            "show_window": False,     # Pi doesn't need local window
            "width": 1280,            # use full width from PiCam3
            "height": 720,
        }
    else:
        raise ValueError(f"Unknown platform: {platform}")


def estimate_distance(h_pixels, ref_height_mm=50, ref_h_pixels=400):
    """
    Simple proportional distance model:
        distance ≈ (ref_pixels / h_pixels) * ref_distance
    You can replace this with calibrated formula later.
    """
    if h_pixels == 0:
        return None

    ref_distance_m = 0.15
    distance_m = (ref_h_pixels / h_pixels) * ref_distance_m
    return distance_m


def main():
    global latest_frame

    cfg = get_config(PLATFORM)

    # ---------------------------------------------------------
    # 1) Initialize PI CAMERA
    # ---------------------------------------------------------
    print("Initializing Pi Camera 3...")
    picam = Picamera2()

    picam.configure(
        picam.create_video_configuration(
            main={"size": (cfg["width"], cfg["height"])}
        )
    )
    picam.start()
    time.sleep(0.2)

    print("Camera started. Launching Flask stream...")

    # ---------------------------------------------------------
    # 2) Start Flask in background thread
    # ---------------------------------------------------------
    threading.Thread(
        target=start_flask, args=(latest_frame,), daemon=True
    ).start()

    print("Flask viewer: http://<PI-IP>:5000/video\n")

    # ---------------------------------------------------------
    # 3) Main Loop
    # ---------------------------------------------------------
    frame_count = 0

    try:
        while True:
            # Capture raw frame
            frame_rgb = picam.capture_array()

            # Convert to BGR for OpenCV
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Rotate BEFORE processing (Camera mounted sideways)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # -------------------------------------
            # BLUE OBJECT DETECTION
            # -------------------------------------
            annotated, info = detect_blue_object(frame)

            if info:
                x, y, w, h, cx, cy, ex_rel, ey_rel, angle_rel = info

                # Distance estimation
                dist_m = estimate_distance(h)

                print(
                    f"errX:{ex_rel:.3f}, errY:{ey_rel:.3f}, "
                    f"angle:{angle_rel:.3f}, "
                    f"height:{h}px, dist:{dist_m:.2f} m"
                )
            else:
                print("No blue object detected.                     ", end="\r")

            # ------------------------------------------------------
            # Update Flask shared buffer
            # ------------------------------------------------------
            latest_frame = annotated

            # Optional debug window (disabled on Pi)
            if cfg["show_window"]:
                cv2.imshow("Debug", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        picam.stop()
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()
