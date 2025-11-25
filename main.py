import cv2
import time
from blue_detector import detect_blue_object
# from can_toolbox import send_message
# from pin_toggle import toggle

# Try to import Picamera2 for Pi Camera 3 support
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

# ---------------------------------------------------------------------
# Set this to "laptop" or "pi" depending on where you're running.
# ---------------------------------------------------------------------
PLATFORM = "pi"   # << change as needed
# ---------------------------------------------------------------------


def get_config(platform: str):
    """
    Returns a config dict depending on where we're running.
    """
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
            # ---------------------------------------------------------
            # OPTION 1 (SSH + X11 LIVE VIEW)
            # Set this to True when you SSH with:  ssh -X shivy@IP
            # ---------------------------------------------------------
            "show_window": True,     # <-- turn live display on for Pi
            "width": 640,
            "height": 480,
        }
    else:
        raise ValueError(f"Unknown PLATFORM: {platform}")


def call_arduino():
    pass
    # toggle(27, 2)


def main():
    cfg = get_config(PLATFORM)
    backend = cfg["backend"]

    # ---------------------------------------------------------
    # 1) Initialize camera
    # ---------------------------------------------------------
    if backend == "opencv":
        cap = cv2.VideoCapture(cfg["device_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])

        if not cap.isOpened():
            print("Could not open laptop camera")
            return

        print("Running on LAPTOP | window:", cfg["show_window"])

    elif backend == "picamera2":
        if not PICAM_AVAILABLE:
            print("Picamera2 missing; install with: sudo apt install python3-picamera2")
            return

        picam = Picamera2()
        picam.configure(
            picam.create_video_configuration(
                main={"size": (cfg["width"], cfg["height"])}
            )
        )
        picam.start()

        print("Running on PI CAMERA 3 | window:", cfg["show_window"])
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")

    # ---------------------------------------------------------
    # 2) Main Loop
    # ---------------------------------------------------------
    frame_count = 0

    try:
        while True:

            # Grab frame
            if backend == "opencv":
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
            else:
                frame_rgb = picam.capture_array()
                # Convert RGB → BGR so colours are correct in OpenCV
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Run detection
            annotated, info = detect_blue_object(frame)

            if info is not None:
                x, y, w, h, cx, cy, ex_rel, ey_rel, angle_rel = info
                # send_message(0x100, ex_rel, ey_rel)
                # send_message(0x101, angle_rel, 0.0)

                print(
                    f"Center error X,Y: ({ex_rel:.3f}, {ey_rel:.3f})  "
                    f"Size: {w}x{h}, angle: {angle_rel:.3f}"
                )
            else:
                print("No blue object detected.                              ", end="\r")

            # ----------- OPTION 1: Show annotated over SSH -X -----------
            if cfg["show_window"]:
                cv2.imshow("Blue Object Detection (SSH Stream)", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Window closed by user.")
                    break
            # ------------------------------------------------------------

            # Save debug frame every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                cv2.imwrite("debug_frame.jpg", annotated)

    except KeyboardInterrupt:
        print("\nCTRL+C detected")

    finally:
        if backend == "opencv":
            cap.release()
            if cfg["show_window"]:
                cv2.destroyAllWindows()
        else:
            picam.stop()
            if cfg["show_window"]:
                cv2.destroyAllWindows()

        print("Camera released.")


if __name__ == "__main__":
    main()
