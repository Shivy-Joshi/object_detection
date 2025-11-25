import cv2
import time
from blue_detector import detect_blue_object  # Allows for object detection
#from can_toolbox import send_message          # allows for sending msgs
#from pin_toggle import toggle                 # allows for GPIO access

# Try to import Picamera2 for Pi Camera 3 support
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

# ---------------------------------------------------------------------
# Set this to "laptop" or "pi" depending on where you're running.
# ---------------------------------------------------------------------
PLATFORM = "pi"   # change to "laptop" when testing on your PC
# ---------------------------------------------------------------------


def get_config(platform: str):
    """
    Returns a config dict depending on where we're running.
    """
    if platform == "laptop":
        return {
            "backend": "opencv",
            "device_index": 0,     # usually the built-in or first USB cam
            "show_window": True,   # show live window
            "width": 640,
            "height": 480,
        }
    elif platform == "pi":
        return {
            "backend": "picamera2",
            "show_window": False,  # headless by default on Pi
            "width": 640,
            "height": 480,
        }
    else:
        raise ValueError(f"Unknown PLATFORM: {platform}")


def call_arduino():
    """
    Sets pin high to let Arduino know to actuate the arm.
    """
    toggle(27, 2)  # toggles pin on Arduino


def main():
    cfg = get_config(PLATFORM)

    # ---------------------------------------------------------
    # 1) Initialize the camera depending on platform/backend
    # ---------------------------------------------------------
    backend = cfg["backend"]

    if backend == "opencv":
        cap = cv2.VideoCapture(cfg["device_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])

        if not cap.isOpened():
            print("Could not open laptop camera")
            return

        print(
            "Running on LAPTOP",
            "| window:" + ("ON" if cfg["show_window"] else "OFF"),
            "| press 'q' to quit.",
        )

    elif backend == "picamera2":
        if not PICAM_AVAILABLE:
            print("Picamera2 not installed! Run: sudo apt install python3-picamera2")
            return

        picam = Picamera2()
        picam.configure(
            picam.create_video_configuration(
                main={"size": (cfg["width"], cfg["height"])}
            )
        )
        picam.start()

        print(
            "Running on PI (Pi Camera 3 via Picamera2)",
            "| window:" + ("ON" if cfg["show_window"] else "OFF"),
        )
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")

    # ---------------------------------------------------------
    # 2) Main loop
    # ---------------------------------------------------------
    frame_count = 0

    try:
        while True:
            # ----------- Grab frame differently per backend -----------
            if backend == "opencv":
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
            else:
                # Picamera2 returns a NumPy array in BGR order
                frame = picam.capture_array()
            # ---------------------------------------------------------

            # ----- YOUR BLUE OBJECT DETECTOR -----
            annotated, info = detect_blue_object(frame)
            # -------------------------------------
            if info is not None:
                x, y, w, h, cx, cy, ex_rel, ey_rel, angle_rel = info
                # send center error over CAN
               # send_message(0x100, ex_rel, ey_rel)
                # send angle error over CAN
                #send_message(0x101, angle_rel, 0.0)

                print(
                    f"Center error X,Y: ({ex_rel:.3f}, {ey_rel:.3f})  "
                    f"Size: {w}x{h}, angle error: {angle_rel:.3f}"
                )

                # you can call Arduino here if you want, e.g.:
                # if abs(ex_rel) < 0.05 and abs(ey_rel) < 0.05:
                #     call_arduino()

            else:
                print("No blue object detected.                               ", end="\r")

            # Show live window only on laptop (or if you explicitly enable it)
            if cfg["show_window"]:
                cv2.imshow("Blue Object Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n'q' pressed, exiting...")
                    break

            # Optional: save a debug frame every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                cv2.imwrite("debug_frame.jpg", annotated)

    except KeyboardInterrupt:
        print("\nCTRL+C detected, stopping...")

    finally:
        if backend == "opencv":
            cap.release()
            if cfg["show_window"]:
                cv2.destroyAllWindows()
        else:
            picam.stop()
        print("Camera released.")


if __name__ == "__main__":
    main()
