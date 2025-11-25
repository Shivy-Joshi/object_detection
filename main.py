import cv2
from blue_detector import detect_blue_object
from can_toolbox import send_message

# PiCamera support
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

PLATFORM = "pi"   # change to "laptop" or "pi"


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
            "show_window": False,
            "width": 640,
            "height": 480,
        }
    else:
        raise ValueError(f"Unknown PLATFORM: {platform}")


def calculate_offset_error(img_x, obj_x):
    img_center = img_x / 2
    center_abs_error = img_center - obj_x
    center_rel_error = center_abs_error / img_x
    return center_rel_error


def main():
    cfg = get_config(PLATFORM)

    # ---------------------------------------------------------
    # 1) Initialize the camera depending on platform
    # ---------------------------------------------------------
    if cfg["backend"] == "opencv":
        cap = cv2.VideoCapture(cfg["device_index"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])

        if not cap.isOpened():
            print("Could not open laptop camera")
            return

        print("Running on LAPTOP | press 'q' to quit")

    else:
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
        print("Running on PI CAMERA 3 (Picamera2)")

    # ---------------------------------------------------------
    # 2) Main loop
    # ---------------------------------------------------------
    frame_count = 0

    try:
        while True:

            # ----------- Grab frame differently -----------
            if cfg["backend"] == "opencv":
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
            else:
                frame = picam.capture_array()    # NumPy BGR image
            # ----------------------------------------------

            annotated, info = detect_blue_object(frame)

            if info is not None:
                x, y, w, h, cx, cy, ex_rel, ey_rel = info
                send_message(0x100, ex_rel, ey_rel)
                print(f"Center error X,Y: ({ex_rel:.3f}, {ey_rel:.3f}) Size: {w}x{h}")
            else:
                print("No blue object detected.      ", end="\r")

            # Show window only on laptop
            if cfg["show_window"]:
                cv2.imshow("Blue Object Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Save debug frames
            frame_count += 1
            if frame_count % 30 == 0:
                cv2.imwrite("debug_frame.jpg", annotated)

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        if cfg["backend"] == "opencv":
            cap.release()
            cv2.destroyAllWindows()
        else:
            picam.stop()
        print("Camera released.")


if __name__ == "__main__":
    main()
