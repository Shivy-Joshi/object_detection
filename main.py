import cv2
from blue_detector import detect_blue_object  # Allows for object detection
from can_toolbox import send_message # allows for sending msgs
# ---------------------------------------------------------------------
# Set this to "laptop" or "pi" depending on where you're running.
# ---------------------------------------------------------------------
PLATFORM = "laptop"   # change to "pi" when you move to the Pi
# ---------------------------------------------------------------------


def get_config(platform: str):
    """
    Returns a config dict depending on where we're running.
    """
    if platform == "laptop":
        return {
            "device_index": 0,     # usually the built-in or first USB cam
            "use_v4l2": False,     # don't force V4L2 backend on laptop
            "show_window": True,   # show live window
            "width": 640,
            "height": 480,
        }
    elif platform == "pi":
        return {
            "device_index": 0,     # /dev/video0 on Pi
            "use_v4l2": True,      # use V4L2 backend
            "show_window": False,  # headless by default
            "width": 640,
            "height": 480,
        }
    else:
        raise ValueError(f"Unknown PLATFORM: {platform}")



def calculate_offset_error(img_x,obj_x):
    """
        Returns a relative error of how centered the object is in the image given image width and object center location.
    """
    img_center  = img_x/2
    center_abs_error = img_center - obj_x
    center_rel_error = center_abs_error /img_x
    return center_rel_error



def main():
    cfg = get_config(PLATFORM)

    # Open camera with appropriate backend
    if cfg["use_v4l2"]:
        cap = cv2.VideoCapture(cfg["device_index"], cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(cfg["device_index"])

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])

    if not cap.isOpened():
        print("Could not open camera")
        return

    print(
        "Running on",
        PLATFORM,
        "| window:" + ("ON" if cfg["show_window"] else "OFF") +
        " | press 'q' to quit (if window shown) or CTRL+C in terminal."
    )

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # ----- YOUR BLUE OBJECT DETECTOR -----
            annotated, info = detect_blue_object(frame)
            # -------------------------------------
            if info is not None:
                x, y, w, h, cx, cy, ex_rel, ey_rel = info
                send_message(0X100,ex_rel,ey_rel) # sends the error
                print(f"Center error X,Y: ({ex_rel}, {ey_rel})  Size: {w}x{h}")

            else:
                print("No blue object detected.                               ", end="\r")

            # Show live window only on laptop
            if cfg["show_window"]:
                cv2.imshow("Blue Object Detection", annotated)
                # check for 'q' key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n'q' pressed, exiting...")
                    break



            # Optional: on Pi or laptop, save a debug frame every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                cv2.imwrite("debug_frame.jpg", annotated)

    except KeyboardInterrupt:
        print("\nCTRL+C detected, stopping...")

    finally:
        cap.release()
        if cfg["show_window"]:
            cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()


