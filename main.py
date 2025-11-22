import cv2
from blue_detector import detect_blue_object, detect_center_face   # if your file is named detect_blue.py

DEVICE_INDEX = 0   # /dev/video0


def main():
    # Open webcam with V4L2 backend
    cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Could not open camera")
        return

    frame_count = 0

    print("Camera is running... press CTRL+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # ----- YOUR DETECTION CODE -----
            annotated, info = detect_center_face(frame)
            # --------------------------------

            if info is not None:
                cx_face, cy_face, w_face, h_face = info
                print(f"CenterFace: cx={cx_face:4d}, cy={cy_face:4d}, w={w_face:3d}, h={h_face:3d}", end="\r")
            else:
                print("No center face detected.        ", end="\r")

            # Save debug image every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                cv2.imwrite("debug_frame.jpg", annotated)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        cap.release()
        print("Camera released.")


if __name__ == "__main__":
    main()
