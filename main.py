import cv2
from blue_detector import detect_blue_object   # your file containing the function

DEVICE_INDEX = 0   # /dev/video0

def main():
    # Open webcam
    cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Could not open camera")
        return

    print("Running... press 'q' in the window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Run your blue object detector
            annotated, info = detect_blue_object(frame)

            # Show the annotated frame
            cv2.imshow("Blue Object Detection", annotated)

            # Quit if user presses q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n'q' pressed, exiting...")
                break

    except KeyboardInterrupt:
        print("\nCTRL+C detected, stopping...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and window closed.")

if __name__ == "__main__":
    main()
