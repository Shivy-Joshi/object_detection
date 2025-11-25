# test.py  --> takes a picture every 5 seconds on the Pi
import os
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2


def start_diagnostic_photos(save_folder="diagnostics"):
    """
    Takes a photo every 5 seconds using Pi Camera 3 (Picamera2).
    Images are saved with timestamp names in the specified folder.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    picam = Picamera2()
    picam.configure(
        picam.create_still_configuration(
            main={"size": (640, 480)}  # change resolution if you want
        )
    )
    picam.start()
    print(f"[DIAG] Diagnostic photo capture started. Saving to: {save_folder}/")
    print("[DIAG] Press CTRL+C to stop.")

    try:
        while True:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_folder, f"diag_{timestamp}.jpg")

            frame = picam.capture_array()   # BGR numpy array
            cv2.imwrite(filename, frame)

            print(f"[DIAG] Saved {filename}")
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n[DIAG] Diagnostic capture stopped by user.")

    finally:
        picam.stop()
        print("[DIAG] Camera released.")


if __name__ == "__main__":
    # 👈 THIS is what was missing before
    start_diagnostic_photos()
