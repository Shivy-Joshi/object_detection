# test_diag.py
# Take a picture every 5 seconds on the Raspberry Pi (Pi Camera 3)
# Images saved in ./diagnostics with correct colour (RGB -> BGR)

import os
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2


def start_diagnostic_photos(save_folder="diagnostics", interval_sec=5):
    """
    Takes a photo every `interval_sec` seconds using Pi Camera 3 (Picamera2).
    Images are saved with timestamped filenames in the specified folder.
    """

    # Make folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Set up PiCamera2 for still capture
    picam = Picamera2()
    picam.configure(
        picam.create_still_configuration(
            main={"size": (640, 480)}  # change resolution if you want
        )
    )
    picam.start()

    print(f"[DIAG] Diagnostic photo capture started. Saving to: {save_folder}/")
    print(f"[DIAG] Interval: {interval_sec} seconds. Press CTRL+C to stop.")

    try:
        while True:
            # Make timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_folder, f"diag_{timestamp}.jpg")

            # Capture frame (Picamera2 gives RGB)
            frame_rgb = picam.capture_array()

            # Convert RGB -> BGR for OpenCV/imwrite so colours look correct
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Save image
            cv2.imwrite(filename, frame_bgr)

            print(f"[DIAG] Saved {filename}")

            # Wait before next shot
            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\n[DIAG] Diagnostic capture stopped by user.")

    finally:
        picam.stop()
        print("[DIAG] Camera released.")


if __name__ == "__main__":
    start_diagnostic_photos()
