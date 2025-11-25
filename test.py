import time
from picamera2 import Picamera2
from datetime import datetime


def start_diagnostic_photos(save_folder="diagnostics"):
    """
    Takes a photo every 5 seconds using Pi Camera 3 (Picamera2).
    Images are saved with timestamp names in the specified folder.
    """

    import os
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    picam = Picamera2()
    picam.configure(
        picam.create_still_configuration(main={"size": (640, 480)})  # change size if you want
    )
    picam.start()
    print(f"[DIAG] Diagnostic photo capture started. Saving to: {save_folder}/")

    try:
        while True:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_folder}/diag_{timestamp}.jpg"

            frame = picam.capture_array()
            from cv2 import imwrite
            imwrite(filename, frame)

            print(f"[DIAG] Saved {filename}")
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n[DIAG] Diagnostic capture stopped.")

    finally:
        picam.stop()
        print("[DIAG] Camera released.")

