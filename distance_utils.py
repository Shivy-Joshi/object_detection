import json
import os

CALIB_FILE = "distance_calibration.json"

# Physical heights (mm)
H_WHITE_MM = 41.0   # calibration cube height
H_BLUE_MM = 50.0    # blue object height (max)

if not os.path.exists(CALIB_FILE):
    raise FileNotFoundError(
        f"{CALIB_FILE} not found. "
        f"Run calibrate_camera.py first to create it."
    )

with open(CALIB_FILE, "r") as f:
    calib = json.load(f)

REF_DIST_M = calib["ref_distance_m"]        # meters
REF_H_PX = calib["ref_pixel_height"]        # pixels


def estimate_blue_distance(h_blue_px: float) -> float | None:
    """
    Estimate distance (meters) to the BLUE object, using calibration
    from a 41 mm white cube and a 50 mm blue object height.

    Formula:
        D_blue = REF_DIST_M * (H_BLUE_MM / H_WHITE_MM) * (REF_H_PX / h_blue_px)
    """
    if h_blue_px is None or h_blue_px <= 0:
        return None

    scale_factor = H_BLUE_MM / H_WHITE_MM
    dist_m = REF_DIST_M * scale_factor * (REF_H_PX / float(h_blue_px))
    return dist_m
