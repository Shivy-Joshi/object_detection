import cv2
import numpy as np


def detect_blue_object(bgr_img):
    """
    Input:
        bgr_img: OpenCV image in BGR format

    Output:
        annotated_img: image with box/center drawn
        info:
            (x, y, w, h, cx, cy, ex_rel, ey_rel, angle_rel)
            or None if nothing found

        where:
            (x, y, w, h)   = bounding box (top-left, width, height)
            (cx, cy)       = center of the box (pixels)
            ex_rel         = relative x error: (cx - frame_cx) / frame_cx
            ey_rel         = relative y error: (cy - frame_cy) / frame_cy
            angle_rel      = relative “yaw-like” error from left vs right height
    """

    # 1) Convert BGR -> HSV
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # 2) HSV range for blue (we'll tune later if needed)
    lower_blue = np.array([100, 120, 80])
    upper_blue = np.array([125, 255, 255])

    # 3) Threshold to get only blue pixels
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 4) Clean up noise (morphological operations)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove small blobs
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # close small gaps

    # 5) Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return bgr_img, None

    # 6) Take the largest contour (assume that's our blue object)
    c = max(contours, key=cv2.contourArea)

    # optional safety: ignore tiny contours
    if cv2.contourArea(c) < 500:
        return bgr_img, None

    x, y, w, h = cv2.boundingRect(c)

    # 7) Draw bounding box
    annotated = bgr_img.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 8) Compute and draw center point
    cx = x + w // 2
    cy = y + h // 2
    cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

    # 9) Relative centering error (same as before)
    img_h, img_w = bgr_img.shape[:2]
    frame_cx = img_w // 2
    frame_cy = img_h // 2

    err_x_pix = cx - frame_cx
    err_y_pix = cy - frame_cy

    ex_rel = err_x_pix / frame_cx
    ey_rel = err_y_pix / frame_cy

    # 10) Angle-like error from left vs right height
    # Take the object mask inside the bounding box
    submask = mask[y:y + h, x:x + w]

    col_heights = []
    for col in range(w):
        column = submask[:, col]
        ys = np.where(column > 0)[0]
        if len(ys) > 0:
            col_heights.append(ys.max() - ys.min())
        else:
            col_heights.append(0)

    col_heights = np.array(col_heights, dtype=np.float32)

    # Split into left and right regions (here: first and last quarter of bbox)
    if w >= 4:
        left_region = col_heights[: w // 4]
        right_region = col_heights[3 * w // 4 :]
    else:
        left_region = col_heights
        right_region = col_heights

    left_nonzero = left_region[left_region > 0]
    right_nonzero = right_region[right_region > 0]

    if len(left_nonzero) == 0 or len(right_nonzero) == 0:
        h_left = 0.0
        h_right = 0.0
        angle_rel = 0.0
    else:
        h_left = float(left_nonzero.mean())
        h_right = float(right_nonzero.mean())
        avg_h = (h_left + h_right) / 2.0
        if avg_h > 0:
            angle_rel = (h_left - h_right) / avg_h
        else:
            angle_rel = 0.0

    # 11) VISUAL DEBUG OVERLAYS

    # draw the left/right region boundaries in magenta
    if w >= 4:
        left_x = x + w // 4
        right_x = x + 3 * w // 4
        cv2.line(annotated, (left_x, y), (left_x, y + h), (255, 0, 255), 1)
        cv2.line(annotated, (right_x, y), (right_x, y + h), (255, 0, 255), 1)

    # show text with left/right heights and angle_rel above the box
    text_y = max(y - 10, 0)
    cv2.putText(
        annotated,
        f"L:{h_left:.0f} R:{h_right:.0f} ang:{angle_rel:.2f}",
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # also draw a small arrow indicating which side appears "closer"
    # arrow from center of box, pointing left or right
    arrow_length = int(w * 0.2 * abs(angle_rel))  # scale with magnitude
    if arrow_length > 0:
        if angle_rel > 0:
            # left side taller => arrow to left
            end_x = cx - arrow_length
        else:
            # right side taller => arrow to right
            end_x = cx + arrow_length
        cv2.arrowedLine(
            annotated,
            (cx, cy),
            (end_x, cy),
            (0, 255, 255),
            2,
            tipLength=0.3,
        )

    return annotated, (x, y, w, h, cx, cy, ex_rel, ey_rel, angle_rel)


def detect_white_object(bgr_img): #used to detect the package
    """
        Input:
            bgr_img: OpenCV image in BGR format

        Output:
            annotated_img: image with box/center drawn
            info:
                (x, y, w, h, cx, cy, ex_rel, ey_rel)
                or None if nothing found

            where:
                (x, y, w, h)   = bounding box (top-left, width, height)
                (cx, cy)       = center of the box (pixels)
                ex_rel         = relative x error: (cx - frame_cx) / frame_cx
                ey_rel         = relative y error: (cy - frame_cy) / frame_cy
                                (0 means centered, ±1 ≈ left/right or top/bottom edge)
        """

    # 1) Convert BGR -> HSV
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # 2) HSV range for blue (we'll tune later if needed)
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180, 40, 255])

    # 3) Threshold to get only blue pixels
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 4) Clean up noise (morphological operations)
    kernel = np.ones((5, 5), np.uint8) # structuring element for morphological ops
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove small blobs (Erosion → Dilation)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # close small gaps (Dilation → Erosion)

    # 5) Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return bgr_img, None

    # 6) Take the largest contour (assume that's our blue object)
    c = max(contours, key=cv2.contourArea)

    # optional safety: ignore tiny contours
    if cv2.contourArea(c) < 500:
        return bgr_img, None

    x, y, w, h = cv2.boundingRect(c)

    # 7) Draw bounding box
    annotated = bgr_img.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 8) Compute and draw center point
    cx = x + w // 2
    cy = y + h // 2
    cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

    # after computing cx, cy:
    img_h, img_w = bgr_img.shape[:2]
    frame_cx = img_w // 2
    frame_cy = img_h // 2

    err_x_pix = cx - frame_cx
    err_y_pix = cy - frame_cy

    ex_rel = err_x_pix / frame_cx
    ey_rel = err_y_pix / frame_cy

    # 10) Angle-like error from left vs right height
    # Take the object mask inside the bounding box
    submask = mask[y:y + h, x:x + w]

    col_heights = []
    for col in range(w):
        column = submask[:, col]
        ys = np.where(column > 0)[0]
        if len(ys) > 0:
            col_heights.append(ys.max() - ys.min())
        else:
            col_heights.append(0)

    col_heights = np.array(col_heights, dtype=np.float32)

    # Split into left and right regions (e.g., first & last quarter of bbox)
    if w >= 4:
        left_region = col_heights[: w // 4]
        right_region = col_heights[3 * w // 4:]
    else:
        # too narrow, bail out
        left_region = col_heights
        right_region = col_heights

    # Use mean height in each region, ignoring zeros
    left_nonzero = left_region[left_region > 0]
    right_nonzero = right_region[right_region > 0]

    if len(left_nonzero) == 0 or len(right_nonzero) == 0:
        angle_rel = 0.0  # can't estimate reliably
    else:
        h_left = float(left_nonzero.mean())
        h_right = float(right_nonzero.mean())
        avg_h = (h_left + h_right) / 2.0
        if avg_h > 0:
            angle_rel = (h_left - h_right) / avg_h
        else:
            angle_rel = 0.0

    return annotated, (x, y, w, h, cx, cy, ex_rel, ey_rel, angle_rel)


if __name__ == "__main__":
    img = cv2.imread("images/package drop off.jpg")
    annotated_img, info = detect_blue_object(img)
    print("Center face info:", info)

    resized = cv2.resize(annotated_img, None, fx=0.5, fy=0.5)
    cv2.imshow("Center Face Detection", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
