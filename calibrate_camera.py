def find_white_object(frame_bgr):
    """
    Segment a white-ish object using HSV thresholding and pick
    a roughly box-shaped blob (to avoid the table top).

    Returns:
        bbox: (x, y, w, h) of chosen white blob, or None
        vis:  frame with drawing/overlay
    """
    vis = frame_bgr.copy()

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Threshold for "white-ish": low saturation, high value
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Clean up noise a bit
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, vis

    best_bbox = None
    best_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 500:          # ignore tiny noise
            continue

        aspect = w / float(h)

        # Reject very flat / table-like blobs (very wide and short)
        if aspect > 3.0:
            continue

        # Reject extremely tall skinny blobs
        if aspect < 0.3:
            continue

        # This looks like a box-ish object, keep the largest
        if area > best_area:
            best_area = area
            best_bbox = (x, y, w, h)

    if best_bbox is None:
        return None, vis

    x, y, w, h = best_bbox

    # Draw visualization: box + height text
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        vis,
        f"h_pixels = {h}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return best_bbox, vis
