import cv2
import numpy as np


def detect_blue_object(bgr_img):
    """
    Input:
        bgr_img: OpenCV image in BGR format
    Output:
        annotated_img: image with box/center drawn
        info: (cx, cy, w, h) or None if nothing found
    """

    # 1) Convert BGR -> HSV
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # 2) HSV range for blue (we'll tune later if needed)
    lower_blue = np.array([90, 80, 40])
    upper_blue = np.array([130, 255, 255])

    # 3) Threshold to get only blue pixels
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

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

    return annotated, (x, y, w, h, cx, cy)


def detect_center_face(bgr_img):
    """
    Returns:
        annotated_img: original image with center face box drawn
        info: (cx_face, cy_face, w_face, h_face) or None
    """
    annotated, base_info = detect_blue_object(bgr_img)
    if base_info is None:
        return annotated, None

    x, y, w, h, _, _ = base_info

    # 1) Crop a central ROI where the middle box lives
    roi_x1 = int(x + 0.2 * w)
    roi_x2 = int(x + 0.8 * w)
    roi_y1 = int(y + 0.1 * h)
    roi_y2 = int(y + 0.95 * h)

    roi = bgr_img[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return annotated, None

    # 2) Edge detection on the ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    # 3) Contours on edges -> likely rectangles
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return annotated, None

    # biggest contour in this central region ~ center face
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 200:
        return annotated, None

    rx, ry, rw, rh = cv2.boundingRect(c)

    # 4) Map ROI coords back to full image coords
    face_x1 = roi_x1 + rx
    face_y1 = roi_y1 + ry
    face_x2 = face_x1 + rw
    face_y2 = face_y1 + rh

    cx_face = face_x1 + rw // 2
    cy_face = face_y1 + rh // 2

    # draw face box on annotated image
    cv2.rectangle(annotated, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)
    cv2.circle(annotated, (cx_face, cy_face), 4, (0, 255, 255), -1)

    return annotated, (cx_face, cy_face, rw, rh)

if __name__ == "__main__":
    img = cv2.imread("images/package drop off.jpg")
    annotated_img, info = detect_center_face(img)
    print("Center face info:", info)

    resized = cv2.resize(annotated_img, None, fx=0.5, fy=0.5)
    cv2.imshow("Center Face Detection", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
