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
    lower_blue = np.array([100, 120, 80])
    upper_blue = np.array([125, 255, 255])

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

def detect_white_object(bgr_img): #used to detect the package
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

    return annotated, (x, y, w, h, cx, cy)



if __name__ == "__main__":
    img = cv2.imread("images/package drop off.jpg")
    annotated_img, info = detect_blue_object(img)
    print("Center face info:", info)

    resized = cv2.resize(annotated_img, None, fx=0.5, fy=0.5)
    cv2.imshow("Center Face Detection", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
