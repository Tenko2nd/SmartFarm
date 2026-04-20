import cv2
import os
import numpy as np

# --- 1. CHOOSE YOUR IMAGE ---
BASE_PATH = '../calibrated_captures'
# This automatically finds the first "SIDE" image in the first folder
IMAGE_PATH = r"../calibrated_captures/20260330-080100/SIDE_20260330-080100_undistorted.jpg"

if not os.path.exists(IMAGE_PATH):
    print(f"Error: Could not find image at {IMAGE_PATH}")
    exit()


def nothing(x):
    pass


# Load image
img = cv2.imread(IMAGE_PATH)
# Resize if too large for screen
h, w = img.shape[:2]
if w > 1000:
    img = cv2.resize(img, (int(w / 2), int(h / 2)))

# Create window and trackbars
cv2.namedWindow('HSV Tuner', cv2.WINDOW_AUTOSIZE)

# HSV Ranges: H (0-179), S (0-255), V (0-255)
cv2.createTrackbar('Low H', 'HSV Tuner', 35, 179, nothing)
cv2.createTrackbar('High H', 'HSV Tuner', 85, 179, nothing)
cv2.createTrackbar('Low S', 'HSV Tuner', 40, 255, nothing)
cv2.createTrackbar('High S', 'HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('Low V', 'HSV Tuner', 40, 255, nothing)
cv2.createTrackbar('High V', 'HSV Tuner', 255, 255, nothing)

print("--- INSTRUCTIONS ---")
print("1. Adjust sliders until ONLY the plants are white.")
print("2. Press 'q' to exit and print final values.")

while True:
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get trackbar positions
    l_h = cv2.getTrackbarPos('Low H', 'HSV Tuner')
    u_h = cv2.getTrackbarPos('High H', 'HSV Tuner')
    l_s = cv2.getTrackbarPos('Low S', 'HSV Tuner')
    u_s = cv2.getTrackbarPos('High S', 'HSV Tuner')
    l_v = cv2.getTrackbarPos('Low V', 'HSV Tuner')
    u_v = cv2.getTrackbarPos('High V', 'HSV Tuner')

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])

    # Create mask
    mask = cv2.inRange(hsv, lower, upper)

    # Show results
    # Bitwise-AND mask and original image to show colored plants
    result = cv2.bitwise_and(img, img, mask=mask)

    # Stack images horizontally to compare
    stacked = np.hstack([img, result])
    cv2.imshow('HSV Tuner', stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Output the final result for easy copy-pasting
print("\n--- FINAL VALUES FOR YOUR MAIN SCRIPT ---")
print(f"LOWER_GREEN = np.array([{l_h}, {l_s}, {l_v}])")
print(f"UPPER_GREEN = np.array([{u_h}, {u_s}, {u_v}])")