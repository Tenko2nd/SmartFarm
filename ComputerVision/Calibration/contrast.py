import cv2
import numpy as np

# Callback function (required by trackbar but doesn't need to do anything)
def nothing(x):
    pass

# 1. Load your image
# Replace 'image.jpg' with the actual path to your file
image_path = '../calibrated_captures/20260330-000100/SIDE_20260330-000100_undistorted.jpg'
original = cv2.imread(image_path)

if original is None:
    print("Error: Could not open or find the image.")
    exit()

# 2. Create a window for the sliders
cv2.namedWindow('Adjustments')

# 3. Create Trackbars
# Contrast (Alpha): 0 to 300 (we divide by 100 to get 0.0 to 3.0)
cv2.createTrackbar('Contrast', 'Adjustments', 100, 300, nothing)

# Brightness (Beta): 0 to 200 (we subtract 100 to get -100 to 100)
cv2.createTrackbar('Brightness', 'Adjustments', 100, 200, nothing)

print("Controls: Adjust sliders. Press 's' to save or 'q' to quit.")

while True:
    # Get current positions of trackbars
    alpha = cv2.getTrackbarPos('Contrast', 'Adjustments') / 100.0
    beta = cv2.getTrackbarPos('Brightness', 'Adjustments') - 100

    # 4. Apply adjustments
    # convertScaleAbs handles the formula and clips values to [0, 255] automatically
    adjusted = cv2.convertScaleAbs(original, alpha=alpha, beta=beta)

    # 5. Show the image
    cv2.imshow('Adjusted Image', adjusted)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): # Quit
        break
    if key == ord('s'): # Save the result
        cv2.imwrite('adjusted_result.jpg', adjusted)
        print("Image saved as 'adjusted_result.jpg'")

cv2.destroyAllWindows()