import cv2
import os

camera_idx = 2
# Create a directory to save the checkerboard images
save_dir = f"calibration_images/Camera_{camera_idx}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cap = cv2.VideoCapture(camera_idx)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press [SPACE] to take a picture.")
print("Press [ESC] or[q] to quit.")

img_counter = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Show the live camera feed
    cv2.imshow("Checkerboard Capture", frame)

    # Wait for a key press for 1 millisecond
    k = cv2.waitKey(1)

    if k % 256 == 27 or k == ord('q'):
        # ESC or 'q' pressed - exit the program
        print("Closing the camera.")
        break
    elif k % 256 == 32:
        # SPACEBAR pressed - save the image
        img_name = os.path.join(save_dir, f"checkerboard_{img_counter:02d}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        img_counter += 1

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()