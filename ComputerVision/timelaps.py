import cv2
import os

# --- SETTINGS ---
base_path = 'calibrated_captures'
display_delay = 500  # ms
target_width = 800  # Set your desired display width (height will auto-calculate)


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size
    (h, w) = image.shape[:2]
    # If both are None, return original
    if width is None and height is None:
        return image
    # Calculate the ratio and new dimensions
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # Resize the image
    return cv2.resize(image, dim, interpolation=inter)


# Get sorted subfolders
subfolders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])

# Create the window (AUTOSIZE prevents manual stretching)
window_name = "Corrected Side View"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

for folder in subfolders:
    folder_path = os.path.join(base_path, folder)
    files = os.listdir(folder_path)
    side_image_name = next((f for f in files if "SIDE" in f and f.endswith(".jpg")), None)

    if side_image_name:
        img = cv2.imread(os.path.join(folder_path, side_image_name))

        if img is not None:
            # RESIZE while keeping ratio so it fits your screen nicely
            img_display = resize_with_aspect_ratio(img, width=target_width)

            # Add timestamp label
            cv2.putText(img_display, folder, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(window_name, img_display)

            if cv2.waitKey(display_delay) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()