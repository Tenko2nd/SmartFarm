import cv2 as cv
import numpy as np

# --- 1. CONFIGURATION ---
# Ask the user for the camera index in the terminal
cam_idx_input = 2
CAM_IDX = int(cam_idx_input)

# The specific ArUco IDs we are looking for to make our 4 corners
TOP_LEFT_ID = 4
TOP_RIGHT_ID = 2
BOTTOM_RIGHT_ID =3
BOTTOM_LEFT_ID = 5

REQUIRED_IDS = [TOP_LEFT_ID, TOP_RIGHT_ID, BOTTOM_RIGHT_ID, BOTTOM_LEFT_ID]

# The size of the output "flattened" window (in pixels)
ratio = 20
OUTPUT_SIZE = [25*ratio, 17*ratio]

# Define where we want the 4 markers to go in the final flattened image
# (Matching the Top-Left, Top-Right, Bottom-Right, Bottom-Left order)
dst_pts = np.array([
    [0, 0],  # Top Left
    [OUTPUT_SIZE[1] - 1, 0],                # Top Right
    [OUTPUT_SIZE[1] - 1, OUTPUT_SIZE[0] - 1],  # Bottom Right
    [0, OUTPUT_SIZE[0] - 1]  # Bottom Left
], dtype=np.float32)

# --- 2. SETUP CAMERA AND DETECTOR ---
cap = cv.VideoCapture(CAM_IDX)

if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAM_IDX}")
    exit()

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)

print(f"Camera {CAM_IDX} opened. Looking for markers 0, 1, 2, 3...")
print("Press 'q' to quit.")

yaml_file = r"C:\Users\mathi\PycharmProjects\SmartFarm\ComputerVision\Calibration\calibration_images" + f"/Camera_{CAM_IDX}/params.yaml"

cv_file = cv.FileStorage(yaml_file, cv.FILE_STORAGE_READ)
if cv_file.isOpened():
    mtx = cv_file.getNode("camera_matrix").mat()
    dist = cv_file.getNode("dist_coeff").mat()
    cv_file.release()
# --- 3. MAIN LOOP ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        # 1. Undistort the frame before doing any measurements!
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        frame = cv.undistort(frame, mtx, dist, None, newcameramtx)

    # Detect markers
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            # Draw all found markers for debugging
            cv.aruco.drawDetectedMarkers(frame, corners, ids)

            # Flatten the IDs array for easier checking
            ids_flat = ids.flatten()

            # Check if ALL 4 of our required IDs are currently visible
            if all(req_id in ids_flat for req_id in REQUIRED_IDS):

                # Dictionary to store the center point of each marker
                marker_centers = {}

                for i, marker_id in enumerate(ids_flat):
                    if marker_id in REQUIRED_IDS:
                        # corners[i][0] contains the 4 corners of the marker itself
                        # We average them to find the exact center of the marker
                        marker_corners = corners[i][0]
                        center_x = int(np.mean(marker_corners[:, 0]))
                        center_y = int(np.mean(marker_corners[:, 1]))

                        marker_centers[marker_id] = [center_x, center_y]

                        # Draw a red dot in the center of the marker
                        cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                # Extract the centers in the strictly correct order (TL, TR, BR, BL)
                src_pts = np.array([
                    marker_centers[TOP_LEFT_ID],
                    marker_centers[TOP_RIGHT_ID],
                    marker_centers[BOTTOM_RIGHT_ID],
                    marker_centers[BOTTOM_LEFT_ID]
                ], dtype=np.float32)

                # --- 4. CALCULATE HOMOGRAPHY AND WARP ---
                # Get the Perspective Transform Matrix
                matrix = cv.getPerspectiveTransform(src_pts, dst_pts)

                # Warp the frame to our bird's-eye view flat square!
                warped_image = cv.warpPerspective(frame, matrix, (OUTPUT_SIZE[1], OUTPUT_SIZE[0]))

                # Show the flattened image in a separate window
                cv.imshow("Homography (Flattened View)", warped_image)

            else:
                # If we don't see all 4 markers, we can close the warped window if it exists
                # (or you can just leave this blank so it keeps the last valid frame)
                pass

        # Show the live camera feed
        cv.imshow("Live Camera", frame)

        # Quit condition
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()