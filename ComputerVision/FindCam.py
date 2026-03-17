import cv2 as cv
import time
import numpy as np

map_cam_id = {"SIDE": [1], "TOP": [2,3,4,5]}

def map_id_cam(cam_id):
    match cam_id:
        case 1:
            return "SIDE"
        case 2|3|4|5:
            return "TOP"
        case _:
            return None

def map_cameras_with_aruco(max_to_check=5):
    """
    Checks camera indices from 0 to max_to_check.
    Returns a dictionary mapping the ArUco ID to the camera index.
    Example output: {1: 2, 2: 0} -> (Marker ID 1 is on Camera index 2)
    """
    camera_mapping = {}

    # Setup ArUco detector (using a 4x4 dictionary, which has 50 unique IDs)
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)

    print("Scanning USB ports for ArUco markers...")

    for idx in range(max_to_check):
        cap = cv.VideoCapture(idx, cv.CAP_DSHOW)

        if not cap.isOpened():
            continue  # No camera found here

        print(f"  -> Camera found at index {idx}, warming up...")

        # 2. THE CALIBRATION STEP
        start_warmup = time.time()
        while (time.time() - start_warmup) < 2:
            ret, _ = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

        # Grab a frame to process
        ret, frame = cap.read()
        if ret:
            # Detect ArUco markers
            corners, ids, rejected = detector.detectMarkers(frame)
            print(ids)

            if ids is not None:
                # Get the first marker ID detected in this camera
                marker_id = ids[0][0]
                cam = map_id_cam(marker_id)
                print(f"     [SUCCESS] ArUco ID '{marker_id}' found on USB index {idx}!")
                camera_mapping[cam] = idx
            else:
                print(f"     [FAILED] No ArUco marker detected on index {idx}.")

        cap.release()

    return camera_mapping


# --- Main Execution ---
if __name__ == "__main__":

    # 1. Map the cameras using ArUco IDs
    my_cameras = map_cameras_with_aruco(max_to_check=5)

    print("\n--- Final Camera Mapping ---")
    print(my_cameras)

    # Let's say we are looking for the camera that has Marker ID 1
    TARGET_CAM = "SIDE"

    if TARGET_CAM in my_cameras:
        cam_index = my_cameras[TARGET_CAM]
        print(f"\nTarget Camera (Marker {TARGET_CAM}) is on USB index {cam_index}")

        # 2. Open the camera
        cap = cv.VideoCapture(cam_index)

        # 3. Load the specific calibration file for this camera!
        # (Assuming you saved it as camera_1_params.yaml)
        yaml_file = r"C:\Users\mathi\PycharmProjects\SmartFarm\ComputerVision\Calibration\calibration_images"+f"/Camera_{TARGET_CAM}/params.yaml"
        print(f"Loading calibration from: {yaml_file}")

        cv_file = cv.FileStorage(yaml_file, cv.FILE_STORAGE_READ)
        if cv_file.isOpened():
            mtx = cv_file.getNode("camera_matrix").mat()
            dist = cv_file.getNode("dist_coeff").mat()
            cv_file.release()
            print("Camera parameters loaded successfully.")

            # --- SCALE CALCULATION ---
            KNOWN_MARKER_SIZE_MM = 30.0  # Your physical marker size

            # Setup ArUco detector
            dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
            parameters = cv.aruco.DetectorParameters()
            detector = cv.aruco.ArucoDetector(dictionary, parameters)

            # 2. THE CALIBRATION STEP
            start_warmup = time.time()
            while (time.time() - start_warmup) < 5:
                ret, _ = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

            # Grab the actual frame for measurement
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]

                # 1. Undistort the frame before doing any measurements!
                newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                undistorted_frame = cv.undistort(frame, mtx, dist, None, newcameramtx)

                undistorted_frame = cv.rotate(undistorted_frame, cv.ROTATE_180)
                # 2. Detect the ArUco marker in the undistorted image
                corners, ids, rejected = detector.detectMarkers(undistorted_frame)

                marker_id = map_cam_id[TARGET_CAM][0]

                # 3. Check if we found markers, and specifically OUR target marker
                if ids is not None and marker_id in ids:

                    # Find which index corresponds to our specific target marker
                    marker_index = np.where(ids == marker_id)[0][0]
                    marker_corners = corners[marker_index][0]

                    # Corners are returned as: [top-left, top-right, bottom-right, bottom-left]
                    top_left = marker_corners[0]
                    top_right = marker_corners[1]

                    # Calculate the width of the marker in pixels (Euclidean distance)
                    pixel_width = np.linalg.norm(top_right - top_left)

                    # Calculate the crucial conversion factor
                    mm_per_pixel = KNOWN_MARKER_SIZE_MM / pixel_width

                    print("\n--- Measurement Scale Established ---")
                    print(f"Marker Width in Image: {pixel_width:.2f} pixels")
                    print(f"Measurement Scale:     {mm_per_pixel:.4f} mm/pixel")

                    # Optional: Draw it and show it to verify it works
                    cv.aruco.drawDetectedMarkers(undistorted_frame, corners, ids)
                    cv.imshow(f"Camera {TARGET_CAM} - Scale Check", undistorted_frame)
                    cv.waitKey(15000)  # Show for 2 seconds
                    cv.destroyAllWindows()

                    # --> From here, you can start your main loop cap.read()
                    # --> and multiply any pixel length by `mm_per_pixel` to get mm!

                else:
                    print(f"\nError: Could not see ArUco Marker {TARGET_CAM} to calculate scale.")

            else:
                print("\nError: Could not grab a frame from the camera.")

        else:
            print(f"Could not find {yaml_file}")

        cap.release()
    else:
        print(f"\nWarning: Could not find Camera with Marker ID {TARGET_CAM}.")