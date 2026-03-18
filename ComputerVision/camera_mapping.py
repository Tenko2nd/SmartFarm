import cv2
import time
import os
from config_cv import SIDE_MARKERS, TOP_MARKERS, CALIBRATION_DIR


def get_cam_type(cam_id):
    if cam_id in SIDE_MARKERS:
        return "SIDE"
    elif cam_id in TOP_MARKERS:
        return "TOP"
    return None


def map_cameras_with_aruco(max_to_check=5, scan_time_sec=5.0):
    camera_mapping = {}

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    print(f"Scanning up to {max_to_check} USB ports for ArUco markers (max {scan_time_sec}s per port)...")

    for idx in range(max_to_check):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue

        print(f"  -> Camera found at index {idx}, scanning for markers...")

        start_time = time.time()
        found_cam_type = None

        # Look for 5 seconds
        while (time.time() - start_time) < scan_time_sec:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            corners, ids, rejected = detector.detectMarkers(frame)
            if ids is not None:
                marker_id = ids[0][0]
                found_cam_type = get_cam_type(marker_id)

                if found_cam_type:
                    print(
                        f"     [SUCCESS] Marker '{marker_id}' found! Mapped to '{found_cam_type}' camera on port {idx}.")
                    camera_mapping[found_cam_type] = idx
                    break  # Stop looking, move to next port

        if not found_cam_type:
            print(f"     [FAILED] No relevant ArUco marker detected on index {idx} after {scan_time_sec}s.")

        cap.release()

    return camera_mapping


def load_calibration_params(cam_type):
    """Loads distortion matrices given a camera type ('SIDE' or 'TOP')."""
    yaml_file = os.path.join(CALIBRATION_DIR, f"Camera_{cam_type}", "params.yaml")

    cv_file = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
    if cv_file.isOpened():
        mtx = cv_file.getNode("camera_matrix").mat()
        dist = cv_file.getNode("dist_coeff").mat()
        cv_file.release()
        return mtx, dist
    else:
        print(f"  [WARNING] Could not find calibration file: {yaml_file}")
        return None, None


def undistort_frame(frame, mtx, dist):
    if mtx is None or dist is None:
        return frame  # Return original if no parameters are loaded

    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    return undistorted_frame