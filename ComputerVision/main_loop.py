import cv2
import time
import os
from config_cv import INTERVAL_SEC, WARMUP_SEC, SAVE_DIR, OUTPUT_SIZE
from camera_mapping import map_cameras_with_aruco, load_calibration_params, undistort_frame
from homography import get_homography_matrix


def start_capture_loop(initial_mapping):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    calibrations = {
        "SIDE": load_calibration_params("SIDE"),
        "TOP": load_calibration_params("TOP")
    }

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    current_mapping = initial_mapping.copy()

    # STATE VARIABLE: Holds the last successful homography calculation
    last_known_homography_matrix = None

    print(f"\n[INFO] Loop interval set to {INTERVAL_SEC} seconds.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            # --- START TIMER FOR DYNAMIC SLEEP ---
            cycle_start_time = time.time()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            timestamp_dir = f"{SAVE_DIR}/{timestamp}/"
            os.makedirs(timestamp_dir)
            print(f"\n--- Cycle Start: {timestamp} ---")

            # 1. Re-map cameras to counter USB instability
            print("[INFO] Re-verifying camera mappings...")
            new_mapping = map_cameras_with_aruco(max_to_check=5, scan_time_sec=3.0)

            for cam_type, idx in new_mapping.items():
                current_mapping[cam_type] = idx

            for expected_cam in ["SIDE", "TOP"]:
                if expected_cam not in new_mapping:
                    fallback_idx = current_mapping.get(expected_cam)
                    if fallback_idx is not None:
                        print(f"[WARNING] {expected_cam} marker not seen. Falling back to index {fallback_idx}.")

            # 2. Capture Process
            for cam_type, idx in current_mapping.items():
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    print(f"  [ERROR] Cannot open {cam_type} camera (Index {idx})")
                    continue

                print(f"  [{cam_type} Cam] Warming up for {WARMUP_SEC}s...")

                # Retrieve parameters for undistortion
                mtx, dist = calibrations[cam_type]
                matrix_found_this_cycle = None

                # --- WARMUP & HUNTING PHASE ---
                start_warmup = time.time()
                while (time.time() - start_warmup) < WARMUP_SEC:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue

                    # If this is the TOP camera and we haven't found the markers yet, hunt for them!
                    if cam_type == "TOP" and matrix_found_this_cycle is None:
                        # Frame must be undistorted to accurately calculate homography
                        temp_undistorted = undistort_frame(frame, mtx, dist)
                        matrix_found_this_cycle = get_homography_matrix(temp_undistorted, detector)

                # --- FINAL PICTURE CAPTURE ---
                ret, final_frame = cap.read()
                cap.release()

                if not ret:
                    print(f"  [ERROR] Failed to grab final frame from {cam_type} camera.")
                    continue

                # 3. Process Original Image
                undistorted_final = undistort_frame(final_frame, mtx, dist)
                if cam_type == "SIDE":
                    undistorted_final = cv2.rotate(undistorted_final, cv2.ROTATE_180)
                base_filename = f"{timestamp_dir}/{cam_type}_{timestamp}"

                cv2.imwrite(f"{base_filename}_undistorted.jpg", undistorted_final)
                print(f"  [{cam_type} Cam] Undistorted photo saved.")

                # 4. Process Homography (with fallback logic)
                if cam_type == "TOP":
                    # Update our fallback memory if we successfully found it this cycle
                    if matrix_found_this_cycle is not None:
                        last_known_homography_matrix = matrix_found_this_cycle
                        print("  [TOP Cam] New homography matrix successfully calculated.")
                    else:
                        print("  [WARNING] TOP Cam: Markers missing/obstructed. Using LAST KNOWN homography.")

                    # Apply warp if we have a valid matrix (either new or from a previous hour)
                    if last_known_homography_matrix is not None:
                        warped_img = cv2.warpPerspective(undistorted_final, last_known_homography_matrix,
                                                         (OUTPUT_SIZE[1], OUTPUT_SIZE[0]))
                        cv2.imwrite(f"{base_filename}_homography.jpg", warped_img)
                        print(f"  [TOP Cam] Homography photo saved.")
                    else:
                        print("  [ERROR] TOP Cam: No known homography matrix available to fallback on yet.")

            # --- END TIMER AND CALCULATE SLEEP ---
            elapsed_time = int(time.time() - cycle_start_time)
            # Subtract elapsed time from the intended interval. max(0, x) prevents negative sleep times.
            sleep_time = max(0, INTERVAL_SEC - elapsed_time)

            print(f"\nCycle completed in {elapsed_time:.2f} seconds.")
            print(f"Waiting {sleep_time:.2f} seconds until next scheduled cycle...")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nProcess stopped by user.")


if __name__ == "__main__":
    print("=== Step 1: Initial Camera Identification ===")
    my_cameras = map_cameras_with_aruco(max_to_check=5, scan_time_sec=5.0)

    print("\n=== Initial Camera Mapping ===")
    for cam_type, idx in my_cameras.items():
        print(f" * {cam_type} Camera -> USB Index {idx}")

    print("\n=== Step 2: Camera Loop ===")
    start_capture_loop(my_cameras)