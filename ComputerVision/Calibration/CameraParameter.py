import numpy as np
import cv2 as cv
import glob
import os

camera_idx = 'SIDE'
output_dir = f"calibration_images/Camera_{camera_idx}"
video_name = f"{output_dir}/calibration_grid_view.mp4"

# prepare object points
objp = np.zeros((4 * 5, 3), np.float32)
objp[:, :2] = np.mgrid[0:4, 0:5].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(f"{output_dir}/*.jpg")

# Video Settings
video_writer = None
fps = 8

for fname in images:
    img = cv.imread(cv.samples.findFile(fname))
    if img is None:
        continue

    h, w = img.shape[:2]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Initialize VideoWriter once we know the image size
    if video_writer is None:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(video_name, fourcc, fps, (w, h))

    flags = cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
    ret, corners = cv.findChessboardCornersSB(gray, (4, 5), flags)

    if ret:
        print("current image: ", fname)
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw the corners
        cv.drawChessboardCorners(img, (4, 5), corners, ret)

        # Write the frame to the video
        video_writer.write(img)

        # Show the preview
        cv.namedWindow('img', cv.WINDOW_NORMAL)
        cv.imshow('img', img)
        cv.waitKey(200)
    else:
        print("Failed to find corners in: ", fname)

# Release the video writer
if video_writer is not None:
    video_writer.release()
    print(f"\nVideo saved to: {video_name}")

cv.destroyAllWindows()

# --- Calibration Logic ---
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Define the output file name based on the camera index
    output_file = f"{output_dir}/params.yaml"

    # Open the file in write mode
    cv_file = cv.FileStorage(output_file, cv.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("dist_coeff", dist)
    cv_file.release()

    print(f"\nFocal length (fx, fy): ({mtx[0][0]:.3f}, {mtx[1][1]:.3f})")
    print(f"Optical center (cx, cy): ({mtx[0][2]:.3f}, {mtx[1][2]:.3f})")

    # Undistort Example
    sample_img_path = f"{output_dir}/checkerboard_19.jpg"
    if os.path.exists(sample_img_path):
        img = cv.imread(sample_img_path)
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        cv.imwrite(f"etallonage_19.jpg", dst)

    # Error calculation
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("\ntotal error: {}".format(mean_error / len(objpoints)))
else:
    print("No corners were found. Calibration could not proceed.")