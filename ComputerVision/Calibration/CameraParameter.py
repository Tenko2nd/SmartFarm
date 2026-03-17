import numpy as np
import cv2 as cv
import glob

camera_idx = 2

# prepare object points
objp = np.zeros((4 * 5, 3), np.float32)
objp[:, :2] = np.mgrid[0:4, 0:5].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(f"calibration_images/Camera_{camera_idx}/*.jpg")

for fname in images:
    img = cv.imread(cv.samples.findFile(fname))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    flags = cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
    ret, corners = cv.findChessboardCornersSB(gray, (4, 5), flags)

    if ret:
        print("current image: ", fname)
        objpoints.append(objp)

        imgpoints.append(corners)

        cv.drawChessboardCorners(img, (4, 5), corners, ret)

        cv.namedWindow('img', cv.WINDOW_NORMAL)
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        print("Failed to find corners in: ", fname)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Define the output file name based on the camera index
output_file = f"calibration_images/Camera_{camera_idx}/params.yaml"

# Open the file in write mode
cv_file = cv.FileStorage(output_file, cv.FILE_STORAGE_WRITE)

# Write the essential camera parameters
cv_file.write("camera_matrix", mtx)
cv_file.write("dist_coeff", dist)

cv_file.release()

print(f"\nFocal length (fx, fy): ({mtx[0][0]:.3f}, {mtx[1][1]:.3f})")
print(f"Optical center (cx, cy): ({mtx[0][2]:.3f}, {mtx[1][2]:.3f})")

img = cv.imread(f"calibration_images/Camera_{camera_idx}/checkerboard_02.jpg")
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv.imwrite(f"etallonage (02).jpg", dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("\ntotal error: {}".format(mean_error / len(objpoints)))