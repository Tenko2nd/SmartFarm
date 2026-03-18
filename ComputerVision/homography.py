import cv2
import numpy as np
from config_cv import OUTPUT_SIZE, TOP_LEFT_ID, TOP_RIGHT_ID, BOTTOM_RIGHT_ID, BOTTOM_LEFT_ID, REQUIRED_HOMOGRAPHY_IDS


def get_homography_matrix(frame, detector):
    """
    Attempts to find all 4 markers in the given frame.
    Returns the Perspective Matrix if successful, otherwise returns None.
    """
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is None:
        return None

    ids_flat = ids.flatten()

    # Check if ALL 4 required IDs are visible
    if not all(req_id in ids_flat for req_id in REQUIRED_HOMOGRAPHY_IDS):
        return None

    marker_centers = {}
    for i, marker_id in enumerate(ids_flat):
        if marker_id in REQUIRED_HOMOGRAPHY_IDS:
            marker_corners = corners[i][0]
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))
            marker_centers[marker_id] = [center_x, center_y]

    # Organize points strictly (TL, TR, BR, BL)
    src_pts = np.array([
        marker_centers[TOP_LEFT_ID],
        marker_centers[TOP_RIGHT_ID],
        marker_centers[BOTTOM_RIGHT_ID],
        marker_centers[BOTTOM_LEFT_ID]
    ], dtype=np.float32)

    dst_pts = np.array([[0, 0],
                        [OUTPUT_SIZE[1] - 1, 0], [OUTPUT_SIZE[1] - 1, OUTPUT_SIZE[0] - 1], [0, OUTPUT_SIZE[0] - 1]
                        ], dtype=np.float32)

    # Return the calculation matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return matrix