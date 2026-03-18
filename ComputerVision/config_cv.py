import os

# --- ARUCO MARKER CONFIGURATION ---
SIDE_MARKERS = [1]
TOP_MARKERS = [2, 3, 4, 5]

# --- HOMOGRAPHY CONFIGURATION (TOP CAMERA) ---
TOP_LEFT_ID = 4
TOP_RIGHT_ID = 2
BOTTOM_RIGHT_ID = 3
BOTTOM_LEFT_ID = 5

REQUIRED_HOMOGRAPHY_IDS =[TOP_LEFT_ID, TOP_RIGHT_ID, BOTTOM_RIGHT_ID, BOTTOM_LEFT_ID]

# Output size (ratio = 20) -> [Height, Width]
RATIO = 20
OUTPUT_SIZE = [25 * RATIO, 17 * RATIO]

# --- PATHS AND TIMINGS ---
INTERVAL_SEC = 2*3600 # Take pictures every 2 hours
WARMUP_SEC = 5.0  # Seconds to let the camera adjust lighting/focus


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR = os.path.join(CURRENT_DIR, "Calibration", "calibration_images")
SAVE_DIR = os.path.join(CURRENT_DIR, "calibrated_captures")