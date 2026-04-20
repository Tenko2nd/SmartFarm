from warnings import deprecated

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go
import os
# from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    BASE_PATH = 'calibrated_captures'
    MARKER_SIZE_CM = 3.0
    TARGET_ID = 1

    # === HEIGHT CALCULATION ===
    # HSV Thresholds
    LOWER_GREEN = np.array([36, 40, 0])
    UPPER_GREEN = np.array([60, 255, 167])

    # Filtering
    MIN_PLANT_AREA = 100
    ROI_X_BUFFER = 10  # Pixels to offset from marker
    MIN_TOTAL_CLUSTER_AREA = 500  # Minimum sum of all pixels in the cluster
    MORPH_KERNEL_SIZE = (3,3)
    MAX_GROWTH_STEP_CM = 5.0      # Max allowed height increase between frames
    MIN_BRIGHTNESS = 40 # for night images ignore

    # YOLO Settings
    MODEL_NAME = 'yolov8n-seg.pt'
    PLANT_CLASS_ID = 58  # COCO dataset ID for "potted plant"
    CONF_THRESHOLD = 0.25

    # === GAI CALCULATION ===
    K_EXTINCTION = 0.6  # Beer-Lambert constant
    ALPHA_GAI = 0.5    # Empirical height/volume coefficient
    MIN_BRIGHTNESS_GAI = 3
    MAX_GROWTH_CC_PCT = 0.04
    GAI_MARKER_MARGIN = 45

    # Visualization
    DISPLAY_DELAY =200
    SAVE_GRAPH_NAME = r'OUTPUT\plant_growth_curve.png'


# ==========================================
# ANALYSIS ENGINE
# ==========================================
class PlantAnalyzer:
    def __init__(self):
        # Initialize ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Memory (State)
        self.ppcm = None  # Pixels Per CM
        self.x_boundary = None  # X-coordinate limit for ROI
        self.last_height = 0.0  # Last valid height measurement
        self.last_gai = 0.0
        self.last_cc = 0.0

        self.model = None # YOLO(Config.MODEL_NAME)

    # === SIDE VIEW METHODS ===
    def _update_calibration(self, frame):
        """Detects ArUco markers but only updates using ID 1."""
        corners, ids, _ = self.detector.detectMarkers(frame)

        if ids is not None:
            # Flatten the ids list to make it easier to search
            ids_flat = ids.flatten()

            # Check if our specific target ID is in the list
            if Config.TARGET_ID in ids_flat:
                # Find the index of Target ID 1
                idx = np.where(ids_flat == Config.TARGET_ID)[0][0]

                # Get the corners specifically for marker ID 1
                marker_pts = corners[idx][0]

                # Calculate Pixels Per CM
                px_height = np.linalg.norm(marker_pts[0] - marker_pts[3])
                self.ppcm = px_height / Config.MARKER_SIZE_CM

                # Calculate ROI Boundary (Left edge of marker 1)
                self.x_boundary = int(np.min(marker_pts[:, 0])) - Config.ROI_X_BUFFER

                # We return only the corner/id for our specific marker for drawing
                return [corners[idx]], np.array([[Config.TARGET_ID]])

        return None, None

    def _get_plant_bbox_hsv(self, frame):
        h_img, w_img = frame.shape[:2]
        mid_y = h_img / 2

        # 1. BRIGHTNESS CHECK (Is it night?)
        hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv_full[:,:,2])
        if avg_brightness < Config.MIN_BRIGHTNESS:
            return None, 0 # Too dark to see anything

        # 2. ROI and Masking
        limit = self.x_boundary if self.x_boundary else int(w_img * 0.7)
        roi = frame[:, 0:limit]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, Config.LOWER_GREEN, Config.UPPER_GREEN)

        kernel = np.ones(Config.MORPH_KERNEL_SIZE, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 3. Find Stalks
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_stalks = []
        centers_x = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > Config.MIN_PLANT_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                if (y + h) > mid_y:
                    valid_stalks.append({'bbox': (x, y, w, h), 'area': area})
                    centers_x.append(x + w//2)

        if not valid_stalks:
            return None, limit

        # 4. Clustering (Median X)
        median_x = np.median(centers_x)
        max_spread = w_img * 0.25
        cluster_parts = [s for s in valid_stalks if abs((s['bbox'][0] + s['bbox'][2]//2) - median_x) < max_spread]

        # 5. DENSITY CHECK (Noise Filter)
        total_cluster_area = sum([s['area'] for s in cluster_parts])
        if total_cluster_area < Config.MIN_TOTAL_CLUSTER_AREA:
            return None, limit # Not enough "green mass" to be a plant

        # 6. Group Bounding Box
        min_x = min([s['bbox'][0] for s in cluster_parts])
        min_y = min([s['bbox'][1] for s in cluster_parts])
        max_x = max([s['bbox'][0] + s['bbox'][2] for s in cluster_parts])
        max_y = max([s['bbox'][1] + s['bbox'][3] for s in cluster_parts])

        return (min_x, min_y, max_x - min_x, max_y - min_y), limit

    @deprecated("Use _get_plant_bbox_hsv instead")
    def _get_plant_bbox_ai(self, frame):
        """Uses AI to segment plants.
            Doesn't work!!"""

        # 1. Apply Spatial ROI (Ignore background noise)
        limit = self.x_boundary if self.x_boundary else int(frame.shape[1] * 0.7)
        roi = frame[:, 0:limit]

        # 2. RUN AI PREDICTION
        # We tell YOLO to only look for class 58 (plants)
        results = self.model.predict(
            source=roi,
            classes=[Config.PLANT_CLASS_ID],
            conf=Config.CONF_THRESHOLD,
            verbose=False  # Keeps the terminal clean
        )

        # 3. EXTRACT CONTOURS/MASKS
        # Results is a list; we take the first element [0]
        result = results[0]

        # If the AI found a plant:
        if result.masks is not None:
            # Get all detected masks (contours)
            # xyn: coordinates normalized (0-1), xy: coordinates in pixels
            all_masks = result.masks.xy

            # Combine all points from all detected plants into one array
            all_pts = np.concatenate(all_masks).astype(np.int32)

            # Calculate the overall bounding box
            bbox = cv2.boundingRect(all_pts)
            return bbox, limit

        return None, limit

    def process_frame_side(self, frame):
        """Main pipeline for a single frame."""
        display_frame = frame.copy()

        # Step 1: Calibration
        corners, ids = self._update_calibration(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids, (255, 0, 0))

        # Step 2: Plant Detection
        bbox, x_limit = self._get_plant_bbox_hsv(frame)

        new_h = self.last_height  # Default to last height

        if bbox and self.ppcm:
            x, y, w, h = bbox
            detected_h = h / self.ppcm

            # If the plant "grew" too fast, it's probably noise.
            if self.last_height > 0 and abs(detected_h - self.last_height) > Config.MAX_GROWTH_STEP_CM:
                new_h = self.last_height  # Ignore the jump
            else:
                new_h = detected_h
                self.last_height = new_h

            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Step 3: Visual overlays
        cv2.line(display_frame, (x_limit, 0), (x_limit, frame.shape[0]), (255, 255, 0), 2)
        cv2.putText(display_frame, f"H: {new_h:.2f}cm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return display_frame, new_h

    # === TOP VIEW METHODS (GAI) ===
    def calculate_gai(self, top_frame, current_height):
        """Implements your GAI logic using ExG and Otsu."""
        if top_frame is None:
            return self.last_gai, self.last_cc, None

        # Black out the 4 corners [y1:y2, x1:x2]
        clean_top = top_frame.copy()
        m = Config.GAI_MARKER_MARGIN
        h, w = clean_top.shape[:2]

        clean_top[0:m, 0:m] = 0  # Top-Left
        clean_top[0:m, w - m:w] = 0  # Top-Right
        clean_top[h - m:h, 0:m] = 0  # Bottom-Left
        clean_top[h - m:h, w - m:w] = 0  # Bottom-Right

        hsv_top = cv2.cvtColor(clean_top, cv2.COLOR_BGR2HSV)
        avg_v = np.mean(hsv_top[:, :, 2])

        if avg_v < Config.MIN_BRIGHTNESS_GAI or current_height <= 0.01:
            # Create an empty black mask for visualization
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return self.last_gai, self.last_cc, empty_mask

        # 1. Excess Green Index (ExG)
        b, g, r = cv2.split(clean_top.astype(float))
        exg = 2 * g - r - b

        # 2. Otsu Thresholding
        exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        otsu_threshold, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Canopy Cover (CC)
        cc = np.count_nonzero(mask) / mask.size

        if self.last_cc > 0 and abs(cc - self.last_cc) > Config.MAX_GROWTH_CC_PCT:
            cc = self.last_cc  # Ignore the jump
        else:
            self.last_cc = cc

        # 4. GAI Estimation
        cc_clipped = np.clip(cc, 0.01, 0.99)

        gai_2d = -np.log(1 - cc_clipped) / Config.K_EXTINCTION

        # 5. Refine with Side-View Height
        gai_final = Config.ALPHA_GAI * (cc * current_height)

        self.last_gai = round(gai_final, 2)
        self.last_cc = cc

        return round(gai_final, 2), cc, mask


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    analyzer = PlantAnalyzer()

    results = {'time': [], 'height': [], 'gai': [], 'cc': []}

    # Setup Data Source
    folders = sorted([f for f in os.listdir(Config.BASE_PATH) if os.path.isdir(os.path.join(Config.BASE_PATH, f))])

    for folder in folders:
        img_path = os.path.join(Config.BASE_PATH, folder)
        side_img_name = next((f for f in os.listdir(img_path) if "SIDE" in f and f.endswith(".jpg")), None)
        top_img_name  = next((f for f in os.listdir(img_path) if "TOP" in f and f.endswith("homography.jpg")), None)

        if not side_img_name or not top_img_name: continue
        side_img = cv2.imread(os.path.join(img_path, side_img_name))
        top_img = cv2.imread(os.path.join(img_path, top_img_name))

        # Process
        annotated_img, h_cm = analyzer.process_frame_side(side_img)
        gai_val, cc_val, top_mask = analyzer.calculate_gai(top_img, h_cm)

        # Store Data
        time_str = f"{folder[0:4]}-{folder[4:6]}-{folder[6:8]} {folder[9:11]}:00"
        dt_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
        results['time'].append(dt_obj)
        results['height'].append(h_cm)
        results['gai'].append(gai_val)
        results['cc'].append(cc_val)

        # Show
        cv2.imshow("Analysis", annotated_img)
        cv2.imshow("TOP Mask (Vegetation)", top_mask)
        if cv2.waitKey(Config.DISPLAY_DELAY) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Final Plotting
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Create the twin axes
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    # This moves the third y-axis to the right so it doesn't overlap ax2
    ax3.spines.right.set_position(("axes", 1.15))

    # Plot the data
    p1, = ax1.plot(results['time'], results['height'], 'g-', label='Height (cm)')
    p2, = ax2.plot(results['time'], results['gai'], 'b-', label='GAI')
    p3, = ax3.plot(results['time'], results['cc'], 'r-', label='CC')

    # Set labels and colors
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Height', color='g')
    ax2.set_ylabel('GAI', color='b')
    ax3.set_ylabel('CC', color='r')

    # Match tick colors to the line colors for clarity
    ax1.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='b')
    ax3.tick_params(axis='y', labelcolor='r')

    # Adjust the right margin to make room for the 3rd axis
    fig.subplots_adjust(right=0.8)

    # This puts labels in one legend box
    lines = [p1, p2, p3]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')

    plt.title("Combined Plant Growth Analysis")
    plt.show()

    directory = os.path.dirname(Config.SAVE_GRAPH_NAME)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(Config.SAVE_GRAPH_NAME, bbox_inches='tight', dpi=300)
