from warnings import deprecated
from scipy.stats import gaussian_kde

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import plotly.graph_objects as go
import os
# from ultralytics import YOLO

import matplotlib
matplotlib.use('Agg')

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
            return None, 0, None, None # Too dark to see anything

        # 2. ROI and Masking
        limit = self.x_boundary if self.x_boundary else int(w_img * 0.7)
        roi = frame[:, 0:limit]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, Config.LOWER_GREEN, Config.UPPER_GREEN)
        mask_first = mask.copy()

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
            return None, limit, mask_first, mask

        # 4. Clustering (Median X)
        median_x = np.median(centers_x)
        max_spread = w_img * 0.25
        cluster_parts = [s for s in valid_stalks if abs((s['bbox'][0] + s['bbox'][2]//2) - median_x) < max_spread]

        # 5. DENSITY CHECK (Noise Filter)
        total_cluster_area = sum([s['area'] for s in cluster_parts])
        if total_cluster_area < Config.MIN_TOTAL_CLUSTER_AREA:
            return None, limit, mask_first, mask # Not enough "green mass" to be a plant

        # 6. Group Bounding Box
        min_x = min([s['bbox'][0] for s in cluster_parts])
        min_y = min([s['bbox'][1] for s in cluster_parts])
        max_x = max([s['bbox'][0] + s['bbox'][2] for s in cluster_parts])
        max_y = max([s['bbox'][1] + s['bbox'][3] for s in cluster_parts])

        return (min_x, min_y, max_x - min_x, max_y - min_y), limit, mask_first, mask

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
        bbox, x_limit, mask_first, mask = self._get_plant_bbox_hsv(frame)

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

        return display_frame, new_h, mask_first, mask

    def _save_diagnostic_histograms(self, r, g, b, exg, threshold_val):
        """Saves 4 separate PNG files, including the Otsu cut line on ExG."""
        os.makedirs('OUTPUT', exist_ok=True)

        # Data mapping: (Data, Title, Color, Filename, Bins, ShowThreshold)
        plot_configs = [
            (r[r > 0], "Red Channel Distribution", "red", "hist_frame45_red.png", 256, False),
            (g[g > 0], "Green Channel Distribution", "green", "hist_frame45_green.png", 256, False),
            (b[b > 0], "Blue Channel Distribution", "blue", "hist_frame45_blue.png", 256, False),
            (exg, "ExG Distribution", "purple", "hist_frame45_exg_threshold.png", 100, True),
            (exg, "ExG Distribution", "purple", "hist_frame45_exg.png", 100, False)
        ]

        for data, title, color, filename, bins, show_thresh in plot_configs:
            plt.figure(figsize=(10, 6))
            plt.hist(data.flatten(), bins=bins, color=color, alpha=0.6, edgecolor='black', linewidth=0.2, density=True)

            data_flat = data.flatten()
            try:
                # If there are too many pixels, we subsample to speed up calculation
                if len(data_flat) > 10000:
                    sample_data = np.random.choice(data_flat, 10000, replace=False)
                else:
                    sample_data = data_flat

                kde = gaussian_kde(sample_data)
                x_range = np.linspace(data_flat.min(), data_flat.max(), 500)
                plt.plot(x_range, kde(x_range), color='black', linewidth=2, label='Smooth Density Curve')
            except Exception as e:
                print(f"Could not calculate KDE for {title}: {e}")

            # Add the Otsu Threshold Line
            if show_thresh:
                plt.axvline(threshold_val, color='black', linestyle='--', linewidth=2,
                            label=f'Otsu Threshold: {threshold_val:.2f}')
                # Shade the areas to show what is "Plant" vs "Background"
                plt.axvspan(threshold_val, np.max(data), color='green', alpha=0.1)
                plt.text(threshold_val, plt.ylim()[1]*0.8, '  PLANT (Detected)', color='green', fontweight='bold')
                plt.text(threshold_val, plt.ylim()[1]*0.8, 'SOIL  ', color='brown',
                         fontweight='bold', horizontalalignment='right')

            plt.legend()
            plt.title(title)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(axis='y', alpha=0.3)

            save_path = os.path.join('OUTPUT', filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")

    # === TOP VIEW METHODS (GAI) ===
    def calculate_gai(self, top_frame, current_height, frame_idx=None):
        """Implements your GAI logic using ExG and Otsu."""
        if top_frame is None:
            return self.last_gai, self.last_cc, None

        # Black out the 4 corners [y1:y2, x1:x2]
        clean_top = top_frame.copy()
        m = Config.GAI_MARKER_MARGIN
        h, w = clean_top.shape[:2]

        # clean_top[0:m, 0:m] = 0  # Top-Left
        # clean_top[0:m, w - m:w] = 0  # Top-Right
        # clean_top[h - m:h, 0:m] = 0  # Bottom-Left
        # clean_top[h - m:h, w - m:w] = 0  # Bottom-Right

        hsv_top = cv2.cvtColor(clean_top, cv2.COLOR_BGR2HSV)
        avg_v = np.mean(hsv_top[:, :, 2])

        if avg_v < Config.MIN_BRIGHTNESS_GAI or current_height <= 0.01:
            # Create an empty black mask for visualization
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return self.last_gai, self.last_cc, empty_mask

        # 1. Excess Green Index (ExG)
        b, g, r = cv2.split(clean_top.astype(float))
        exg = 2 * g - r - b
        exg_min, exg_max = np.min(exg), np.max(exg)

        # 2. Otsu Thresholding
        exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        otsu_threshold, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        raw_threshold = (otsu_threshold / 255.0) * (exg_max - exg_min) + exg_min

        if frame_idx == 141:
            self._save_diagnostic_histograms(r, g, b, exg, raw_threshold)
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


class VideoManager:
    def __init__(self, side_size, top_size, fps=10):
        os.makedirs('OUTPUT/videos', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.writer_side = cv2.VideoWriter('OUTPUT/videos/height_growth.mp4', fourcc, fps, side_size)
        self.writer_top = cv2.VideoWriter('OUTPUT/videos/canopy_coverage.mp4', fourcc, fps, top_size)

        self.graph_size = (720, 1080)
        self.writer_graph = cv2.VideoWriter('OUTPUT/videos/growth_graph.mp4', fourcc, fps, self.graph_size)

    def write_frames(self, side_img, top_overlay, graph_img):
        self.writer_side.write(side_img)
        self.writer_top.write(top_overlay)
        self.writer_graph.write(graph_img)

    def release(self):
        self.writer_side.release()
        self.writer_top.release()
        self.writer_graph.release()


def create_plot_frame(data, current_idx, size_px):
    """Generates a BGR image of the graph (Height and CC only)."""
    w, h = size_px
    dpi = 100
    fig, ax1 = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)

    times = data['time'][:current_idx + 1]
    heights = data['height'][:current_idx + 1]
    ccs = data['cc'][:current_idx + 1]

    # Two Axes: Height (Left) and CC (Right)
    ax2 = ax1.twinx()

    p1, = ax1.plot(times, heights, 'g-', linewidth=2, label='Height (cm)')
    p2, = ax2.plot(times, ccs, 'r-', linewidth=2, label='Canopy Cover')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Height (cm)', color='g')
    ax2.set_ylabel('Canopy Cover (%)', color='r')

    ax1.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title("Plant Growth: Height & Canopy Coverage")
    lines = [p1, p2]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')

    fig.tight_layout()

    # Convert to OpenCV image
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    img = np.array(rgba_buffer)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    plt.close(fig)
    return cv2.resize(img, size_px)


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    analyzer = PlantAnalyzer()
    results = {'time': [], 'height': [], 'gai': [], 'cc': []}
    video_manager = None

    folders = sorted([f for f in os.listdir(Config.BASE_PATH) if os.path.isdir(os.path.join(Config.BASE_PATH, f))])

    for idx, folder in enumerate(folders):
        img_path = os.path.join(Config.BASE_PATH, folder)
        side_img_name = next((f for f in os.listdir(img_path) if "SIDE" in f and f.endswith(".jpg")), None)
        top_img_name = next((f for f in os.listdir(img_path) if "TOP" in f and f.endswith("homography.jpg")), None)

        if not side_img_name or not top_img_name: continue

        side_img = cv2.imread(os.path.join(img_path, side_img_name))
        top_img = cv2.imread(os.path.join(img_path, top_img_name))

        # 1. Process Data
        annotated_side, h_cm, mask_first, mask = analyzer.process_frame_side(side_img)
        gai_val, cc_val, top_mask = analyzer.calculate_gai(top_img, h_cm, frame_idx=idx)

        # 2. Create Top View Overlay (Contours on original image)
        top_overlay = top_img.copy()
        if top_mask is not None:
            contours, _ = cv2.findContours(top_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # Draw contours in Bright Green with thickness 2
            cv2.drawContours(top_overlay, contours, -1, (0, 255, 0), 2)
            # Add text label
            cv2.putText(top_overlay, f"CC: {cc_val * 100:.1f}%", (100, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # 3. Store Data
        time_str = f"{folder[0:4]}-{folder[4:6]}-{folder[6:8]} {folder[9:11]}:00"
        dt_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
        results['time'].append(dt_obj)
        results['height'].append(h_cm)
        results['gai'].append(gai_val)
        results['cc'].append(cc_val)

        # 4. Handle Video writing
        if video_manager is None:
            side_h, side_w = annotated_side.shape[:2]
            top_h, top_w = top_overlay.shape[:2]
            video_manager = VideoManager((side_w, side_h), (top_w, top_h), fps=7)

        graph_frame = create_plot_frame(results, len(results['time']) - 1, video_manager.graph_size)
        video_manager.write_frames(annotated_side, top_overlay, graph_frame)

        # Visual Feedback
        cv2.imshow("Live Analysis (Side)", annotated_side)
        cv2.imshow("Live Canopy (Contours)", top_overlay)
        key = cv2.waitKey(1) & 0xFF

        # Press 'q' to quit
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Generate a unique filename using a timestamp
            timestamp = int(time.time())

            # 1. Save the annotated height frame
            cv2.imwrite(f"height_frame_{timestamp}.png", top_overlay)

            # 2. Save the mask (Replace 'mask' with your actual mask variable name)
            # If your mask is from the side analysis, use that specific variable
            cv2.imwrite(f"height_mask_{timestamp}.png", top_mask)
            cv2.imwrite(f"height_initial_{timestamp}.png", top_img)

            print(f"Screenshots saved at {timestamp}")

    if video_manager:
        video_manager.release()
    cv2.destroyAllWindows()

    # --- FINAL SUMMARY GRAPH (Without GAI) ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    p1, = ax1.plot(results['time'], results['height'], 'g-', label='Height (cm)')
    p2, = ax2.plot(results['time'], results['cc'], 'r-', label='Canopy Cover')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Height (cm)', color='g')
    ax2.set_ylabel('Canopy Cover', color='r')
    ax1.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title("Final Growth Report")
    ax1.legend([p1, p2], [l.get_label() for l in [p1, p2]], loc='upper left')

    # Save final graph
    os.makedirs(os.path.dirname(Config.SAVE_GRAPH_NAME), exist_ok=True)
    plt.savefig(Config.SAVE_GRAPH_NAME, dpi=300)
