import cv2
import time
import os

INTERVAL_SEC, WARMUP_SEC = 30, 5
CAM_INDICES = [1,2]

# TODO: Make code more robust (can switch cam indices and still recognize camera) (QRCode?)
def take_pictures_multi_cam(interval=30, warmup_seconds=1.0):
    if not os.path.exists("calibrated_captures"):
        os.makedirs("calibrated_captures")

    try:
        while True:
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            for idx in CAM_INDICES:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    continue

                time.sleep(0.5) # For handshake hardware

                print(f"  [Cam {idx}] Warming up for {warmup_seconds}s...")

                # 2. THE CALIBRATION STEP
                start_warmup = time.time()
                while (time.time() - start_warmup) < warmup_seconds:
                    ret, _ = cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue

                # 3. TAKE THE FINAL PICTURE
                ret, frame = cap.read()
                if ret:
                    filename = f"calibrated_captures/cam{idx}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"  [Cam {idx}] Photo SAVED.")

                cap.release()

            print(f"Waiting {interval} seconds until next cycle...\n")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == "__main__":
    take_pictures_multi_cam(INTERVAL_SEC, WARMUP_SEC)