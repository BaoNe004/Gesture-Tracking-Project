import cv2
import mediapipe as mp
import time
import csv
import os

MODEL_PATH = "hand_landmarker.task"
CSV_PATH = "landmarks_dataset.csv"

BOX_SIZE = 560
COUNTDOWN_SEC = 3.0
SAVE_INTERVAL_SEC = 0.4

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

def build_header():
    header = ["label", "left_present", "right_present"]
    for side in ["left", "right"]:
        for i in range(21):
            header += [f"{side}_{i}_x", f"{side}_{i}_y", f"{side}_{i}_z"]
    return header

def ensure_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(build_header())

def hand_to_vec(hand_landmarks):
    vec = []
    for lm in hand_landmarks:
        vec.extend([lm.x, lm.y, lm.z])
    return vec

def extract_features(result):
    left_present = 0
    right_present = 0
    left_vec = [0.0] * 63
    right_vec = [0.0] * 63

    if result.hand_landmarks and result.handedness:
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name
            vec = hand_to_vec(hand_landmarks)

            if label == "Left":
                left_present = 1
                left_vec = vec
            elif label == "Right":
                right_present = 1
                right_vec = vec

    return [left_present, right_present] + left_vec + right_vec

def draw_landmarks(frame, result):
    h, w, _ = frame.shape
    if not result.hand_landmarks:
        return

    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]

    for hand_landmarks in result.hand_landmarks:
        pts = []
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            pts.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

def append_row(label, features):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label] + features)

def main():
    ensure_csv()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    mode = None
    armed_time = None
    last_save_time = 0.0
    total_saved = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape
            cx, cy = w // 2, h // 2
            half = BOX_SIZE // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)

            roi = frame[y1:y2, x1:x2]
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            features = extract_features(result)
            left_present = features[0]
            right_present = features[1]

            roi_draw = roi.copy()
            draw_landmarks(roi_draw, result)
            frame[y1:y2, x1:x2] = roi_draw

            status = "idle"
            color = (255, 255, 255)
            now = time.time()

            if mode is not None and armed_time is not None:
                remaining = COUNTDOWN_SEC - (now - armed_time)
                if remaining > 0:
                    status = f"{mode} starts in {remaining:.1f}s"
                    color = (0, 255, 255)
                else:
                    status = f"auto capturing: {mode}"
                    color = (0, 255, 0)

                    can_save = True
                    if mode == "shadow_clone":
                        can_save = (left_present == 1 and right_present == 1)

                    if can_save and now - last_save_time >= SAVE_INTERVAL_SEC:
                        append_row(mode, features)
                        total_saved += 1
                        last_save_time = now

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"saved: {total_saved}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"left_present: {left_present}  right_present: {right_present}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status, (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, "1=shadow_clone  2=other  0=stop  q=quit", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

            cv2.imshow("Capture Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("1"):
                mode = "shadow_clone"
                armed_time = time.time()
                last_save_time = 0.0
            elif key == ord("2"):
                mode = "other"
                armed_time = time.time()
                last_save_time = 0.0
            elif key == ord("0"):
                mode = None
                armed_time = None
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()