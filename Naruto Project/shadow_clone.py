import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (13,17),
    
]

def draw_hand(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = []

    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

with HandLandmarker.create_from_options(options) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        result = hands.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                draw_hand(frame, hand_landmarks)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()