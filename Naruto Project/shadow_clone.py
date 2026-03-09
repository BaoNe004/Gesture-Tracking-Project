import cv2
import mediapipe as mp
import time
import joblib
import numpy as np
from collections import deque

MODEL_PATH = "landmark_rf.joblib"
HAND_MODEL_PATH = "hand_landmarker.task"
SEGMENTER_MODEL_PATH = "selfie_segmenter.tflite"
SMOKE_PATH = "smoke_effect.png"
POSE_ICON_PATH = "clone_effect.png"


# Hand detection box for the jutsu classifier
BOX_SIZE = 1080

# Trigger tuning
SMOOTHING_FRAMES = 9
CONF_THRESHOLD = 0.8
HOLD_SECONDS = 0.3
ACTIVATION_SECONDS = 5.0
COOLDOWN_SECONDS = 3.5

SMOKE_DURATION = 0.3
SMOKE_SCALE = 1
SMOKE_Y_OFFSET = 20

POSE_ICON_SCALE = 0.15
POSE_ICON_DARK = 0.2
POSE_ICON_BRIGHT = 1.0
POSE_ICON_BOTTOM_MARGIN = 20
#FLASH_SECONDS = 0.20

# Clone behavior
# dx, dy are in pixels here, easier to tune visually
CLONE_SPECS = [
    {"dx": -260, "dy":  100, "scale": 0.70, "alpha": 1.00, "delay": 0.00},
    {"dx":  260, "dy":  100, "scale": 0.70, "alpha": 1.00, "delay": 0.08},
    {"dx": -180, "dy":  80, "scale": 0.70, "alpha": 1.00, "delay": 0.16},
    {"dx":  180, "dy":  80, "scale": 0.70, "alpha": 1.00, "delay": 0.24},
    {"dx":  90, "dy":  50, "scale": 0.80, "alpha": 1.00, "delay": 0.32},
    {"dx": -90, "dy":  50, "scale": 0.80, "alpha": 1.00, "delay": 0.4},

]
CLONE_FADE = 0.18

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
RunningMode = mp.tasks.vision.RunningMode


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

    return np.array(
        [left_present, right_present] + left_vec + right_vec,
        dtype=np.float32
    ).reshape(1, -1)


def draw_landmarks(frame, result):
    if not result.hand_landmarks:
        return

    h, w, _ = frame.shape

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
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


def get_person_alpha_mask(segmenter, frame_bgr, timestamp_ms):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = segmenter.segment_for_video(mp_image, timestamp_ms)
    category_mask = result.category_mask.numpy_view()

    # Usually 0 = person, 1 = background
    mask = (category_mask == 0).astype(np.float32)

    # soften edges
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

    # clean small noise
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = (mask * 255).astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    return mask_u8.astype(np.float32) / 255.0


def blend_person(base_bgr, person_bgr, alpha_mask, extra_alpha=1.0):
    base = base_bgr.astype(np.float32)
    person = person_bgr.astype(np.float32)

    alpha = np.expand_dims(alpha_mask * extra_alpha, axis=2)
    out = person * alpha + base * (1.0 - alpha)

    return np.clip(out, 0, 255).astype(np.uint8)


def put_live_clone(base, person_bgr, alpha_mask, dx, dy, scale=1.0, extra_alpha=1.0):
    h, w = base.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, 0, scale)
    M[0, 2] += dx
    M[1, 2] += dy

    warped_person = cv2.warpAffine(
        person_bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    warped_alpha = cv2.warpAffine(
        alpha_mask,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return blend_person(base, warped_person, warped_alpha, extra_alpha)


def render_live_clones(base_frame, person_bgr, alpha_mask, activation_progress, smoke_img):
    out = base_frame.copy()
    h, w = out.shape[:2]


    for spec in CLONE_SPECS:
        local_t = (activation_progress - spec["delay"]) / CLONE_FADE
        local_t = max(0.0, min(1.0, local_t))

        smoke_t = (activation_progress - spec["delay"]) / SMOKE_DURATION
        smoke_t =max(0.0, min(1.0, smoke_t))
        clone_center_x = w // 2 + spec["dx"]
        clone_center_y = h // 2 + spec["dy"]

        if 0.0 < smoke_t < 1.0:
            out = draw_smoke_burst(
                out,
                smoke_img,
                clone_center_x,
                clone_center_y,
                spec["scale"],
                smoke_t
            )
            

        if local_t <= 0.0:
            continue

        current_alpha = spec["alpha"] * local_t
        out = put_live_clone(
            out,
            person_bgr,
            alpha_mask,
            dx=spec["dx"],
            dy=spec["dy"],
            scale=spec["scale"],
            extra_alpha=current_alpha
        )

    # important: redraw the real person on top
    out = blend_person(out, person_bgr, alpha_mask, extra_alpha=1.0)
    return out


# def add_flash(frame, strength):
#     strength = max(0.0, min(1.0, strength))
#     white = np.full_like(frame, 255)
#     return cv2.addWeighted(white, strength, frame, 1.0 - strength, 0)

def smoke_effect(base_bgr, overlay_rgba, x, y, alpha_mult = 1.0):
    result = base_bgr.copy()

    if overlay_rgba is None or overlay_rgba.shape[2] != 4:
        return result
    
    h, w = overlay_rgba.shape[:2]
    H, W = result.shape[:2] 

    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return result   

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2-x1)
    overlay_y2 = overlay_y1 + (y2-y1)

    overlay_crop = overlay_rgba[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    overlay_rgb = overlay_crop[:, :, :3].astype(np.float32)
    overlay_alpha = (overlay_crop[:, :, 3].astype(np.float32) / 255.0) * alpha_mult
    overlay_alpha = np.expand_dims(overlay_alpha, axis= 2)

    roi = result[y1:y2, x1:x2].astype(np.float32)
    blended = overlay_rgb * overlay_alpha + roi * (1.0 - overlay_alpha)

    result[y1:y2, x1:x2] = np.clip(blended, 0 , 255).astype(np.uint8) 
    return result

def draw_smoke_burst(base_bgr, smoke_rgba, center_x, center_y, clone_scale, smoke_t):
    result = base_bgr.copy()

    if smoke_t < 0.0 or smoke_t > 1.0:
        return result
    
    current_scale = clone_scale * SMOKE_SCALE * (0.7 + 0.6 * smoke_t)
    current_alpha = 1.0 - smoke_t
    current_y = center_y + SMOKE_Y_OFFSET - int(30 * smoke_t)

    smoke_h, smoke_w = smoke_rgba.shape[:2]
    new_w = max(1, int(smoke_w * current_scale))
    new_h = max(1, int(smoke_h * current_scale))

    smoke_resized = cv2.resize(smoke_rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    x = int(center_x - new_w / 2)
    y = int(current_y - new_h / 2)

    result = smoke_effect(result, smoke_resized, x, y, alpha_mult = current_alpha)

    return result

def draw_pose_icon(base_bgr, icon_bgr, brightness, scale, bottom_margin):
    result = base_bgr.copy()

    if icon_bgr is None:
        print("icon_bgr is None")
        return result
    
    
    H, W = result.shape[:2]
    h, w = icon_bgr.shape[:2]

    new_w = max(1, int(W * scale))
    new_h = max(1, int(h * (new_w / w)))

    icon_resized = cv2.resize(icon_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    icon_adjust = np.clip(icon_resized.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

    x = (W - new_w) // 2
    y = H - new_h - bottom_margin

  

    if x < 0 or y < 0 or x + new_w > W or y + new_h > H:  
        return result
    
    result[y:y + new_h, x:x + new_w] = icon_adjust
    return result

def main():
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    class_names = list(model.classes_)
    smoke_img = cv2.imread(SMOKE_PATH, cv2.IMREAD_UNCHANGED)
    icon_img = cv2.imread(POSE_ICON_PATH)
    icon_img = cv2.flip(icon_img, 1)

    if icon_img is None:
        print("Error: icon png not found")
        return

    if smoke_img is None:
        print("Error: smoke png not found")
        return


    if "shadow_clone" not in class_names:
        print("Error: 'shadow_clone' class not found in model.")
        return

    shadow_idx = class_names.index("shadow_clone")

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    segmenter_options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=SEGMENTER_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        output_category_mask=True,
        output_confidence_masks=False
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    history = deque(maxlen=SMOOTHING_FRAMES)
    stable_start = None
    activated_until = 0.0
    cooldown_until = 0.0
    activation_started_at = 0.0
    flash_until = 0.0

    with HandLandmarker.create_from_options(hand_options) as landmarker, \
         ImageSegmenter.create_from_options(segmenter_options) as segmenter:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            raw_frame = frame.copy()

            h, w, _ = raw_frame.shape
            cx, cy = w // 2, h // 2
            half = BOX_SIZE // 2

            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)

            roi = raw_frame[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            hand_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)

            timestamp_ms = int(time.time() * 1000)

            hand_result = landmarker.detect_for_video(hand_image, timestamp_ms)
            features = extract_features(hand_result)

            left_present = int(features[0][0])
            right_present = int(features[0][1])
            both_hands_ok = (left_present == 1 and right_present == 1)

            probs = model.predict_proba(features)[0]
            shadow_conf = float(probs[shadow_idx])

            history.append(shadow_conf)
            avg_shadow_conf = sum(history) / len(history)

            now = time.time()

            raw_pose_ok = both_hands_ok and avg_shadow_conf > CONF_THRESHOLD
            pose_ok = raw_pose_ok

            if now < cooldown_until:
                pose_ok = False

            if pose_ok:
                if stable_start is None:
                    stable_start = now

                held = now - stable_start

                if held >= HOLD_SECONDS:
                    activation_started_at = now
                    activated_until = now + ACTIVATION_SECONDS
                    cooldown_until = now + COOLDOWN_SECONDS
                    #flash_until = now + FLASH_SECONDS
                    stable_start = None
                    history.clear()
            else:
                stable_start = None

            alpha_mask = get_person_alpha_mask(segmenter, raw_frame, timestamp_ms)

            if now < activated_until:
                person_bgr = (raw_frame.astype(np.float32) * alpha_mask[..., None]).astype(np.uint8)

                activation_progress = ((now - activation_started_at) / ACTIVATION_SECONDS) * (ACTIVATION_SECONDS/ 2.0)
                activation_progress = max(0.0, min(1.0, activation_progress))

                frame = render_live_clones(raw_frame, person_bgr, alpha_mask, activation_progress, smoke_img)
            else:
                frame = raw_frame.copy()

            roi_draw = frame[y1:y2, x1:x2]
            draw_landmarks(roi_draw, hand_result)

            # if now < flash_until:
            #     frame = add_flash(frame, 0.28)
        
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
          
            icon_brightness = POSE_ICON_BRIGHT if raw_pose_ok else POSE_ICON_DARK
            frame = draw_pose_icon(
                frame,
                icon_img,
                brightness = icon_brightness,
                scale = POSE_ICON_SCALE,
                bottom_margin = POSE_ICON_BOTTOM_MARGIN,
            )

            if pose_ok and stable_start is not None:
                held = now - stable_start

            cv2.imshow("Naruto Jutsu", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()