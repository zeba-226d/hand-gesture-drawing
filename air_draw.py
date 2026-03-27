# Air Drawing with Hand Gestures
# Built on workshop starter code (GDG USyd - Hand Gesture Volume Controller).

import cv2
from math import hypot
import numpy as np
from collections import deque
import os
import time

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

MODEL_DIR = os.path.join(os.path.expanduser("~"), ".mediapipe_models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

def download_tracking_model():
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        os.makedirs(MODEL_DIR, exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        print("Downloading model...")
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Done.")

download_tracking_model()

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

COLORS = [
    ("Red",(0,0,255)),
    ("Green",(0,200,0)),
    ("Blue",(255,100,0)),
    ("Yellow",(0,255,255)),
    ("Magenta",(255,0,200)),
    ("Cyan",(255,255,0)),
    ("White",(255,255,255)),
    ("Orange",(0,140,255)),
]

PALETTE_RADIUS = 20
PALETTE_Y = 40
PINCH_THRESHOLD = 35
BRUSH_SIZE_DEFAULT = 4

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback,
)
landmarker = HandLandmarker.create_from_options(options)

def get_pixel_coords(hand_landmarks, frame_shape):
    h, w = frame_shape[:2]
    return [(i, int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(hand_landmarks)]

def distance_between(pts, id_a, id_b):
    ax, ay = pts[id_a][1], pts[id_a][2]
    bx, by = pts[id_b][1], pts[id_b][2]
    return hypot(bx - ax, by - ay)

def is_finger_extended(pts, tip_id, pip_id):
    return pts[tip_id][2] < pts[pip_id][2]

def is_thumb_extended(pts):
    return abs(pts[4][1] - pts[2][1]) > abs(pts[3][1] - pts[2][1])

def all_fingers_open(pts):
    return (
        is_thumb_extended(pts)
        and is_finger_extended(pts, 8, 6)
        and is_finger_extended(pts, 12, 10)
        and is_finger_extended(pts, 16, 14)
        and is_finger_extended(pts, 20, 18)
    )

def draw_hand_skeleton(frame, hand_landmarks_list):
    h, w = frame.shape[:2]
    for hand_lms in hand_landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 0, 0), -1)

def draw_palette(frame, selected_index, frame_width):
    spacing = frame_width // (len(COLORS) + 1)
    positions = []
    for i, (name, color) in enumerate(COLORS):
        cx = spacing * (i + 1)
        cy = PALETTE_Y
        if i == selected_index:
            cv2.circle(frame, (cx, cy), PALETTE_RADIUS + 6, (255, 255, 255), 3)
        cv2.circle(frame, (cx, cy), PALETTE_RADIUS, color, -1)
        cv2.circle(frame, (cx, cy), PALETTE_RADIUS, (200, 200, 200), 1)
        positions.append((cx, cy))
    return positions

def check_palette_hover(index_tip, palette_positions):
    ix, iy = index_tip
    for i, (cx, cy) in enumerate(palette_positions):
        if hypot(ix - cx, iy - cy) < PALETTE_RADIUS + 10:
            return i
    return None

print("Opening camera...")
cap = cv2.VideoCapture(0)
time.sleep(2)
timestamp_ms = 0

canvas = None
current_color_idx = 0
brush_size = BRUSH_SIZE_DEFAULT
prev_draw_pt = None
drawing_active = False
clear_cooldown = 0
smooth_queue = deque(maxlen=3)

print("Air Drawing - Ready!")
print("Pinch to draw | Point at colors to switch | Open palm to clear")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms += 1
    landmarker.detect_async(mp_image, timestamp_ms)

    if latest_result and latest_result.hand_landmarks:
        draw_hand_skeleton(frame, latest_result.hand_landmarks)

    if latest_result and latest_result.hand_landmarks:
        pts = get_pixel_coords(latest_result.hand_landmarks[0], frame.shape)
        index_tip = (pts[8][1], pts[8][2])
        pinch_dist = distance_between(pts, 4, 8)

        if clear_cooldown > 0:
            clear_cooldown -= 1
        elif all_fingers_open(pts):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            prev_draw_pt = None
            clear_cooldown = 20
            cv2.putText(frame, "Canvas Cleared!", (w // 2 - 100, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            palette_positions = draw_palette(frame, current_color_idx, w)
            if pinch_dist > PINCH_THRESHOLD:
                hovered = check_palette_hover(index_tip, palette_positions)
                if hovered is not None:
                    current_color_idx = hovered

            if pinch_dist < PINCH_THRESHOLD:
                smooth_queue.append(index_tip)
                sx = int(np.mean([p[0] for p in smooth_queue]))
                sy = int(np.mean([p[1] for p in smooth_queue]))
                draw_pt = (sx, sy)

                if prev_draw_pt is not None:
                    cv2.line(canvas, prev_draw_pt, draw_pt,
                             COLORS[current_color_idx][1], brush_size)
                prev_draw_pt = draw_pt
                drawing_active = True

                cv2.circle(frame, draw_pt, brush_size + 2,
                           COLORS[current_color_idx][1], -1)
            else:
                prev_draw_pt = None
                drawing_active = False
                smooth_queue.clear()
    else:
        prev_draw_pt = None
        drawing_active = False
        smooth_queue.clear()

    palette_positions = draw_palette(frame, current_color_idx, w)

    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    frame = cv2.add(frame_bg, canvas_fg)

    color_name = COLORS[current_color_idx][0]
    status = "DRAWING" if drawing_active else "Move hand"
    info = "Color: " + color_name + " | Brush: " + str(brush_size) + "px | " + status
    cv2.putText(frame, info, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Pinch=Draw | Open palm=Clear | q=Quit",
                (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    cv2.imshow("Air Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        prev_draw_pt = None
    elif key == ord("+") or key == ord("="):
        brush_size = min(20, brush_size + 1)
    elif key == ord("-"):
        brush_size = max(1, brush_size - 1)
    elif key == ord("s"):
        cv2.imwrite("air_drawing.png", canvas)
        print("Drawing saved to air_drawing.png")

landmarker.close()
cap.release()
cv2.destroyAllWindows()
