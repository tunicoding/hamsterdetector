from pathlib import Path

import cv2
import mediapipe as mp

ASSET_DIR = Path(__file__).resolve().parent

hamster_surprise = cv2.imread(str(ASSET_DIR / "hamster1.jpg"))
hamster_peace = cv2.imread(str(ASSET_DIR / "hamster2.jpg"))
hamster_okay = cv2.imread(str(ASSET_DIR / "hamster3.jpg"))
hamster_idle = cv2.imread(str(ASSET_DIR / "hamster4.jpg"))

current = hamster_idle

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(refine_landmarks=True)
hands = mp_hands.Hands(max_num_hands=1)

green = mp_draw.DrawingSpec(color=(143, 237, 140), thickness=2, circle_radius=3)

cap = cv2.VideoCapture(0)

def mouth_open_ratio(landmarks):
    top = landmarks[13]
    bottom = landmarks[14]
    return abs(top.y - bottom.y)

def detect_peace(hand_landmarks):
    idx_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    mid_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    return idx_up and mid_up and ring_down and pinky_down

def detect_okay(hand_landmarks):
    return hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)

    face_detected = face_result.multi_face_landmarks is not None
    hand_detected = hand_result.multi_hand_landmarks is not None

    if hand_detected:
        hand = hand_result.multi_hand_landmarks[0]

        if detect_peace(hand):
            current = hamster_peace

        elif detect_okay(hand):
            current = hamster_okay

        elif face_detected:
            lm = face_result.multi_face_landmarks[0].landmark
            if mouth_open_ratio(lm) > 0.03:
                current = hamster_surprise
            else:
                current = hamster_surprise

        else:
            current = hamster_surprise

    elif face_detected:
        lm = face_result.multi_face_landmarks[0].landmark
        if mouth_open_ratio(lm) > 0.03:
            current = hamster_surprise
        else:
            current = hamster_idle

    else:
        current = hamster_idle

    if hand_detected:
        for hand in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=green,
                connection_drawing_spec=green
            )

    hamster_resized = cv2.resize(current, (h, h))
    combined = cv2.hconcat([hamster_resized, frame])

    cv2.imshow("Hamster Moves With You", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
