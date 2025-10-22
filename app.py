# app_combined.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque, Counter
import os
import time

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(layout="wide", page_title="Combined Driver Behavior + Wheel Detector")
st.title("🧠 + ✋ Combined: Face Behavior & Wheel On/Off Detector")

# Model file names (you confirmed these exist)
FACE_MODEL_FILE = "svm_hinge_model.pkl"
FACE_SCALER_FILE = "scaler.pkl"

HAND_MODEL_FILE = "svm_volang_model.pkl"
HAND_SCALER_FILE = "scaler_onoff.pkl"

# UI: mode toggles (both run simultaneously, but allow user to disable either)
with st.sidebar:
    st.header("Settings")
    run_face = st.checkbox("Enable Face Behavior Detection", value=True)
    run_hand = st.checkbox("Enable Wheel (Hand) Detection", value=True)
    fps_display = st.checkbox("Show FPS", value=True)
    enable_smoothing = st.checkbox("Enable smoothing (short-term)", value=True)
    st.markdown("---")
    st.markdown("Models (must be in app folder):")
    st.write(f"- Face model: `{FACE_MODEL_FILE}`")
    st.write(f"- Hand model: `{HAND_MODEL_FILE}`")

# ---------------------------
# UTIL: face feature functions (same as you used)
# ---------------------------
EPS = 1e-6
CLASSES_FACE = ["left", "right", "nod", "yawn", "blink", "normal"]  # must match model training

EYE_LEFT_IDX = np.array([33, 159, 145, 133, 153, 144])
EYE_RIGHT_IDX = np.array([362, 386, 374, 263, 380, 385])
MOUTH_IDX = np.array([61, 291, 0, 17, 78, 308])

def eye_aspect_ratio(landmarks, left=True):
    idx = EYE_LEFT_IDX if left else EYE_RIGHT_IDX
    pts = landmarks[idx, :2]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * (C + EPS))

def mouth_aspect_ratio(landmarks):
    pts = landmarks[MOUTH_IDX, :2]
    A = np.linalg.norm(pts[0] - pts[1])
    B = np.linalg.norm(pts[4] - pts[5])
    C = np.linalg.norm(pts[2] - pts[3])
    return (A + B) / (2.0 * (C + EPS))

def head_pose_yaw_pitch_roll(landmarks):
    left_eye, right_eye, nose, chin = landmarks[33][:2], landmarks[263][:2], landmarks[1][:2], landmarks[152][:2]
    dx, dy = right_eye - left_eye
    roll = np.degrees(np.arctan2(dy, dx + EPS))
    interocular = np.linalg.norm(right_eye - left_eye) + EPS
    eyes_center = (left_eye + right_eye) / 2.0
    yaw = np.degrees(np.arctan2((nose[0] - eyes_center[0]), interocular))
    baseline = chin - eyes_center
    pitch = np.degrees(np.arctan2((nose[1] - eyes_center[1]), (np.linalg.norm(baseline) + EPS)))
    return yaw, pitch, roll

# ---------------------------
# UTIL: hand feature functions (same as your webcam_wheel)
# ---------------------------
def landmarks_to_feature_vector_hand(landmarks, frame_w, frame_h):
    lm = np.array([[p.x * frame_w, p.y * frame_h] for p in landmarks])  # (21,2) px
    wrist = lm[0]
    rel = lm - wrist
    diag = np.sqrt(frame_w**2 + frame_h**2)
    rel_norm = rel.flatten() / (diag + EPS)
    return rel_norm  # shape (42,)

# ---------------------------
# LOAD MODELS (cached)
# ---------------------------
@st.cache_resource
def load_face_model():
    if not os.path.exists(FACE_MODEL_FILE) or not os.path.exists(FACE_SCALER_FILE):
        return None
    model_data = joblib.load(FACE_MODEL_FILE)
    scaler = joblib.load(FACE_SCALER_FILE)
    # model_data expected to be dict with W,b and maybe classes
    return model_data, scaler

@st.cache_resource
def load_hand_model():
    if not os.path.exists(HAND_MODEL_FILE) or not os.path.exists(HAND_SCALER_FILE):
        return None
    mdl = joblib.load(HAND_MODEL_FILE)
    scaler = joblib.load(HAND_SCALER_FILE)
    # detect type: dict/manual or sklearn estimator
    if isinstance(mdl, dict) and ("W" in mdl and "b" in mdl):
        return ("manual", mdl, scaler)
    else:
        return ("sklearn", mdl, scaler)

face_loaded = load_face_model()
hand_loaded = load_hand_model()

if run_face and face_loaded is None:
    st.warning("Face model or scaler not found in app folder. Face detection disabled.")
    run_face = False
if run_hand and hand_loaded is None:
    st.warning("Hand model or scaler not found in app folder. Hand detection disabled.")
    run_hand = False

# ---------------------------
# MEDIAPIPE in main process (we create instances in transformer to be safe)
# ---------------------------

# ---------------------------
# VIDEO PROCESSOR: combine both tasks
# ---------------------------
class CombinedProcessor(VideoTransformerBase):
    def __init__(self, run_face, run_hand, face_loaded, hand_loaded, show_fps):
        # face
        self.run_face = run_face
        self.run_hand = run_hand
        self.face_loaded = face_loaded
        self.hand_loaded = hand_loaded
        self.show_fps = show_fps

        # face model unpack
        if self.run_face and self.face_loaded is not None:
            face_model_data, face_scaler = self.face_loaded
            self.face_W = face_model_data["W"]
            self.face_b = face_model_data["b"].reshape(1, -1)
            self.face_scaler = face_scaler
            # optional classes stored in dict
            self.face_classes = face_model_data.get("classes", CLASSES_FACE)
        else:
            self.face_W = None

        # hand model unpack
        if self.run_hand and self.hand_loaded is not None:
            kind, mdl, hand_scaler = self.hand_loaded
            self.hand_kind = kind
            self.hand_model = mdl
            self.hand_scaler = hand_scaler
        else:
            self.hand_kind = None

        # mediapipe objects
        if self.run_face:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                             min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            self.face_mesh = None

        if self.run_hand:
            self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                                  min_detection_confidence=0.6, min_tracking_confidence=0.5)
            self.mp_draw = mp.solutions.drawing_utils
        else:
            self.hands = None

        # buffers for smoothing
        self.face_buffer = deque(maxlen=30)
        self.label_buffer_face = deque(maxlen=5)
        self.label_buffer_hand = deque(maxlen=5)

        # FPS
        self._last_time = time.time()
        self._fps = 0.0

        # wheel zone display (same as your earlier)
        self.WHEEL_CENTER = (0.5, 0.55)
        self.WHEEL_RADIUS = 0.25

    def manual_face_predict(self, feats):
        # feats: 2D array
        Xs = self.face_scaler.transform(feats)
        scores = Xs @ self.face_W + self.face_b
        cls_id = int(np.argmax(scores))
        return cls_id

    def manual_hand_predict(self, feat):
        # feat: 1D or 2D
        if self.hand_kind == "manual":
            mdl = self.hand_model
            W = mdl["W"]
            b = mdl["b"].reshape(1, -1)
            X = feat if feat.ndim == 2 else feat.reshape(1, -1)
            scores = X @ W + b
            return int(np.argmax(scores, axis=1)[0])
        else:
            # sklearn estimator
            X = feat.reshape(1, -1)
            if self.hand_scaler is not None:
                X = self.hand_scaler.transform(X)
            return int(self.hand_model.predict(X)[0])

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror
        h, w, _ = img.shape

        face_label = None
        hand_label = None

        # Face detection + features
        if self.run_face and self.face_mesh is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)
            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]
                landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in face.landmark])

                ear_l = eye_aspect_ratio(landmarks, True)
                ear_r = eye_aspect_ratio(landmarks, False)
                mar = mouth_aspect_ratio(landmarks)
                yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)
                nose, chin = landmarks[1], landmarks[152]
                angle_pitch_extra = np.degrees(np.arctan2(chin[1]-nose[1], chin[2]-nose[2]+EPS))
                forehead_y = np.mean(landmarks[[10,338,297,332,284],1])
                cheek_dist = np.linalg.norm(landmarks[50]-landmarks[280])

                # window features
                feat_vec = np.array([ear_l, ear_r, mar, yaw, pitch, roll, angle_pitch_extra, forehead_y, cheek_dist])
                self.face_buffer.append(feat_vec)

                if len(self.face_buffer) == self.face_buffer.maxlen:
                    window = np.array(self.face_buffer)
                    mean_feats = window.mean(axis=0)
                    std_feats = window.std(axis=0)
                    yaw_diff = np.mean(np.abs(np.diff(window[:,3])))
                    pitch_diff = np.mean(np.abs(np.diff(window[:,4])))
                    roll_diff = np.mean(np.abs(np.diff(window[:,5])))
                    mar_series, ear_series = window[:,2], (window[:,0]+window[:,1])/2.0
                    mar_max, mar_mean = np.max(mar_series), np.mean(mar_series)
                    mar_ear_ratio = mar_mean / (np.mean(ear_series)+EPS)
                    yaw_pitch_ratio = np.mean(np.abs(window[:,3])) / (np.mean(np.abs(window[:,4]))+EPS)

                    feats = np.concatenate([mean_feats, std_feats, [yaw_diff, pitch_diff, roll_diff, mar_max, mar_ear_ratio, yaw_pitch_ratio]]).reshape(1,-1)
                    try:
                        cls_id = self.manual_face_predict(feats)
                        label_face = CLASSES_FACE[cls_id]
                    except Exception:
                        label_face = "error"
                    # smoothing
                    self.label_buffer_face.append(label_face)
                    face_label = Counter(self.label_buffer_face).most_common(1)[0][0]
                else:
                    face_label = "analyzing..."
            else:
                face_label = "no face"
                self.face_buffer.clear()
        # Hand detection + predict
        if self.run_hand and self.hands is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res_h = self.hands.process(rgb)
            if res_h.multi_hand_landmarks:
                preds = []
                for hidx, hand_landmarks in enumerate(res_h.multi_hand_landmarks):
                    self.mp_draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    feat_hand = landmarks_to_feature_vector_hand(hand_landmarks.landmark, w, h)
                    try:
                        if self.hand_kind == "manual":
                            # if scaler exists apply
                            if self.hand_scaler is not None:
                                feat_scaled = self.hand_scaler.transform(feat_hand.reshape(1,-1))
                                pred = self.manual_hand_predict(feat_scaled)
                            else:
                                pred = self.manual_hand_predict(feat_hand)
                        else:
                            # sklearn
                            X_in = feat_hand.reshape(1,-1)
                            if self.hand_scaler is not None:
                                X_in = self.hand_scaler.transform(X_in)
                            pred = int(self.hand_model.predict(X_in)[0])
                        preds.append(pred)
                    except Exception:
                        # fallback heuristic: centroid distance to wheel center
                        lm = np.array([[p.x, p.y] for p in hand_landmarks.landmark])
                        centroid = lm.mean(axis=0)
                        wx, wy = self.WHEEL_CENTER
                        dist = np.sqrt((centroid[0]-wx)**2 + (centroid[1]-wy)**2)
                        preds.append(0 if dist <= self.WHEEL_RADIUS else 1)
                # combine: if any hand is on_wheel (0) => on_wheel
                final_hand = 0 if 0 in preds else 1
                hand_label = "on_wheel" if final_hand == 0 else "off_wheel"
                if enable_smoothing:
                    self.label_buffer_hand.append(hand_label)
                    hand_label = Counter(self.label_buffer_hand).most_common(1)[0][0]
            else:
                hand_label = "nohand"

        # Overlay results
        overlay_y = 40
        if face_label is not None:
            cv2.putText(img, f"Face: {str(face_label).upper()}", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0), 2)
            overlay_y += 40
        if hand_label is not None:
            cv2.putText(img, f"Hand: {str(hand_label).upper()}", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,255), 2)
            overlay_y += 40

        if self.show_fps:
            now = time.time()
            dt = now - self._last_time if self._last_time else 0.001
            self._fps = 0.9 * self._fps + 0.1 * (1.0/dt) if dt>0 else self._fps
            self._last_time = now
            cv2.putText(img, f"FPS: {self._fps:.1f}", (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        return img

# ---------------------------
# START webrtc streamer
# ---------------------------
st.info("Cho phép trình duyệt truy cập camera. Nhấn 'Start' trong widget dưới để chạy webcam.")

rtc_config = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turn:openrelay.metered.ca:443?transport=tcp"
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]
}

webrtc_streamer(
    key="combined",
    video_transformer_factory=lambda: CombinedProcessor(
        run_face, run_hand, face_loaded, hand_loaded, fps_display
    ),
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=rtc_config,
    async_processing=True,
)

