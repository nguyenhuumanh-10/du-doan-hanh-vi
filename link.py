import streamlit as st
import joblib
import numpy as np
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from collections import deque, Counter

# ============================
# CẤU HÌNH & HÀM TIỆN ÍCH
# ============================
WINDOW_SIZE = 30
SMOOTH_WINDOW = 5
CLASSES = ["left", "right", "nod", "yawn", "blink", "normal"]
EPS = 1e-6
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

# ============================
# GIAO DIỆN STREAMLIT
# ============================
st.set_page_config(layout="wide")
st.title("👁️ Hệ Thống Dự Đoán Hành Vi Lái Xe")

try:
    # ============================
    # 1. Tải mô hình và scaler
    # ============================
    @st.cache_resource
    def load_model_and_scaler():
        model_data = joblib.load("svm_hinge_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model_data, scaler

    model_data, scaler = load_model_and_scaler()
    W, b = model_data["W"], model_data["b"]

    # ============================
    # 2. Thiết lập MediaPipe
    # ============================
    @st.cache_resource
    def load_face_mesh():
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    face_mesh = load_face_mesh()

    # ============================
    # 3. Lớp xử lý video (TÍCH HỢP LOGIC NÂNG CAO)
    # ============================
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.buffer = deque(maxlen=WINDOW_SIZE)
            self.label_buffer = deque(maxlen=SMOOTH_WINDOW)
            self.last_label = "normal"
            self.status = "Đang tìm khuôn mặt..."

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            h, w, _ = img.shape
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                self.status = "Đang phân tích..."
                face = results.multi_face_landmarks[0]
                landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in face.landmark])

                # Tính các đặc trưng cơ bản
                ear_l = eye_aspect_ratio(landmarks, True)
                ear_r = eye_aspect_ratio(landmarks, False)
                mar = mouth_aspect_ratio(landmarks)
                yaw, pitch, roll = head_pose_yaw_pitch_roll(landmarks)
                nose, chin = landmarks[1], landmarks[152]
                angle_pitch_extra = np.degrees(np.arctan2(chin[1]-nose[1], chin[2]-nose[2]+EPS))
                forehead_y = np.mean(landmarks[[10,338,297,332,284],1])
                cheek_dist = np.linalg.norm(landmarks[50]-landmarks[280])
                
                # Thêm vào buffer
                self.buffer.append([ear_l, ear_r, mar, yaw, pitch, roll, angle_pitch_extra, forehead_y, cheek_dist])

                if len(self.buffer) == WINDOW_SIZE:
                    self.status = "Đang dự đoán..."
                    # Tính các đặc trưng thống kê trên cửa sổ
                    window = np.array(self.buffer)
                    mean_feats, std_feats = window.mean(axis=0), window.std(axis=0)
                    yaw_diff, pitch_diff, roll_diff = np.mean(np.abs(np.diff(window[:,3]))), np.mean(np.abs(np.diff(window[:,4]))), np.mean(np.abs(np.diff(window[:,5])))
                    mar_series, ear_series = window[:,2], (window[:,0]+window[:,1])/2.0
                    mar_max, mar_mean = np.max(mar_series), np.mean(mar_series)
                    mar_ear_ratio = mar_mean / (np.mean(ear_series)+EPS)
                    yaw_pitch_ratio = np.mean(np.abs(window[:,3])) / (np.mean(np.abs(window[:,4]))+EPS)
                    
                    # Ghép nối tất cả các đặc trưng lại
                    feats = np.concatenate([mean_feats, std_feats, [yaw_diff, pitch_diff, roll_diff, mar_max, mar_ear_ratio, yaw_pitch_ratio]]).reshape(1,-1)
                    
                    # Chuẩn hóa và dự đoán
                    X_input = scaler.transform(feats)
                    scores = X_input @ W + b
                    cls_id = int(np.argmax(scores))
                    label = CLASSES[cls_id]

                    # Làm mượt kết quả
                    self.label_buffer.append(label)
                    smooth_label = Counter(self.label_buffer).most_common(1)[0][0]
                    self.last_label = smooth_label
            else:
                self.status = "Không tìm thấy khuôn mặt"
                self.buffer.clear() # Xóa buffer nếu mất mặt
            
            # Hiển thị kết quả
            cv2.putText(img, self.last_label.upper(), (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            cv2.putText(img, f"Status: {self.status}", (30, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            
            return img

    # ============================
    # 4. Khởi chạy webcam (SỬ DỤNG TURN SERVER ĐỂ VƯỢT TƯỜNG LỬA)
    # ============================
    st.info("💡Cho phép trình duyệt truy cập camera và nhìn thẳng vào webcam.")

    # Cấu hình này bao gồm cả STUN và một máy chủ TURN công cộng (openrelay)
    # để cố gắng vượt qua các tường lửa nghiêm ngặt.
    webrtc_streamer(
        key="webcam",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {
                    "urls": [
                        "turn:openrelay.metered.ca:80",
                        "turn:openrelay.metered.ca:443",
                    ],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
            ]
        }
    )

# ============================
# 5. KHỐI BẮT LỖI
# ============================
except Exception as e:
    st.error("LỖI NGHIÊM TRỌNG ĐÃ XẢY RA")
    st.error("Vui lòng kiểm tra lại các file mô hình và đảm bảo các thư viện đã được cài đặt đầy đủ.")
    st.exception(e)
