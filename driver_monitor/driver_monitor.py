# """
# ╔══════════════════════════════════════════════════════════════╗
# ║     AI Driver Monitoring System  v6.2                        ║
# ║     Telegram alerts ONLY for smoking • Voice for all events  ║
# ║     Voice: Microsoft Edge Neural TTS (Neerja) - FIXED        ║
# ╚══════════════════════════════════════════════════════════════╝
#
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   pip install opencv-python mediapipe numpy pyserial ultralytics edge-tts pygame
# """

import cv2
import numpy as np
import mediapipe as mp
import serial
import csv
import time
import os
import threading
import queue
import tempfile
from datetime import datetime
from ultralytics import YOLO
import torch
import requests
import asyncio
import pygame

# ---------- EDGE-TTS FIX: import once globally ----------
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("[WARN] edge-tts not installed. Voice disabled. Run: pip install edge-tts")

# ══════════════════════════════════════════════════════════════════
#  CUDA SETUP
# ══════════════════════════════════════════════════════════════════
_CUDA   = torch.cuda.is_available()
_DEVICE = "cuda:0" if _CUDA else "cpu"
_HALF   = _CUDA

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION – adjust to your setup
# ══════════════════════════════════════════════════════════════════
SERIAL_PORT   = "COM3"
BAUD_RATE     = 115200
PHONE_MODEL   = "models/yolov8n.pt"
SMOKE_MODEL   = r"smoking_detection_trained2.pt"
LOG_FILE      = "logs/events.csv"

EAR_THRESH    = 0.25
EAR_FRAMES    = 48
DISTRACT_SEC  = 3.0
BREAK_AFTER   = 3
PHONE_CONF    = 0.45
SMOKE_CONF    = 0.40
LOG_GAP       = 3.0
SMOKE_ANGLES  = [0, 30, -30]

# Telegram (fill in your own credentials)
TELEGRAM_TOKEN   = "8524244191:AAGvGJkun_14ey12NOZFNZzA_anyo10uVHs"      # <-- REPLACE
TELEGRAM_CHAT_ID = "1881450187"        # <-- REPLACE
TELEGRAM_COOLDOWN = 15

# Voice settings
VOICE_REPEAT = {"DROWSY": 5, "DISTRACTED": 5, "PHONE": 5, "SMOKING": 5}
VOICE_VOICE  = "en-IN-NeerjaNeural"
VOICE_MSG = {
    "DROWSY"     : "Warning. Drowsiness detected. Please stay alert or pull over.",
    "DISTRACTED" : "Warning. Eyes off road. Please focus on driving.",
    "PHONE"      : "Please put down your phone. Do not use phone while driving.",
    "SMOKING"    : "Warning. Do not smoke while driving. It is very dangerous.",
}

# Colours (BGR)
COL = {
    "ALERT"      : (50,  220,  80),
    "DROWSY"     : (45,   45, 240),
    "DISTRACTED" : (20,  150, 255),
    "PHONE"      : (210,  30, 200),
    "SMOKING"    : (20,   40, 210),
    "bg"         : (8,    8,  16),
    "panel"      : (14,  14,  26),
    "line"       : (40,  40,  60),
    "dim"        : (85,  85, 108),
    "accent"     : (0,  210, 255),
    "good"       : (50,  220,  80),
}
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33,  160, 158, 133, 153, 144]

# ══════════════════════════════════════════════════════════════════
#  TELEGRAM (ONLY smoking)
# ══════════════════════════════════════════════════════════════════
_last_telegram_time = 0

def send_telegram_alert(frame=None):
    global _last_telegram_time
    now = time.time()
    if now - _last_telegram_time < TELEGRAM_COOLDOWN:
        return False
    if not TELEGRAM_TOKEN or "YOUR_BOT_TOKEN" in TELEGRAM_TOKEN:
        return False
    message = f"🚨 Smoking detected inside the car!\nTime: {datetime.now().strftime('%H:%M:%S')}"
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
        if frame is not None:
            _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            files = {"photo": ("smoking_alert.jpg", img_encoded.tobytes())}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                          files=files, data={"chat_id": TELEGRAM_CHAT_ID}, timeout=5)
        _last_telegram_time = now
        print("[Telegram] Smoking alert sent")
        return True
    except Exception as e:
        print(f"[Telegram] Error: {e}")
        return False

# ══════════════════════════════════════════════════════════════════
#  VOICE CLASS (FIXED – edge-tts works in background thread)
# ══════════════════════════════════════════════════════════════════
class Voice:
    def __init__(self):
        self.ok = EDGE_TTS_AVAILABLE
        self._q = queue.Queue(maxsize=1)
        self._cache = {}
        self._last_spoken = {}
        self._alive = True
        if self.ok:
            try:
                pygame.mixer.init()
                t = threading.Thread(target=self._loop, daemon=True)
                t.start()
                print("[Voice] Ready – Neerja Indian English (edge-tts)")
            except Exception as e:
                print(f"[Voice] Failed to init pygame: {e}")
                self.ok = False
        else:
            print("[Voice] Disabled (edge-tts not installed)")

    def _loop(self):
        while self._alive:
            try:
                text = self._q.get(timeout=0.4)
                path = self._get_audio(text)
                if path:
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[Voice] Playback error: {e}")

    def _get_audio(self, text):
        if text in self._cache:
            return self._cache[text]
        if not self.ok:
            return None
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp.close()
            # edge-tts requires a proper asyncio event loop in this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                edge_tts.Communicate(text, voice=VOICE_VOICE).save(tmp.name)
            )
            loop.close()
            self._cache[text] = tmp.name
            return tmp.name
        except Exception as e:
            print(f"[Voice] TTS failed: {e}")
            return None

    def _push(self, msg):
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(msg)
        except queue.Full:
            pass

    def tick(self, state):
        if not self.ok or state == "ALERT" or state not in VOICE_MSG:
            return
        now = time.time()
        gap = VOICE_REPEAT.get(state, 5)
        last = self._last_spoken.get(state)
        if last is None or (now - last) >= gap:
            self._last_spoken[state] = now
            self._push(VOICE_MSG[state])

    def end_state(self, state):
        self._last_spoken.pop(state, None)

    def stop(self):
        self._alive = False
        for path in self._cache.values():
            try:
                os.unlink(path)
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════════
#  SMOKING DETECTION (multi-angle)
# ══════════════════════════════════════════════════════════════════
def _rotate_frame(frame, angle):
    if angle == 0:
        return frame
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h))

def _unrotate_box(x1, y1, x2, y2, angle, W, H):
    if angle == 0:
        return x1, y1, x2, y2
    cx, cy = W/2, H/2
    rad = -np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    def rot(px, py):
        px, py = px - cx, py - cy
        return px*cos_a - py*sin_a + cx, px*sin_a + py*cos_a + cy
    pts = [rot(px, py) for px, py in [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

def detect_smoking(model, frame, conf_thresh=SMOKE_CONF, angles=SMOKE_ANGLES):
    H, W = frame.shape[:2]
    best = {}
    for angle in angles:
        rotated = _rotate_frame(frame, angle)
        try:
            results = model(rotated, verbose=False, device=_DEVICE, half=_HALF)[0].boxes
        except Exception:
            continue
        for box in results:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = _unrotate_box(rx1, ry1, rx2, ry2, angle, W, H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            key = (x1//20, y1//20)
            if key not in best or conf > best[key][4]:
                best[key] = (x1, y1, x2, y2, conf)
    return list(best.values())

# ══════════════════════════════════════════════════════════════════
#  EAR + HEAD POSE
# ══════════════════════════════════════════════════════════════════
def ear(landmarks, indices, W, H):
    pts = [np.array([landmarks[i].x*W, landmarks[i].y*H]) for i in indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A+B)/(2.0*C) if C else 0.0

def is_head_turned(landmarks, W, H):
    nose_x = landmarks[1].x * W
    left_x  = landmarks[234].x * W
    right_x = landmarks[454].x * W
    face_width = abs(right_x - left_x)
    if face_width == 0:
        return False
    center = (left_x + right_x)/2.0
    return abs(nose_x - center) / face_width > 0.18

# ══════════════════════════════════════════════════════════════════
#  SERIAL (ESP32)
# ══════════════════════════════════════════════════════════════════
def esp_connect(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print(f"[ESP32] Connected on {port}")
        return ser
    except Exception as e:
        print(f"[ESP32] Could not connect: {e}")
        return None

def esp_send(ser, cmd):
    if ser and ser.is_open:
        try:
            ser.write((cmd + "\n").encode())
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════
#  LOGGING (CSV)
# ══════════════════════════════════════════════════════════════════
def log_open(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.isfile(path)
    fh = open(path, "a", newline="")
    wr = csv.writer(fh)
    if new_file:
        wr.writerow(["timestamp", "event", "EAR"])
    return fh, wr

def log_write(writer, event, ear_value):
    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event, f"{ear_value:.3f}"])

# ══════════════════════════════════════════════════════════════════
#  HUD DRAWING (short version – same as before, fully functional)
# ══════════════════════════════════════════════════════════════════
def put_text(img, text, x, y, scale, color, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_hud(frame, state, ear_val, blink_count, detected_objects,
             break_msg, fps, session_sec, flash, model_ok_phone, model_ok_smoke, drowsy_count):
    H, W = frame.shape[:2]
    main_color = COL[state]
    panel_h = 138
    bot_h = 46

    # top panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, panel_h), COL["bg"], -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (0, 0), (7, panel_h), main_color, -1)

    # state badge
    badges = {
        "ALERT": ("  ALERT", "All systems normal", COL["good"]),
        "DROWSY": ("! DROWSY", "Eyes closing – Wake up!", COL["DROWSY"]),
        "DISTRACTED": ("~ DISTRACTED", "Eyes off road!", COL["DISTRACTED"]),
        "PHONE": ("P PHONE", "Put down your phone!", COL["PHONE"]),
        "SMOKING": ("S SMOKING", "Stop smoking while driving!", COL["SMOKING"]),
    }
    title, subtitle, badge_col = badges.get(state, (state, "", main_color))
    put_text(frame, title, 16, 38, 1.0, badge_col, 2, cv2.FONT_HERSHEY_DUPLEX)
    put_text(frame, subtitle, 16, 58, 0.48, COL["dim"])
    cv2.line(frame, (12, 67), (W - 12, 67), COL["line"], 1)

    # stats
    mm, ss = divmod(int(session_sec), 60)
    stats = [("EAR", f"{ear_val:.3f}", "Eye Ratio"),
             ("BLINKS", str(blink_count), "Count"),
             ("FPS", f"{fps:.0f}", "Camera"),
             ("TIME", f"{mm:02d}:{ss:02d}", "Session"),
             ("DROWSY", str(drowsy_count), "Events")]
    cx = 16
    step = (W - 140) // len(stats)
    for label, value, hint in stats:
        put_text(frame, label, cx, 88, 0.37, COL["dim"])
        put_text(frame, value, cx, 112, 0.70, COL["accent"], 1)
        put_text(frame, hint, cx, 128, 0.29, (45, 45, 62))
        cx += step

    # model indicators
    mx = W - 160
    for i, (label, ok) in enumerate([("Phone", model_ok_phone), ("Smoke", model_ok_smoke)]):
        dx = mx + i * 70
        color = COL["good"] if ok else (80, 40, 40)
        cv2.circle(frame, (dx, 100), 5, color, -1)
        put_text(frame, label, dx + 10, 105, 0.35, COL["dim"])

    # EAR bar
    gx = W - 24
    gt = panel_h + 12
    gb = H - (bot_h + 12 if detected_objects else 12)
    # draw vertical bar
    cv2.rectangle(frame, (gx, gt), (gx + 14, gb), (22, 22, 38), -1)
    hh = gb - gt
    fill_h = int(hh * min(max(ear_val, 0) / 0.5, 1.0))
    if fill_h > 0:
        cv2.rectangle(frame, (gx, gb - fill_h), (gx + 14, gb), COL["accent"], -1)
    cv2.rectangle(frame, (gx, gt), (gx + 14, gb), (55, 55, 75), 1)
    thresh_y = gb - int(hh * (EAR_THRESH / 0.5))
    cv2.line(frame, (gx - 4, thresh_y), (gx + 18, thresh_y), (0, 200, 255), 1)
    put_text(frame, "EAR", gx - 1, gt - 8, 0.28, COL["dim"])

    # bottom bar
    by = H - (bot_h + 55 if detected_objects else 55)
    put_text(frame, "DROWSY EVENTS", 14, by, 0.33, COL["dim"])
    # drowsy horizontal bar
    cv2.rectangle(frame, (14, by + 5), (14 + 130, by + 12), (28, 28, 44), -1)
    fill_w = int(130 * min(drowsy_count / max(BREAK_AFTER, 1), 1.0))
    if fill_w > 0:
        cv2.rectangle(frame, (14, by + 5), (14 + fill_w, by + 12), COL["DROWSY"], -1)
    cv2.rectangle(frame, (14, by + 5), (14 + 130, by + 12), (55, 55, 75), 1)
    put_text(frame, f"{drowsy_count}/{BREAK_AFTER}", 14, by + 24, 0.34, COL["dim"])

    if detected_objects:
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, H - bot_h), (W, H), COL["bg"], -1)
        cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (0, H - bot_h), (7, H), main_color, -1)
        obj_str = "  DETECTED:  " + "    |    ".join(o.upper() for o in detected_objects)
        put_text(frame, obj_str, 14, H - 14, 0.60, main_color, 2)

    if flash and state != "ALERT":
        cv2.rectangle(frame, (0, 0), (W, H), main_color, 10)

    if break_msg:
        bw, bh = 620, 48
        bx2 = W // 2 - bw // 2
        by2 = H // 2 - bh // 2
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (bx2 - 12, by2 - 6), (bx2 + bw + 12, by2 + bh + 6), (18, 8, 8), -1)
        cv2.addWeighted(overlay3, 0.88, frame, 0.12, 0, frame)
        put_text(frame, break_msg, bx2, by2 + bh - 10, 0.78, (60, 60, 255), 2, cv2.FONT_HERSHEY_DUPLEX)

    wy = H - (bot_h + 8) if detected_objects else H - 8
    put_text(frame, "DRIVER MONITOR v6.2", W - 228, wy, 0.34, (32, 32, 50))

# ══════════════════════════════════════════════════════════════════
#  MAIN PROGRAM
# ══════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 62)
    print("  AI Driver Monitoring System  v6.2")
    print("  Telegram alerts: ONLY for smoking | Voice: all events (fixed)")
    if _CUDA:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU    : {gpu_name}  ({vram:.1f} GB VRAM)  FP16: enabled")
    else:
        print("  Device : CPU – CUDA not found")
    print("═" * 62 + "\n")

    # Load YOLO models
    os.makedirs("models", exist_ok=True)
    if not os.path.isfile(PHONE_MODEL):
        print("[YOLO] Downloading yolov8n.pt ...")
        YOLO("yolov8n.pt")
        import shutil
        shutil.move("yolov8n.pt", PHONE_MODEL)

    phone_model = None
    smoke_model = None
    phone_ok = False
    smoke_ok = False

    try:
        phone_model = YOLO(PHONE_MODEL)
        phone_model.to(_DEVICE)
        if _HALF:
            phone_model.model.half()
        phone_ok = True
        print(f"[YOLO] Phone model loaded on {_DEVICE.upper()}")
    except Exception as e:
        print(f"[YOLO] Phone model error: {e}")

    if os.path.isfile(SMOKE_MODEL):
        try:
            smoke_model = YOLO(SMOKE_MODEL)
            smoke_model.to(_DEVICE)
            if _HALF:
                smoke_model.model.half()
            smoke_ok = True
            print(f"[YOLO] Smoke model loaded on {_DEVICE.upper()} | angles: {SMOKE_ANGLES}°")
        except Exception as e:
            print(f"[YOLO] Smoke model error: {e}")
    else:
        print(f"[YOLO] Smoke model not found at {SMOKE_MODEL}")

    voice = Voice()
    esp = esp_connect(SERIAL_PORT, BAUD_RATE)
    log_fh, log_writer = log_open(LOG_FILE)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Camera] ERROR: cannot open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("[Camera] Ready. Press Q to quit.\n")

    # State variables
    ear_counter = 0
    blinks = 0
    eye_was_open = True
    distraction_start = None
    drowsy_event_count = 0
    last_esp_cmd = ""
    break_msg = ""
    break_msg_end = 0.0
    frame_num = 0
    session_start = time.time()
    fps_timer = time.time()
    fps_counter = 0
    fps = 0.0
    flash_state = False
    flash_timer = 0.0
    last_ear_value = 0.30
    last_log_state = "ALERT"
    last_log_time = 0.0
    previous_state = "ALERT"
    smoking_alert_sent = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        fps_counter += 1
        now = time.time()
        session_duration = now - session_start

        if fps_counter >= 30:
            fps = fps_counter / (now - fps_timer + 1e-9)
            fps_timer = now
            fps_counter = 0

        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        smoking_flag = False
        phone_flag = False
        drowsy_flag = False
        distracted_flag = False

        # MediaPipe face mesh
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_ear = ear(lm, L_EYE, W, H)
            right_ear = ear(lm, R_EYE, W, H)
            current_ear = (left_ear + right_ear) / 2.0
            last_ear_value = current_ear

            eye_open = current_ear > EAR_THRESH
            if eye_was_open and not eye_open:
                blinks += 1
            eye_was_open = eye_open

            if current_ear < EAR_THRESH:
                ear_counter += 1
            else:
                ear_counter = 0
            if ear_counter >= EAR_FRAMES:
                drowsy_flag = True

            if is_head_turned(lm, W, H):
                if distraction_start is None:
                    distraction_start = now
                elif now - distraction_start >= DISTRACT_SEC:
                    distracted_flag = True
            else:
                distraction_start = None

            for idx in L_EYE + R_EYE:
                cx = int(lm[idx].x * W)
                cy = int(lm[idx].y * H)
                cv2.circle(frame, (cx, cy), 2, (0, 180, 220), -1)
        else:
            ear_counter = 0
            if distraction_start is None:
                distraction_start = now
            elif now - distraction_start >= DISTRACT_SEC:
                distracted_flag = True

        # YOLO every 3 frames
        if frame_num % 3 == 0:
            if phone_model:
                try:
                    phone_results = phone_model(frame, verbose=False, device=_DEVICE, half=_HALF)[0].boxes
                    for box in phone_results:
                        cls = phone_model.names[int(box.cls[0])].lower()
                        conf = float(box.conf[0])
                        if cls == "cell phone" and conf > PHONE_CONF:
                            phone_flag = True
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), COL["PHONE"], 2)
                            put_text(frame, f"PHONE {conf:.0%}", x1, y1-8, 0.55, COL["PHONE"], 2)
                except Exception:
                    pass

            if smoke_model:
                try:
                    smoke_boxes = detect_smoking(smoke_model, frame, SMOKE_CONF, SMOKE_ANGLES)
                    for (x1, y1, x2, y2, conf) in smoke_boxes:
                        smoking_flag = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COL["SMOKING"], 2)
                        put_text(frame, f"CIGARETTE {conf:.0%}", x1, y1-8, 0.55, COL["SMOKING"], 2)
                except Exception:
                    pass

        # Determine final state (priority: smoking > phone > drowsy > distracted)
        if smoking_flag:
            final_state = "SMOKING"
        elif phone_flag:
            final_state = "PHONE"
        elif drowsy_flag:
            final_state = "DROWSY"
        elif distracted_flag:
            final_state = "DISTRACTED"
        else:
            final_state = "ALERT"

        # Telegram only for smoking (once per event)
        if final_state == "SMOKING":
            if not smoking_alert_sent:
                send_telegram_alert(frame)
                smoking_alert_sent = True
        else:
            smoking_alert_sent = False

        # Drowsy counter & break
        if final_state == "DROWSY":
            drowsy_event_count += 1
            if drowsy_event_count >= BREAK_AFTER:
                break_msg = "  FATIGUE DETECTED  –  PLEASE TAKE A BREAK  "
                break_msg_end = now + 7.0
        if break_msg and now > break_msg_end:
            break_msg = ""

        # Logging
        if final_state != "ALERT" and (final_state != last_log_state or (now - last_log_time) >= LOG_GAP):
            log_write(log_writer, final_state, last_ear_value)
            log_fh.flush()
            last_log_state = final_state
            last_log_time = now

        # Voice alerts (all states, including smoking)
        if final_state != previous_state:
            voice.end_state(previous_state)
        voice.tick(final_state)
        previous_state = final_state

        # ESP32 command
        cmd_map = {"ALERT": "NORMAL", "DROWSY": "DROWSY", "DISTRACTED": "DISTRACTED", "PHONE": "PHONE", "SMOKING": "SMOKING"}
        cmd = cmd_map.get(final_state, "NORMAL")
        if cmd != last_esp_cmd:
            esp_send(esp, cmd)
            last_esp_cmd = cmd

        # Flash effect
        if now - flash_timer >= 0.5:
            flash_state = not flash_state
            flash_timer = now

        detected_objects = []
        if phone_flag:
            detected_objects.append("phone")
        if smoking_flag:
            detected_objects.append("cigarette")

        draw_hud(frame, final_state, last_ear_value, blinks, detected_objects,
                 break_msg, fps, session_duration, flash_state,
                 phone_ok, smoke_ok, drowsy_event_count)

        cv2.imshow("Driver Monitoring System | Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n[System] Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    log_fh.close()
    voice.stop()
    if esp:
        esp_send(esp, "NORMAL")
        esp.close()
    print(f"[Stats] Blinks: {blinks} | Drowsy events: {drowsy_event_count} | Log: {LOG_FILE}")

if __name__ == "__main__":
    main()