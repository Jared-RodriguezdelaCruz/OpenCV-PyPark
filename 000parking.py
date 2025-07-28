# laptop_parking_monitor.py

import cv2, numpy as np
from collections import deque
from datetime import datetime
from pymongo import MongoClient

# === Configuración ===
CAMERA_URL = "http://192.168.13.251:4747/video"
MIN_AREA = 500
HISTORY_LENGTH = 5
client = MongoClient("mongodb+srv://ANotRealName:54321@pypark.3exozxa.mongodb.net/")
db = client["parking_monitor"]
collection = db["estados"]

slots, slot_states, slot_history = [], [], []
drawing, start_point, frame_copy = False, None, None
background = None

# === Inicializar cámara ===
def init_camera():
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("❌ Error al conectar con la cámara"); exit()
    return cap

# === Calibrar fondo ===
def calibrate_background(cap, num_frames=30):
    print("🧭 Calibrando fondo...")
    avg_frame = None
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        avg_frame = gray.copy().astype("float") if avg_frame is None else cv2.accumulateWeighted(gray, avg_frame, 0.5)
    return cv2.convertScaleAbs(avg_frame)  # type: ignore

# === Dibujar slots manualmente ===
def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True; start_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        slots.append((start_point, end_point))
        slot_states.append(False)
        slot_history.append(deque([False]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH))
        cv2.rectangle(frame_copy, start_point, end_point, (0, 255, 0), 2)

# === Detectar ocupación por área ===
def detect_occupation(roi, bg_roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(bg_roi, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=1)
    thresh = cv2.erode(thresh, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) > MIN_AREA for c in contours)

# === Actualizar MongoDB ===
def enviar_a_mongodb(ocupaciones):
    documento = {
        "timestamp": datetime.utcnow(),
        "totalSpots": len(ocupaciones),
        "availableSpots": ocupaciones.count("empty"),
        "spots": [{"index": i, "status": estado} for i, estado in enumerate(ocupaciones)]
    }
    collection.insert_one(documento)
    print("📤 Estado actualizado en MongoDB")

# === Loop principal ===
def main():
    global frame_copy, background
    cap = init_camera()

    # Paso 1: congelar frame
    while True:
        ret, frame = cap.read()
        if not ret: continue
        cv2.imshow("Vista previa", frame)
        if cv2.waitKey(1) & 0xFF in [ord('c'), 13]:
            frame_copy = frame.copy(); break
    cv2.destroyWindow("Vista previa")

    # Paso 2: dibujar slots
    cv2.namedWindow("Define slots")
    cv2.setMouseCallback("Define slots", mouse_callback)
    while True:
        cv2.imshow("Define slots", frame_copy)
        if cv2.waitKey(1) & 0xFF in [ord('c'), 13]: break
    cv2.destroyWindow("Define slots")

    if not slots: print("⚠️ No hay slots definidos"); return
    print("🕐 Calibra fondo con slots vacíos...")
    background = calibrate_background(cap)

    print("🚦 Monitoreando ocupación (q para salir)")
    while True:
        ret, frame = cap.read()
        if not ret: break

        estados = []
        for idx, (pt1, pt2) in enumerate(slots):
            x1, y1 = pt1; x2, y2 = pt2
            roi, bg_roi = frame[y1:y2, x1:x2], background[y1:y2, x1:x2]
            if roi.size == 0 or bg_roi.size == 0: continue
            ocupado = detect_occupation(roi, bg_roi)
            slot_history[idx].append(ocupado)
            promedio = sum(slot_history[idx]) / HISTORY_LENGTH
            final = promedio > 0.6
            slot_states[idx] = final
            estados.append("occupied" if final else "empty")
            color = (0, 0, 255) if final else (0, 255, 0)
            cv2.rectangle(frame, pt1, pt2, color, 2)
            texto = "Ocupado" if final else "Libre"
            cv2.putText(frame, texto, (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        enviar_a_mongodb(estados)
        cv2.imshow("Estado actual", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()