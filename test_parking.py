import cv2
import numpy as np
from collections import deque

# ===== CONFIGURACIÓN =====
CAMERA_URL = "http://192.168.100.89:4747/video"
MIN_AREA = 500
HISTORY_LENGTH = 5

drawing = False
start_point = None
slots = []
background = None
slot_states = []
slot_history = []

def init_camera():
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("❌ Error al conectar con la cámara")
        exit()
    return cap

def calibrate_background(cap, num_frames=30):
    print("Calibrando fondo...")
    avg_frame = None
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if avg_frame is None:
            avg_frame = gray.copy().astype("float")
        else:
            cv2.accumulateWeighted(gray, avg_frame, 0.5)
    return cv2.convertScaleAbs(avg_frame) # type: ignore

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        slots.append((start_point, end_point))
        slot_states.append(False)
        slot_history.append(deque([False]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH))
        cv2.rectangle(frame_copy, start_point, end_point, (0, 255, 0), 2) # type: ignore

def detect_occupation(roi, background_roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(background_roi, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > MIN_AREA:
            return True
    return False

def main():
    global frame_copy, background

    cap = init_camera()

    # === MOSTRAR VIDEO EN VIVO ANTES DE DIBUJAR ===
    print("Visualizando el área... Presiona 'c' para congelar el frame y empezar a dibujar los slots")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Vista previa (presiona 'c')", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == 13:
            frame_copy = frame.copy()
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Vista previa (presiona 'c')")

    # === DIBUJAR LOS SLOTS ===
    cv2.namedWindow("Dibuja los slots (Presiona 'c' para continuar)")
    cv2.setMouseCallback("Dibuja los slots (Presiona 'c' para continuar)", mouse_callback)

    while True:
        cv2.imshow("Dibuja los slots (Presiona 'c' para continuar)", frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == 13:
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Dibuja los slots (Presiona 'c' para continuar)")

    if not slots:
        print("⚠️ No se definieron espacios de estacionamiento")
        cap.release()
        return

    print("Por favor, asegúrate que los espacios estén vacíos para calibrar el fondo...")
    background = calibrate_background(cap)

    print("Iniciando monitoreo... (Presiona 'q' para salir)")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al capturar frame")
            break

        for idx, (pt1, pt2) in enumerate(slots):
            x1, y1 = pt1
            x2, y2 = pt2
            if x1 >= x2 or y1 >= y2:
                continue
            roi = frame[y1:y2, x1:x2]
            bg_roi = background[y1:y2, x1:x2]
            if roi.size == 0 or bg_roi.size == 0:
                continue
            occupied = detect_occupation(roi, bg_roi)
            slot_history[idx].append(occupied)
            slot_states[idx] = sum(slot_history[idx]) / HISTORY_LENGTH > 0.6
            color = (0, 0, 255) if slot_states[idx] else (0, 255, 0)
            cv2.rectangle(frame, pt1, pt2, color, 2)
            status = "Ocupado" if slot_states[idx] else "Libre"
            cv2.putText(frame, status, (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Estado del estacionamiento", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()