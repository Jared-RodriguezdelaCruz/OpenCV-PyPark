import cv2
import numpy as np
from datetime import datetime

# Simulación de la base de datos MongoDB
parking_slots = [{"ocupado": False, "entrada": None, "salida": None} for _ in range(16)]

# Configuración de cámara
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SLOT_WIDTH = 60
SLOT_HEIGHT = 80

# Zona de entrada/salida
ENTRY_ZONE = (FRAME_WIDTH - 120, FRAME_HEIGHT - 100, 100, 80)  # (x, y, w, h)
EXIT_ZONE = (20, 20, 100, 80)

# Inicializar cámara y background subtractor
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

# Definir posiciones de los slots (en el centro del frame, 8 a la izquierda y 8 a la derecha)
slot_positions = []
start_x = FRAME_WIDTH // 2 - 4 * SLOT_WIDTH
start_y = FRAME_HEIGHT // 2 - SLOT_HEIGHT

for row in range(2):  # 2 filas
    for col in range(8):  # 8 columnas
        x = start_x + col * SLOT_WIDTH
        y = start_y + row * SLOT_HEIGHT
        slot_positions.append((x, y))

def detectar_direccion(cx, cy):
    ex, ey, ew, eh = ENTRY_ZONE
    sx, sy, sw, sh = EXIT_ZONE

    if ex <= cx <= ex + ew and ey <= cy <= ey + eh:
        return "entrada"
    elif sx <= cx <= sx + sw and sy <= cy <= sy + sh:
        return "salida"
    return None

def asignar_slot():
    for idx, slot in enumerate(parking_slots):
        if not slot["ocupado"]:
            return idx
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    mask = fgbg.apply(frame)

    # Detección de movimiento
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) < 500:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        direccion = detectar_direccion(cx, cy)

        if direccion == "entrada":
            slot_id = asignar_slot()  
            if slot_id is not None and not parking_slots[slot_id]["ocupado"]:
                parking_slots[slot_id]["ocupado"] = True
                parking_slots[slot_id]["entrada"] = datetime.now()
                print(f"[ENTRADA] Carro en slot {slot_id + 1} a las {parking_slots[slot_id]['entrada']}")

        elif direccion == "salida":
            for idx in range(len(parking_slots)-1, -1, -1):
                if parking_slots[idx]["ocupado"]:
                    parking_slots[idx]["ocupado"] = False
                    parking_slots[idx]["salida"] = datetime.now()
                    print(f"[SALIDA] Slot {idx + 1} liberado a las {parking_slots[idx]['salida']}")
                    break

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    # Dibujar slots
    for idx, (x, y) in enumerate(slot_positions):
        color = (0, 255, 0) if not parking_slots[idx]["ocupado"] else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + SLOT_WIDTH, y + SLOT_HEIGHT), color, 2)
        cv2.putText(frame, str(idx + 1), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Dibujar zonas de entrada/salida
    cv2.rectangle(frame, (ENTRY_ZONE[0], ENTRY_ZONE[1]),
                  (ENTRY_ZONE[0] + ENTRY_ZONE[2], ENTRY_ZONE[1] + ENTRY_ZONE[3]),
                  (255, 0, 0), 2)
    cv2.putText(frame, "ENTRADA", (ENTRY_ZONE[0], ENTRY_ZONE[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.rectangle(frame, (EXIT_ZONE[0], EXIT_ZONE[1]),
                  (EXIT_ZONE[0] + EXIT_ZONE[2], EXIT_ZONE[1] + EXIT_ZONE[3]),
                  (0, 255, 255), 2)
    cv2.putText(frame, "SALIDA", (EXIT_ZONE[0], EXIT_ZONE[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Mostrar
    cv2.imshow("Estacionamiento", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
