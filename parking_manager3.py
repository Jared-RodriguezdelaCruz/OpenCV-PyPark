""" 
    Necesitamos de una mascara con las marcas exactas en los spots para que esto funcione 
    """

import cv2
import numpy as np

from util import get_parking_spots_bboxes, empty_or_not

MASK_PATH = './mask.png'
DRAW_INTERVAL = 30
DIFF_THRESHOLD = 0.4


# --- Utilidades ---
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


# Leer la máscara en escala de grises
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

cap = cv2.VideoCapture(0)

# Convertir a binaria
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY) # type: ignore
# Asegurar tipo correcto
binary_mask = binary_mask.astype(np.uint8)

# Obtener componentes conectados
num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=4, ltype=cv2.CV_32S)

spots = get_parking_spots_bboxes(labels)

spots_status = [False] * len(spots)
diffs = [0.0] * len(spots)
previous_frame = None
frame_nmr = 0

import sys
print(sys.path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_nmr % DRAW_INTERVAL == 0:
        if previous_frame is not None:
            for i, (x, y, w, h) in enumerate(spots):
                current_crop = frame[y:y + h, x:x + w]
                prev_crop = previous_frame[y:y + h, x:x + w]
                diffs[i] = calc_diff(current_crop, prev_crop)

        # Determinar qué espacios verificar
        if previous_frame is None:
            indices_to_check = range(len(spots))
        else:
            max_diff = np.max(diffs)
            indices_to_check = [i for i, d in enumerate(diffs) if d / max_diff > DIFF_THRESHOLD]

        # Clasificar espacios
        for i in indices_to_check:
            x, y, w, h = spots[i]
            spot_crop = frame[y:y + h, x:x + w]
            spots_status[i] = empty_or_not(spot_crop)

        previous_frame = frame.copy()

    # Dibujar resultados
    for i, (x, y, w, h) in enumerate(spots):
        color = (0, 255, 0) if spots_status[i] else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Mostrar contador de espacios disponibles
    available = sum(spots_status)
    total = len(spots_status)
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, f'Available spots: {available} / {total}', (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostrar frame
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
