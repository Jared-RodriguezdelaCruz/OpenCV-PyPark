import pickle

from skimage.transform import resize
import numpy as np
import cv2


EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("model.p", "rb"))

def empty_or_not(spot_bgr: np.ndarray) -> bool:
    # Redimensionar y asegurar tipo
    img_resized = resize(spot_bgr, (15, 15, 3), anti_aliasing=True)
    img_resized = np.asarray(img_resized, dtype=np.float32)  # Asegura que sea ndarray

    flat_data = np.array([img_resized.flatten()])  # Ahora sí, sin advertencias

    y_output = MODEL.predict(flat_data)
    return y_output[0] == 0




def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots