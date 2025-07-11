import cv2
import pyautogui

def mostrar_camaras(camara1=0, camara2=1):
    """
    Muestra el video de dos cámaras en ventanas separadas, una a la izquierda y otra a la derecha.
    
    camara1 (int): Índice de la primera cámara (por defecto 0)
    camara2 (int): Índice de la segunda cámara (por defecto 1)
    """
    cap1 = cv2.VideoCapture(camara1)
    cap2 = cv2.VideoCapture(camara2)

    if not cap1.isOpened():
        print(f"No se pudo abrir la cámara {camara1}")
        return
    if not cap2.isOpened():
        print(f"No se pudo abrir la cámara {camara2}")
        return

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    screen_width, screen_height = pyautogui.size()

    total_width = width1 * 2
    start_x = (screen_width - total_width) // 2
    start_y = (screen_height - height1) // 2

    # Crear ventanas
    cv2.namedWindow('Cam1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cam2', cv2.WINDOW_NORMAL)

    cv2.moveWindow('Cam1', start_x, start_y)
    cv2.moveWindow('Cam2', start_x + width1, start_y)

    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                print("No se pudo recibir frame. Saliendo...")
                break
            
            cv2.imshow('Cam1', frame1)
            cv2.imshow('Cam2', frame2)
            
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Mostrando video de 2 cámaras. Presione 'q' para salir.")
    mostrar_camaras()