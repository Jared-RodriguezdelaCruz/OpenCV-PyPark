import cv2
import numpy as np
import joblib
from skimage.transform import resize

CAMERA_URL = "http://192.168.11.31:4747/video"

class ParkingSystem:
    def __init__(self, mask_path=None, model_path='parking_model.pkl'):
        # Cargar máscara o definir espacios manualmente
        if mask_path:
            self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if self.mask is None:
                raise ValueError(f"No se pudo cargar la máscara en {mask_path}")
            self.spots = self.get_parking_spots_bboxes()
        else:
            # Definir espacios manualmente (coordenadas x1,y1,x2,y2)
            self.parking_spots = [(50, 80, 150, 200), (160, 80, 260, 200)]  # Ejemplo
            self.mask = self.create_parking_mask((480, 640), self.parking_spots)
            self.spots = [(x1,y1,x2-x1,y2-y1) for x1,y1,x2,y2 in self.parking_spots]
        
        # Cargar modelo
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            self.model = None
        
        self.spots_status = [None] * len(self.spots)
        self.previous_frame = None
    
    def get_parking_spots_bboxes(self):
        # Corrección aquí: usar cv2.CV_32S en lugar de cv2.CV_32S
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.mask, connectivity=4, ltype=cv2.CV_32S)
        
        slots = []
        for i in range(1, num_labels):  # Ignorar el fondo (etiqueta 0)
            x1 = int(stats[i, cv2.CC_STAT_LEFT])
            y1 = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            slots.append((x1, y1, w, h))
        return slots
    
    def create_parking_mask(self, frame_shape, parking_spots):
        """
        Crea una máscara binaria con los espacios de estacionamiento definidos
        frame_shape: dimensiones del frame de video (height, width)
        parking_spots: lista de coordenadas de espacios [(x1,y1,x2,y2), ...]
        """
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        for spot in parking_spots:
            x1, y1, x2, y2 = spot
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    def detect_vehicles(self, spot_img, background_img, threshold=25):
        """
        Detecta si hay un vehículo en el espacio de estacionamiento
        """
        # Convertir a escala de grises
        gray_spot = cv2.cvtColor(spot_img, cv2.COLOR_BGR2GRAY)
        gray_back = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
        
        # Calcular diferencia absoluta
        diff = cv2.absdiff(gray_back, gray_spot)
        
        # Aplicar umbral
        _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Operaciones morfológicas para eliminar ruido
        kernel = np.ones((3,3), np.uint8)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Contar píxeles diferentes
        changed_pixels = cv2.countNonZero(thresholded)
        total_pixels = spot_img.shape[0] * spot_img.shape[1]
        change_ratio = changed_pixels / total_pixels
        
        return change_ratio > 0.1  # Si más del 10% de píxeles cambiaron
    
    def process_frame(self, frame):
        if self.model is None:
            print("Error: Modelo no cargado")
            return frame
            
        for i, (x1, y1, w, h) in enumerate(self.spots):
            # Asegurarse de que las coordenadas están dentro del frame
            if y1+h > frame.shape[0] or x1+w > frame.shape[1]:
                continue
                
            spot_img = frame[y1:y1+h, x1:x1+w]
            
            try:
                # Clasificar el espacio
                img_resized = resize(spot_img, (15, 15, 3))
                flat_data = img_resized.flatten().reshape(1, -1)
                prediction = self.model.predict(flat_data)
                
                self.spots_status[i] = prediction[0] == 0  # 0 es vacío, 1 es ocupado
                
                # Dibujar rectángulo
                color = (0, 255, 0) if self.spots_status[i] else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), color, 2)
                
                # Mostrar estado en el espacio
                status_text = "Libre" if self.spots_status[i] else "Ocupado"
                cv2.putText(frame, status_text, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            except Exception as e:
                print(f"Error procesando espacio {i}: {str(e)}")
                continue
        
        # Mostrar contador
        available = sum(1 for status in self.spots_status if status is True)
        total = len([status for status in self.spots_status if status is not None])
        cv2.putText(frame, f'Disponibles: {available}/{total}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

if __name__ == "__main__":
    try:
        parking_system = ParkingSystem(mask_path='mask.png')
        
        cap = cv2.VideoCapture(CAMERA_URL)  # O usar un video: cv2.VideoCapture('parking.mp4')
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = parking_system.process_frame(frame)
            
            cv2.imshow('Parking System', processed_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error en la ejecución: {str(e)}")