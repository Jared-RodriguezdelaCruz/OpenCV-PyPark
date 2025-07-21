import cv2
import numpy as np
import os
import time
from datetime import datetime

CAMERA_URL = "http://192.168.11.31:4747/video"

class DataCollector:
    def __init__(self, manual_mask=False):
        # Configuración de cámara
        self.cap = cv2.VideoCapture(CAMERA_URL)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")
        
        # Obtener dimensiones del frame
        ret, test_frame = self.cap.read()
        if not ret:
            raise RuntimeError("No se pudo leer el frame de la cámara")
        
        self.frame_height, self.frame_width = test_frame.shape[:2]
        
        # Create Mask binaria automática o manual
        if manual_mask:
            self.mask = self.create_mask_manually()
        else:
            self.mask = self.create_auto_mask()
        
        # Crear directorios
        os.makedirs('data/empty', exist_ok=True)
        os.makedirs('data/occupied', exist_ok=True)
        
        # Estadísticas
        self.captured_empty = len(os.listdir('data/empty'))
        self.captured_occupied = len(os.listdir('data/occupied'))
        
        # Objetivos
        self.target_empty = 100
        self.target_occupied = 100
    
    def create_auto_mask(self):
        """Crea una máscara automática con 3 espacios de estacionamiento"""
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        # Definir 3 espacios de estacionamiento (ajusta estos valores según tu cámara)
        spots = [
            (int(self.frame_width*0.1), int(self.frame_height*0.2), 
             int(self.frame_width*0.3), int(self.frame_height*0.6)),
            (int(self.frame_width*0.4), int(self.frame_height*0.2),
             int(self.frame_width*0.6), int(self.frame_height*0.6)),
            (int(self.frame_width*0.7), int(self.frame_height*0.2),
             int(self.frame_width*0.9), int(self.frame_height*0.6))
        ]
        
        for (x1, y1, x2, y2) in spots:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        return mask
    
    def create_mask_manually(self):
        """Permite dibujar la máscara manualmente con el mouse"""
        print("\nModo de creación de máscara manual")
        print("1. Haz clic y arrastra para dibujar rectángulos (espacios de parking)")
        print("2. Presiona 's' para guardar la máscara")
        print("3. Presiona 'q' para cancelar")
        
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        drawing = False
        ix, iy = -1, -1
        
        def draw_rectangle(event, x, y, flags, param):
            nonlocal ix, iy, drawing, mask
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    temp_mask = mask.copy()
                    cv2.rectangle(temp_mask, (ix, iy), (x, y), 255, -1)
                    frame = self.get_current_frame()
                    display = cv2.addWeighted(frame, 0.7, cv2.cvtColor(temp_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
                    cv2.imshow('Create Mask', display)
                    
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                cv2.rectangle(mask, (ix, iy), (x, y), 255, -1)
                frame = self.get_current_frame()
                display = cv2.addWeighted(frame, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
                cv2.imshow('Create Mask', display)
        
        cv2.namedWindow('Create Mask')
        cv2.setMouseCallback('Create Mask', draw_rectangle)
        
        while True:
            frame = self.get_current_frame()
            display = cv2.addWeighted(frame, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
            cv2.imshow('Create Mask', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cv2.destroyWindow('Create Mask')
                return mask
            elif key == ord('q'):
                cv2.destroyWindow('Create Mask')
                raise ValueError("Creación de máscara cancelada")
    
    def get_current_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("No se pudo obtener el frame de la cámara")
        return frame
    
    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
    
    def calculate_remaining(self):
        return {
            'empty': max(0, self.target_empty - self.captured_empty),
            'occupied': max(0, self.target_occupied - self.captured_occupied)
        }
    
    def capture_samples(self):
        print("\n=== MODO CAPTURA ===")
        print(f"Faltan: {self.calculate_remaining()['empty']} vacíos | {self.calculate_remaining()['occupied']} ocupados")
        
        while True:
            try:
                frame = self.get_current_frame()
                
                # Aplicar máscara
                masked = cv2.bitwise_and(frame, frame, mask=self.mask)
                
                # Mostrar instrucciones
                instructions = [
                    "INSTRUCCIONES:",
                    "1. Presiona 'v' para capturar espacio VACIO",
                    "2. Presiona 'o' para capturar espacio OCUPADO",
                    "3. Presiona 'q' para terminar",
                    f"Capturados: Vacios={self.captured_empty}/{self.target_empty}",
                    f"          Ocupados={self.captured_occupied}/{self.target_occupied}"
                ]
                
                display_frame = frame.copy()
                y_offset = 30
                for line in instructions:
                    cv2.putText(display_frame, line, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    y_offset += 25
                
                # Mostrar áreas de captura
                contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)
                
                cv2.imshow('Captura de Datos', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('v') and self.captured_empty < self.target_empty:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"data/empty/empty_{timestamp}.jpg"
                    cv2.imwrite(filename, masked)
                    self.captured_empty += 1
                    print(f"Capturado vacío: {filename}")
                    
                elif key == ord('o') and self.captured_occupied < self.target_occupied:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"data/occupied/occupied_{timestamp}.jpg"
                    cv2.imwrite(filename, masked)
                    self.captured_occupied += 1
                    print(f"Capturado ocupado: {filename}")
                    
                elif key == ord('q'):
                    break
                
                remaining = self.calculate_remaining()
                if remaining['empty'] <= 0 and remaining['occupied'] <= 0:
                    print("\n¡Se ha alcanzado el número objetivo de muestras!")
                    break
            
            except Exception as e:
                print(f"Error durante la captura: {str(e)}")
                break
        
        print("\nResumen final:")
        print(f"- Espacios vacíos capturados: {self.captured_empty}/{self.target_empty}")
        print(f"- Espacios ocupados capturados: {self.captured_occupied}/{self.target_occupied}")

if __name__ == "__main__":
    try:
        print("""
        ======================================
        SISTEMA DE CAPTURA PARA ENTRENAMIENTO
        ======================================
        
        Recomendaciones:
        1. Para espacios VACÍOS: 
           - Asegúrate de que el espacio esté completamente libre
           - Captura en diferentes condiciones de luz
           
        2. Para espacios OCUPADOS:
           - Usa diferentes tipos de vehículos
           - Varía la posición dentro del espacio
        """)
        
        # Verificar si ya hay datos
        if os.path.exists('data/empty') and os.path.exists('data/occupied'):
            print(f"\nYa existen {len(os.listdir('data/empty'))} imágenes vacías y {len(os.listdir('data/occupied'))} ocupadas")
        
        use_manual = input("¿Deseas crear la máscara manualmente? (s/n): ").lower() == 's'
        
        collector = DataCollector(manual_mask=use_manual)
        collector.capture_samples()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        cv2.destroyAllWindows()