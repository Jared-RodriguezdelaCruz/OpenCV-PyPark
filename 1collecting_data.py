import cv2
import numpy as np
import os
import time
from datetime import datetime

CAMERA_URL = "http://192.168.11.31:4747/video"

class DataCollector:
    def __init__(self, mask_path):
        # Cargar y verificar la máscara
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise ValueError(f"No se pudo cargar la máscara en {mask_path}")
        
        # Verificar dimensiones
        print(f"Dimensiones de la máscara: {self.mask.shape}")
        
        # Crear directorios
        os.makedirs('data/empty', exist_ok=True)
        os.makedirs('data/occupied', exist_ok=True)
        
        # Estadísticas
        self.captured_empty = len(os.listdir('data/empty'))
        self.captured_occupied = len(os.listdir('data/occupied'))
        
        # Objetivos
        self.target_empty = 100
        self.target_occupied = 100
        
        
        # Configuración de cámara
        self.cap = cv2.VideoCapture(CAMERA_URL)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")
        
        # Ajustar resolución para que coincida con la máscara
        ret, test_frame = self.cap.read()
        if ret:
            if test_frame.shape[:2] != self.mask.shape[:2]:
                print(f"Ajustando resolución de cámara a {self.mask.shape[1]}x{self.mask.shape[0]}")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.mask.shape[1])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.mask.shape[0])
    
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
            ret, frame = self.cap.read()
            if not ret:
                print("Error al capturar frame")
                break
            
            # Verificar el tamaño del frame
            if frame is None or frame.size == 0:
                print("Frame capturado es None o vacío")
                break
            
            print(f"Tamaño del frame capturado: {frame.shape}")  # Agregar esta línea para depuración
            
            # Redimensionar frame si es necesario
            if frame.shape[:2] != self.mask.shape[:2]:
                frame = cv2.resize(frame, (self.mask.shape[1], self.mask.shape[0]))
            
            # Aplicar máscara
            try:
                masked = cv2.bitwise_and(frame, frame, mask=self.mask)
            except Exception as e:
                print(f"Error al aplicar máscara: {str(e)}")
                print(f"Frame shape: {frame.shape}, Mask shape: {self.mask.shape}")
                break
            
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
        
        input("Presiona Enter para comenzar la captura...")
        
        collector = DataCollector('mask.png')
        collector.capture_samples()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        cv2.destroyAllWindows()