import cv2
import numpy as np
from collections import deque
import RPi.GPIO as GPIO
import time
from datetime import datetime
import pymongo
from pymongo import MongoClient

class ParkingSystem:
    def __init__(self):
        # ===== CONFIGURACIÓN DE HARDWARE =====
        self.LED_PINS = [17, 18, 27, 22, 23, 24, 25, 5, 6, 13]  # Pines GPIO
        self.setup_gpio()
        
        # ===== CONFIGURACIÓN DE VISIÓN POR COMPUTADORA =====
        self.CAMERA_URL = "http://192.168.13.251:4747/video"
        self.MIN_AREA = 500
        self.HISTORY_LENGTH = 5
        self.cap = self.init_camera()
        
        # ===== CONFIGURACIÓN MONGODB =====
        self.mongo_client = MongoClient("mongodb://localhost:27017/")
        self.db = self.mongo_client["smart_parking_db"]
        self.slots_collection = self.db["parking_slots"]
        self.events_collection = self.db["parking_events"]
        self.initialize_database()
        
        # ===== VARIABLES DEL SISTEMA =====
        self.slots = self.load_slots_from_db()  # Carga slots desde MongoDB
        self.slot_states = [False] * len(self.slots)
        self.slot_history = [deque([False]*self.HISTORY_LENGTH, 
                             maxlen=self.HISTORY_LENGTH) for _ in self.slots]
        
    def setup_gpio(self):
        """Configura los pines GPIO para los LEDs"""
        GPIO.setmode(GPIO.BCM)
        for pin in self.LED_PINS:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
    
    def init_camera(self):
        """Inicializa la conexión con la cámara"""
        cap = cv2.VideoCapture(self.CAMERA_URL)
        if not cap.isOpened():
            print("❌ Error al conectar con la cámara")
            self.cleanup()
            exit()
        return cap
    
    def initialize_database(self):
        """Inicializa la base de datos si está vacía"""
        if self.slots_collection.count_documents({}) == 0:
            print("Inicializando base de datos...")
            for i in range(10):  # 10 slots por defecto
                self.slots_collection.insert_one({
                    "slot_id": i+1,
                    "coordinates": None,  # Se actualizarán al dibujar
                    "occupied": False,
                    "last_updated": datetime.now()
                })
    
    def load_slots_from_db(self):
        """Carga las coordenadas de los slots desde MongoDB"""
        slots = []
        for slot in self.slots_collection.find().sort("slot_id", 1):
            if slot.get("coordinates"):
                slots.append(slot["coordinates"])
            else:
                # Si no hay coordenadas, usar valores por defecto (se actualizarán después)
                slots.append(((100 + 120*slot["slot_id"]-1), 100, 
                             (200 + 120*slot["slot_id"]-1), 200))
        return slots
    
    def save_slots_to_db(self):
        """Guarda las coordenadas actuales en MongoDB"""
        for idx, coords in enumerate(self.slots):
            self.slots_collection.update_one(
                {"slot_id": idx+1},
                {"$set": {
                    "coordinates": coords,
                    "last_updated": datetime.now()
                }}
            )
    
    def log_event(self, slot_id, event_type):
        """Registra un evento en la base de datos"""
        self.events_collection.insert_one({
            "slot_id": slot_id,
            "event_type": event_type,  # "occupied" o "freed"
            "timestamp": datetime.now()
        })
    
    def calibrate_background(self, num_frames=30):
        """Calibra el fondo para la detección de movimiento"""
        print("Calibrando fondo...")
        avg_frame = None
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if avg_frame is None:
                avg_frame = gray.copy().astype("float")
            else:
                cv2.accumulateWeighted(gray, avg_frame, 0.5)
        return cv2.convertScaleAbs(avg_frame)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para dibujar los espacios con el mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)
            slot_id = len(self.slots) + 1
            self.slots.append((self.start_point, end_point))
            self.slot_states.append(False)
            self.slot_history.append(deque([False]*self.HISTORY_LENGTH, 
                                         maxlen=self.HISTORY_LENGTH))
            
            # Guardar en MongoDB
            self.slots_collection.update_one(
                {"slot_id": slot_id},
                {"$set": {
                    "coordinates": (self.start_point, end_point),
                    "last_updated": datetime.now()
                }}
            )
            
            cv2.rectangle(self.frame_copy, self.start_point, end_point, (0, 255, 0), 2)
    
    def detect_occupation(self, roi, background_roi):
        """Detecta si un espacio está ocupado"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(background_roi, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > self.MIN_AREA:
                return True
        return False
    
    def update_system_state(self):
        """Actualiza LEDs y base de datos según los estados actuales"""
        for idx, state in enumerate(self.slot_states):
            if idx >= len(self.LED_PINS):
                continue
                
            # Actualizar LED
            GPIO.output(self.LED_PINS[idx], GPIO.HIGH if state else GPIO.LOW)
            
            # Actualizar MongoDB
            previous_state = self.slots_collection.find_one(
                {"slot_id": idx+1})["occupied"]
            
            if state != previous_state:
                self.slots_collection.update_one(
                    {"slot_id": idx+1},
                    {"$set": {
                        "occupied": state,
                        "last_updated": datetime.now()
                    }}
                )
                # Registrar evento
                event_type = "occupied" if state else "freed"
                self.log_event(idx+1, event_type)
                print(f"Slot {idx+1} {event_type} at {datetime.now()}")
    
    def run(self):
        """Ejecuta el sistema principal"""
        print("Sistema de Estacionamiento Inteligente - Inicializando...")
        
        # Paso 1: Configuración de slots
        if not self.slots or any(coord is None for coord in self.slots):
            print("Configuración de slots requerida...")
            self.configure_slots()
        
        # Paso 2: Calibración
        print("Calibrando fondo...")
        self.background = self.calibrate_background()
        
        # Paso 3: Bucle principal
        print("Iniciando monitoreo... (Presiona 'q' para salir)")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error al capturar frame")
                    time.sleep(1)
                    continue

                self.process_frame(frame)
                self.update_system_state()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nDeteniendo sistema...")
        finally:
            self.cleanup()
    
    def configure_slots(self):
        """Interfaz para configurar los slots con el mouse"""
        print("Por favor, dibuja los espacios de estacionamiento...")
        ret, frame = self.cap.read()
        if not ret:
            print("Error al capturar frame para configuración")
            self.cleanup()
            exit()
            
        self.frame_copy = frame.copy()
        cv2.namedWindow("Dibuja los slots (Presiona 'c' para continuar)")
        cv2.setMouseCallback("Dibuja los slots (Presiona 'c' para continuar)", 
                           self.mouse_callback)

        while True:
            cv2.imshow("Dibuja los slots (Presiona 'c' para continuar)", self.frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') or key == 13:  # 13 es Enter
                break
            elif key == ord('q'):
                self.cleanup()
                exit()

        cv2.destroyWindow("Dibuja los slots (Presiona 'c' para continuar)")
        self.save_slots_to_db()
    
    def process_frame(self, frame):
        """Procesa un frame para detectar ocupación"""
        for idx, (pt1, pt2) in enumerate(self.slots):
            x1, y1 = pt1
            x2, y2 = pt2
            if x1 >= x2 or y1 >= y2:
                continue
                
            roi = frame[y1:y2, x1:x2]
            bg_roi = self.background[y1:y2, x1:x2]
            
            if roi.size == 0 or bg_roi.size == 0:
                continue
            
            # Detección de ocupación
            occupied = self.detect_occupation(roi, bg_roi)
            self.slot_history[idx].append(occupied)
            new_state = sum(self.slot_history[idx]) / self.HISTORY_LENGTH > 0.6
            
            # Actualizar estado si cambió
            if new_state != self.slot_states[idx]:
                self.slot_states[idx] = new_state
            
            # Visualización
            color = (0, 0, 255) if self.slot_states[idx] else (0, 255, 0)
            cv2.rectangle(frame, pt1, pt2, color, 2)
            status = "Ocupado" if self.slot_states[idx] else "Libre"
            cv2.putText(frame, f"Slot {idx+1}: {status}", 
                       (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255,255,255), 2)
        
        cv2.imshow("Estado del estacionamiento", frame)
    
    def cleanup(self):
        """Libera recursos al terminar"""
        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        self.mongo_client.close()
        print("Sistema detenido correctamente")

if __name__ == "__main__":
    system = ParkingSystem()
    system.run()