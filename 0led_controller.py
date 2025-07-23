import pymongo
import RPi.GPIO as GPIO
import time
from datetime import datetime

# Configuración de GPIO
LED_PINS = [17, 18, 27, 22, 23, 24, 25, 5, 6, 13]  # Pines GPIO para 10 LEDs
GPIO.setmode(GPIO.BCM)
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Configuración MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["parking_system"]
collection = db["parking_slots"]

def update_leds():
    try:
        # Obtener estados actuales de MongoDB
        slots = list(collection.find().sort("slot_number", 1))
        
        if len(slots) != len(LED_PINS):
            print(f"Error: Se esperaban {len(LED_PINS)} slots pero hay {len(slots)} en DB")
            return
        
        for i, slot in enumerate(slots):
            GPIO.output(LED_PINS[i], GPIO.HIGH if slot["occupied"] else GPIO.LOW)
            print(f"Slot {slot['slot_number']}: {'ON' if slot['occupied'] else 'OFF'}")
            
    except Exception as e:
        print(f"Error al actualizar LEDs: {str(e)}")

def initialize_db():
    """Inicializa la base de datos con 10 slots vacíos"""
    if collection.count_documents({}) == 0:
        for i in range(10):
            collection.insert_one({
                "slot_number": i+1,
                "occupied": False,
                "last_updated": datetime.now()
            })
        print("Base de datos inicializada con 10 slots")

if __name__ == "__main__":
    try:
        print("Iniciando controlador de LEDs...")
        initialize_db()
        
        while True:
            update_leds()
            time.sleep(1)  # Actualizar cada segundo
            
    except KeyboardInterrupt:
        print("\nDeteniendo controlador...")
    finally:
        GPIO.cleanup()
        client.close()