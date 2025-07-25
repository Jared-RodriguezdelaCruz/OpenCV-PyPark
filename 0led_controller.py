import RPi.GPIO as GPIO
import pymongo
import random
import time
from datetime import datetime

# Configuración de GPIO para los 10 LEDs
LED_PINS = [17, 18, 27, 22, 23, 24, 25, 5, 6, 13]
GPIO.setmode(GPIO.BCM)
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Conexión a MongoDB local
client = MongoClient("mongodb+srv://ANotRealName:54321@pypark.3exozxa.mongodb.net/")
db = client.db("parking_monitor");
collection = db.collection("estados");

def initialize_grouped_db():
    """Inicializa la colección con un documento agrupado si está vacía"""
    if collection.count_documents({}) == 0:
        documento = {
            "timestamp": datetime.utcnow(),
            "totalSpots": len(LED_PINS),
            "availableSpots": len(LED_PINS),  # Todos desocupados al iniciar
            "spots": [
                {"index": i, "status": "empty"} for i in range(len(LED_PINS))
            ]
        }
        collection.insert_one(documento)
        print("Documento inicial agrupado insertado.")

def update_leds_grouped():
    """Lee el documento más reciente y actualiza los LEDs en base a los estados"""
    documento = collection.find_one(sort=[("timestamp", -1)])
    if not documento:
        print("No se encontró ningún documento.")
        return

    for i, spot in enumerate(documento["spots"]):
        estado = GPIO.HIGH if spot["status"] == "occupied" else GPIO.LOW
        GPIO.output(LED_PINS[i], estado)
        print(f"Slot {spot['index']}: {'ON' if estado == GPIO.HIGH else 'OFF'}")
    
def simular_estado_aleatorio():
    """Simula ocupación aleatoria de cada slot e inserta nuevo documento"""
    estados = []
    libres = 0

    for i in range(len(LED_PINS)):
        estado = random.choice(["empty", "occupied"])
        if estado == "empty":
            libres += 1
        estados.append({"index": i, "status": estado})

    documento = {
        "timestamp": datetime.utcnow(),
        "totalSpots": len(LED_PINS),
        "availableSpots": libres,
        "spots": estados
    }
    collection.insert_one(documento)
    print("Documento aleatorio insertado.")

if __name__ == "__main__":
    try:
        print("🟢 Iniciando controlador de LEDs agrupado...")
        initialize_grouped_db()
        # Puedes descomentar esta línea si quieres simular la ocupación:
        # simular_ocupacion()

        while True:
            simular_estado_aleatorio()
            update_leds_grouped()
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n🟥 Deteniendo controlador...")
    finally:
        GPIO.cleanup()
        client.close()