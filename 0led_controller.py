import RPi.GPIO as GPIO
import pymongo
import random
import time
from datetime import datetime

# Configuración de pines GPIO para los 10 LEDs
LED_PINS = [17, 18, 27, 22, 23, 24, 25, 5, 6, 13]

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)  # Evita advertencias si se reinicia el script
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Conexión a MongoDB Atlas
MONGO_URI = "mongodb+srv://ANotRealName:54321@pypark.3exozxa.mongodb.net/"
client = pymongo.MongoClient(MONGO_URI)

# Selección de base de datos y colección
db = client["parking_monitor"]
collection = db["estados"]

def initialize_grouped_db():
    """Inicializa la colección con un documento agrupado si está vacía"""
    if collection.count_documents({}) == 0:
        documento = {
            "timestamp": datetime.utcnow(),
            "totalSpots": len(LED_PINS),
            "availableSpots": len(LED_PINS),
            "spots": [
                {"index": i, "status": "empty"} for i in range(len(LED_PINS))
            ]
        }
        collection.insert_one(documento)
        print("✅ Documento inicial agrupado insertado.")

def update_leds_grouped():
    """Lee el documento más reciente y actualiza los LEDs según su estado"""
    documento = collection.find_one(sort=[("timestamp", -1)])
    if not documento:
        print("⚠️ No se encontró ningún documento.")
        return

    for i, spot in enumerate(documento["spots"]):
        estado_gpio = GPIO.HIGH if spot["status"] == "empty" else GPIO.LOW
        GPIO.output(LED_PINS[i], estado_gpio)
        print(f"Slot {spot['index']} -> {'🟥 Libre' if estado_gpio == GPIO.HIGH else '🟩 Ocupado'}")

def simular_estado_aleatorio():
    """Simula ocupación aleatoria de los espacios de estacionamiento"""
    estados = []
    libres = 0

    for i in range(len(LED_PINS)):
        status = random.choice(["empty", "occupied"])
        if status == "empty":
            libres += 1
        estados.append({"index": i, "status": status})

    documento = {
        "timestamp": datetime.utcnow(),
        "totalSpots": len(LED_PINS),
        "availableSpots": libres,
        "spots": estados
    }
    collection.insert_one(documento)
    print("🆕 Documento aleatorio insertado.")

# ------------------ PROGRAMA PRINCIPAL ------------------
if __name__ == "__main__":
    try:
        print("🟢 Iniciando sistema de monitoreo de parqueo...")
        initialize_grouped_db()

        while True:
            simular_estado_aleatorio()
            update_leds_grouped()
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n🟥 Deteniendo el sistema por interrupción del usuario.")

    finally:
        GPIO.cleanup()
        client.close()
        print("✅ Recursos liberados correctamente.")

# Motor puente H,fotoresistor (digital), mandar esta info por bluetooth a appinventor