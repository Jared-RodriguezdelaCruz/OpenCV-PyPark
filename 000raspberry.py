# pi_led_listener.py

import RPi.GPIO as GPIO
import pymongo
import time

# Pines físicos
LED_PINS = [17, 18, 27, 22, 23, 24, 25, 5, 6, 13]

GPIO.setmode(GPIO.BCM)
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Conectar a MongoDB
client = pymongo.MongoClient("mongodb+srv://ANotRealName:54321@pypark.3exozxa.mongodb.net/")
db = client["parking_monitor"]
collection = db["estados"]

def update_leds_grouped():
    doc = collection.find_one(sort=[("timestamp", -1)])
    if not doc: print("⚠️ No hay documento reciente"); return

    for i, slot in enumerate(doc["spots"]):
        estado = GPIO.HIGH if slot["status"] == "occupied" else GPIO.LOW
        GPIO.output(LED_PINS[i], estado)
        print(f"LED {i} → {'ON' if estado == GPIO.HIGH else 'OFF'}")

try:
    print("🟢 Escuchando cambios en MongoDB...")
    while True:
        update_leds_grouped()
        time.sleep(3)
except KeyboardInterrupt:
    print("🛑 Interrupción manual")
finally:
    GPIO.cleanup(); client.close()