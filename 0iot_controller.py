import RPi.GPIO as GPIO
import pymongo
import random
import time
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# IP: http://169.254.65.154:8000/

# Pines para LEDs de parqueo
LED_PINS = [17, 18, 27, 22, 23, 24, 25, 5, 6, 13]

# Pines para motor con puente H
MOTOR_IN1 = 20
MOTOR_IN2 = 21

# Pin para fotoresistor (lectura digital con tiempo de carga)
LDR_PIN = 26

# Configurar GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Configurar pines LED
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)
    GPIO.output(pin, GPIO.LOW)

# Configurar motor
GPIO.setup(MOTOR_IN1, GPIO.OUT)
GPIO.setup(MOTOR_IN2, GPIO.OUT)
GPIO.output(MOTOR_IN1, GPIO.LOW)
GPIO.output(MOTOR_IN2, GPIO.LOW)

# Conexión MongoDB
MONGO_URI = "mongodb+srv://ANotRealName:54321@pypark.3exozxa.mongodb.net/"
client = pymongo.MongoClient(MONGO_URI)
db = client["parking_monitor"]
collection = db["estados"]

# FastAPI setup
app = FastAPI()

# ---------- FUNCIONES ----------
def initialize_grouped_db():
    if collection.count_documents({}) == 0:
        documento = {
            "timestamp": datetime.utcnow(),
            "totalSpots": len(LED_PINS),
            "availableSpots": len(LED_PINS),
            "spots": [{"index": i, "status": "empty"} for i in range(len(LED_PINS))]
        }
        collection.insert_one(documento)
        print("✅ Documento inicial agrupado insertado.")

def update_leds_grouped():
    documento = collection.find_one(sort=[("timestamp", -1)])
    if not documento:
        print("⚠️ No se encontró ningún documento.")
        return

    for i, spot in enumerate(documento["spots"]):
        estado_gpio = GPIO.HIGH if spot["status"] == "empty" else GPIO.LOW
        GPIO.output(LED_PINS[i], estado_gpio)

def simular_estado_aleatorio():
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

# Lectura del LDR digital (basado en carga RC)
def read_ldr_digital(pin):
    count = 0
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, False)
    time.sleep(0.1)

    GPIO.setup(pin, GPIO.IN)
    while GPIO.input(pin) == GPIO.LOW and count < 10000:
        count += 1
    return count

def leer_luz():
    valor = read_ldr_digital(LDR_PIN)
    return "Luz" if valor < 1000 else "Oscuro"

# Control del motor
def motor_avanzar():
    GPIO.output(MOTOR_IN1, GPIO.HIGH)
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    return "Motor avanzando"

def motor_reversa():
    GPIO.output(MOTOR_IN1, GPIO.LOW)
    GPIO.output(MOTOR_IN2, GPIO.HIGH)
    return "Motor en reversa"

def motor_stop():
    GPIO.output(MOTOR_IN1, GPIO.LOW)
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    return "Motor detenido"

# ---------- RUTAS DE FASTAPI ----------
@app.get("/estado")
def estado_parqueo():
    """Obtiene el estado de los espacios de parqueo"""
    documento = collection.find_one(sort=[("timestamp", -1)])
    if documento:
        return {"timestamp": documento["timestamp"], "spots": documento["spots"]}
    return {"message": "No se encontró estado de parqueo"}

@app.get("/motor/avanzar")
def avanzar_motor():
    return motor_avanzar()

@app.get("/motor/reversa")
def reversa_motor():
    return motor_reversa()

@app.get("/motor/stop")
def detener_motor():
    return motor_stop()

@app.get("/luz")
def estado_luz():
    """Obtiene el estado actual del fotoresistor"""
    return {"estado_luz": leer_luz()}

@app.get("/simular_estado")
def simular_estado():
    """Simula el estado de los espacios de parqueo"""
    simular_estado_aleatorio()
    return {"message": "Estado simulado y almacenado."}

# ---------- BUCLE PRINCIPAL (Solo se ejecuta una vez) ----------
if __name__ == "__main__":
    try:
        print("🟢 Iniciando sistema de monitoreo de parqueo...")
        initialize_grouped_db()

        # Para simulación y control, pero no necesitamos el bucle mientras el servidor FastAPI esté corriendo
        simular_estado_aleatorio()  # Simulación del estado de los espacios de parqueo
        update_leds_grouped()  # Actualización de LEDs según el estado de los espacios

        # Se mantiene el servidor FastAPI corriendo
        uvicorn.run(app, host="0.0.0.0", port=8000)

    except KeyboardInterrupt:
        print("\n🟥 Detenido por el usuario.")

    finally:
        GPIO.cleanup()
        client.close()
        print("✅ Recursos liberados.")
