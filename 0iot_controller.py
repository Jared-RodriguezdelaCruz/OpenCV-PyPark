import RPi.GPIO as GPIO
import pymongo
import random
import time
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Server will run at: http://169.254.65.154:8000/

# LED pins representing parking spots
LED_PINS = [17, 18, 27, 22, 23, 24, 25, 5, 6, 13]

# H-bridge motor control pins
MOTOR_IN1 = 20
MOTOR_IN2 = 21

# LS (Light sensor) pin for digital reading via RC charge
LS_PIN = 26

# Configure GPIO board settings
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Set up LED pins
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Set up motor control pins
GPIO.setup(MOTOR_IN1, GPIO.OUT)
GPIO.setup(MOTOR_IN2, GPIO.OUT)
GPIO.output(MOTOR_IN1, GPIO.LOW)
GPIO.output(MOTOR_IN2, GPIO.LOW)

# Connect to MongoDB cluster
MONGO_URI = "mongodb+srv://ANotRealName:54321@pypark.3exozxa.mongodb.net/"
client = pymongo.MongoClient(MONGO_URI)
db = client["parking_monitor"]
collection = db["estados"]

# Initialize FastAPI app
app = FastAPI()

# Inserts a default document into the collection if it's empty
def initialize_grouped_db():
    if collection.count_documents({}) == 0:
        document = {
            "timestamp": datetime.utcnow(),
            "totalSpots": len(LED_PINS),
            "availableSpots": len(LED_PINS),
            "spots": [{"index": i, "status": "empty"} for i in range(len(LED_PINS))]
        }
        collection.insert_one(document)
        print("✅ Initial grouped document inserted.")

# Retrieves the latest parking status and sets each LED accordingly
def update_leds_grouped():
    document = collection.find_one(sort=[("timestamp", -1)])
    if not document:
        print("⚠️ No document found.")
        return

    for i, spot in enumerate(document["spots"]):
        gpio_state = GPIO.HIGH if spot["status"] == "empty" else GPIO.LOW
        GPIO.output(LED_PINS[i], gpio_state)

# Simulates random occupied/empty states for each parking spot
def simulate_random_state():
    states = []
    available = 0

    for i in range(len(LED_PINS)):
        status = random.choice(["empty", "occupied"])
        if status == "empty":
            available += 1
        states.append({"index": i, "status": status})

    document = {
        "timestamp": datetime.utcnow(),
        "totalSpots": len(LED_PINS),
        "availableSpots": available,
        "spots": states
    }
    collection.insert_one(document)

# Reads light intensity using digital method based on RC discharge
def read_LS_digital(pin):
    count = 0
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, False)
    time.sleep(0.1)

    GPIO.setup(pin, GPIO.IN)
    while GPIO.input(pin) == GPIO.LOW and count < 10000:
        count += 1
    return count

# Interprets LS value and returns ON/OFF based on threshold
def read_light_status():
    value = read_LS_digital(LS_PIN)
    return "OFF" if value < 1000 else "ON"

# Activates motor in forward direction
def motor_forward():
    GPIO.output(MOTOR_IN1, GPIO.HIGH)
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    return "Motor moving forward"

# Activates motor in reverse direction
def motor_reverse():
    GPIO.output(MOTOR_IN1, GPIO.LOW)
    GPIO.output(MOTOR_IN2, GPIO.HIGH)
    return "Motor in reverse"

# Stops motor
def motor_stop():
    GPIO.output(MOTOR_IN1, GPIO.LOW)
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    return "Motor stopped"

@app.get("/status")
# Returns latest parking spot status from MongoDB
def get_parking_status():
    document = collection.find_one(sort=[("timestamp", -1)])
    if document:
        return {"timestamp": document["timestamp"], "spots": document["spots"]}
    return {"message": "No parking data found"}

@app.get("/motor/forward")
# Triggers forward motion of motor
def trigger_motor_forward():
    return motor_forward()

@app.get("/motor/reverse")
# Triggers reverse motion of motor
def trigger_motor_reverse():
    return motor_reverse()

@app.get("/motor/stop")
# Stops the motor
def trigger_motor_stop():
    return motor_stop()

@app.get("/light")
# Reads current light sensor status
def get_light_status():
    return {"light_status": read_light_status()}

@app.get("/simulate_state")
# Simulates random parking states and stores in DB
def trigger_simulated_state():
    simulate_random_state()
    return {"message": "Simulated state stored."}

if __name__ == "__main__":
    try:
        print("🟢 Starting parking monitoring system...")
        initialize_grouped_db()
        simulate_random_state()
        update_leds_grouped()

        # Runs FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000)

    except KeyboardInterrupt:
        print("\n🟥 Shutdown triggered by user.")

    finally:
        GPIO.cleanup()
        client.close()
        print("✅ Resources released.")