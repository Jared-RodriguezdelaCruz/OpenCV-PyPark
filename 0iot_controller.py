import RPi.GPIO as GPIO
import pymongo
import random
import time
import uvicorn
import threading
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Server will run at: http://169.254.65.154:8000/

# LED pins representing parking spots
LED_PINS = [18, 23, 24, 25, 8, 7, 12, 16, 20, 21]

# Fotoresistor pin for digital reading of light
FOTORESISTOR_PIN = 26

# BUZZER pin for reproducing a little sound
BUZZER_PIN = 19

# ULTRA SONIC SENSOR  pins for measuring the distance
ECHO_PIN = 13
TRIGGER_PIN = 6

# H-bridge motor control pins
MOTOR_PIN1 = 22
MOTOR_PIN2 = 27

# Configure GPIO board settings
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Set up LED pins
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# Set up buzzer pin
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Set up ultrasonic sensor pins
GPIO.setup(TRIGGER_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

# Set up motor control pins
GPIO.setup(MOTOR_PIN1, GPIO.OUT)
GPIO.setup(MOTOR_PIN2, GPIO.OUT)
GPIO.output(MOTOR_PIN1, GPIO.LOW)
GPIO.output(MOTOR_PIN2, GPIO.LOW)

# Connect to MongoDB cluster
MONGO_URI = "mongodb+srv://ANotRealName:54321@pypark.3exozxa.mongodb.net/"
client = pymongo.MongoClient(MONGO_URI)
db = client["parking_monitor"]
collection = db["estados"]

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        # Activate buzzer if slot 1 is occupied
        if i == 0 and spot["status"] == "occupied":
            activate_buzzer()       

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

# Interprets LS value and returns ON/OFF based on threshold
def fotoresistor():
    count = 0

    # Descarga el pin
    GPIO.setup(FOTORESISTOR_PIN, GPIO.OUT)
    GPIO.output(FOTORESISTOR_PIN, GPIO.LOW)
    time.sleep(0.2)

    # Empieza a contar
    GPIO.setup(FOTORESISTOR_PIN, GPIO.IN)
    while GPIO.input(FOTORESISTOR_PIN) == GPIO.LOW:
        count += 1

    return count

# Activates motor in forward direction
def motor_forward():
    GPIO.output(MOTOR_PIN1, GPIO.HIGH)
    GPIO.output(MOTOR_PIN2, GPIO.LOW)
    time.sleep(3)
    return "Motor moving forward"

# Activates motor in reverse direction
def motor_reverse():
    GPIO.output(MOTOR_PIN1, GPIO.LOW)
    GPIO.output(MOTOR_PIN2, GPIO.HIGH)
    time.sleep(3)
    return "Motor in reverse"

# Stops motor
def motor_stop():
    GPIO.output(MOTOR_PIN1, GPIO.LOW)
    GPIO.output(MOTOR_PIN2, GPIO.LOW)
    return "Motor stopped"

# Ultra sonicsensor
def ultrasonic_sensor():
    # Send  ultrasonic pulse
    GPIO.output(TRIGGER_PIN, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIGGER_PIN, GPIO.LOW)
    
    timeout_start = time.time() 
    # Wait until Echo activates
    while GPIO.input(ECHO_PIN) == 0:
        if time.time() - timeout_start > 1:
            print("Timeout esperando el inicio de echo")
            return -1
        initial_pulse = time.time()
    timeout_start = time.time() 
    while GPIO.input(ECHO_PIN) == 1:
        if time.time() - timeout_start > 1:
            print("Timeout esperando el fin de echo")
            return -1
        final_pulse = time.time()
    # Calculate distance (cm)
    duracion = final_pulse - initial_pulse
    distance = (duracion * 34300) / 2  # Sounds speed (343 m/s)
    
    return round(distance, 2)
    
def beep(duration=0.1):
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(0.08)
    
# Activates the buzzer for a specified duration
def activate_buzzer():
    beep(0.1)
    beep(0.1)
    time.sleep(0.2)
    beep(0.08)
    beep(0.08)
    beep(0.08)
    time.sleep(0.2)
    beep(0.1)
    beep(0.1)
    
@app.get("/status")
# Returns latest parking spot status from MongoDB
def get_parking_status():
    document = collection.find_one(sort=[("timestamp", -1)])
    if document:
        return {"timestamp": document["timestamp"], "spots": document["spots"]}
    return {"message": "No parking data found"}

@app.get("/buzzer")
# Triggers a buzzing sound
def get_buzzer():
    activate_buzzer()
    return {"message": "Buzzing"}

@app.get("/ultrasonic")
# Triggers the ultrasonic sensor
def get_distance():
    return {"Distance": ultrasonic_sensor()}
        
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

@app.get("/fotoresistor")
# Reads current light sensor status
def get_fotoresistor():
    light = fotoresistor()
    
    if light < 300:
        return {"light_status": f"Too dark ({light})"}
    elif light < 900:
        return {"light_status": f"Average ilumination ({light})"}
    else:
        return {"light_status": f"Too much brightness! ({light})"}
        
@app.get("/led_dance")
def led_dance():
    for pin in LED_PINS:
        GPIO.output(pin, GPIO.LOW)
        time.sleep(0.1)
    
    for pin in LED_PINS:
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.1)

    for pin in reversed(LED_PINS):
        GPIO.output(pin, GPIO.LOW)
        time.sleep(0.1)

    for _ in range(3):
        for pin in LED_PINS:
            GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.1)
        for pin in LED_PINS:
            GPIO.output(pin, GPIO.LOW)
        time.sleep(0.1)

    update_leds_grouped()

    return {"message": "💃 LED dance completed and restored"}

@app.get("/sensor_data")
def get_sensor_data():    
    return {
        "light": fotoresistor(),
        "distance": ultrasonic_sensor()
    }
        
@app.get("/simulate_state")
# Simulates random parking states and stores in DB
def trigger_simulated_state():
    simulate_random_state()
    update_leds_grouped()
    return {"message": "Simulated state stored."}

def periodic_update():
    while True:
        time.sleep(3)  # Simulate every 3 seconds
        simulate_random_state()
        update_leds_grouped()

if __name__ == "__main__":
    try:
        print("🟢 Starting parking monitoring system...")
        initialize_grouped_db()
        
        # Start the periodic update in a separate thread
        update_thread = threading.Thread(target=periodic_update)
        update_thread.daemon = True
        update_thread.start()

        # Runs FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000)

    except KeyboardInterrupt:
        print("\n🟥 Shutdown triggered by user.")

    finally:
        GPIO.cleanup()
        client.close()
        print("✅ Resources released.")