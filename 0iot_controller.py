import RPi.GPIO as GPIO
import pymongo
import random
import time
import uvicorn
import threading
from datetime import datetime
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Server will run at: http://192.168.63.8:8000/

# ======== HARDWARE CONFIGURATION ========
# LED pins representing parking spots
LED_PINS = [18, 23, 24, 25, 8, 7, 12, 16, 20, 21]

# Buzzer pin for audio feedback
BUZZER_PIN = 19

# Ultrasonic sensor pins for distance measurement
ECHO_PIN = 13
TRIGGER_PIN = 6

# H-bridge motor control pins
MOTOR_PIN1 = 22
MOTOR_PIN2 = 27

# ======== GPIO INITIALIZATION ========
GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
GPIO.setwarnings(False)  # Disable GPIO warnings

# Initialize LED pins as outputs
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)  # Start with all LEDs off

# Initialize buzzer pin
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Initialize ultrasonic sensor pins
GPIO.setup(TRIGGER_PIN, GPIO.OUT)  # Trigger = output
GPIO.setup(ECHO_PIN, GPIO.IN)      # Echo = input

# Initialize motor control pins
GPIO.setup(MOTOR_PIN1, GPIO.OUT)
GPIO.setup(MOTOR_PIN2, GPIO.OUT)
GPIO.output(MOTOR_PIN1, GPIO.LOW)  # Start with motor off
GPIO.output(MOTOR_PIN2, GPIO.LOW)

# ======== DATABASE CONNECTION ========
MONGO_URI = "mongodb+srv://ANotRealName:54321@pypark.3exozxa.mongodb.net/"
client = pymongo.MongoClient(MONGO_URI)
db = client["parking_monitor"]
collection = db["estados"]

# ======== FASTAPI CONFIGURATION ========
app = FastAPI()

# Configure CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins
    allow_credentials=True,       # Support credentials
    allow_methods=["POST", "GET"],# Allowed HTTP methods
    allow_headers=["*"],          # Allowed headers
)

class LightData(BaseModel):
    light: int

# Global variable to store the fotoresistor value
last_light_value = None

motor_state = "stop"  # Values: "forward", "reverse", "stop"

# ======== DATABASE INITIALIZATION ========
def initialize_grouped_db():
    """Creates initial document in MongoDB if collection is free"""
    if collection.count_documents({}) == 0:
        document = {
            "timestamp": datetime.utcnow(),
            "totalSpots": len(LED_PINS),
            "availableSpots": len(LED_PINS),
            "spots": [{"index": i, "status": "free"} for i in range(len(LED_PINS))]
        }
        collection.insert_one(document)
        print("✅ Initial grouped document inserted.")

# ======== PARKING STATUS FUNCTIONS ========
def update_leds_grouped():
    """Updates LED states based on latest parking status from database"""
    document = collection.find_one(sort=[("timestamp", -1)])
    if not document:
        print("⚠️ No document found.")
        return

    for i, spot in enumerate(document["spots"]):
        # Turn LED on if spot is free, off if occupied
        gpio_state = GPIO.HIGH if spot["status"] == "free" else GPIO.LOW
        GPIO.output(LED_PINS[i], gpio_state)
        
        # Sound buzzer if first spot is occupied
        if i == 0 and spot["status"] == "occupied":
            activate_buzzer()   

# Simulates random occupied/free states for each parking spot
def simulate_random_state():
    states = []
    available = 0

    for i in range(len(LED_PINS)):
        status = random.choice(["free", "occupied"])
        if status == "free":
            available += 1
        states.append({"index": i, "status": status})

    document = {
        "timestamp": datetime.utcnow(),
        "totalSpots": len(LED_PINS),
        "availableSpots": available,
        "spots": states
    }
    collection.insert_one(document)
    
# ======== SENSOR FUNCTIONS ========
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
    
# Reproduces a sound from the buzzer
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
    
    
def motor_control_loop():
    global motor_state
    while True:
        if motor_state == "forward":
            GPIO.output(MOTOR_PIN1, GPIO.HIGH)
            GPIO.output(MOTOR_PIN2, GPIO.LOW)
        elif motor_state == "reverse":
            GPIO.output(MOTOR_PIN1, GPIO.LOW)
            GPIO.output(MOTOR_PIN2, GPIO.HIGH)
        else:  # "stop"
            GPIO.output(MOTOR_PIN1, GPIO.LOW)
            GPIO.output(MOTOR_PIN2, GPIO.LOW)
        time.sleep(5)  # Refresh interval

# ======== API ENDPOINTS ========
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

@app.post("/light_data")
# Receives and stores light sensor data from the frontend
async def receive_light_data(data: LightData):
    global last_light_value
    last_light_value = data.light  # Save the received light value globally

@app.get("/fotoresistor")
# Returns the latest photoresistor (light) value if available
def get_fotoresistor():
    global last_light_value

    if last_light_value is not None:
        return f"Light: {last_light_value}%"
    
    # Error response if no light data has been received yet
    return {"status": "error", "message": "No data available"}

@app.get("/ultrasonic")
# Activates the ultrasonic sensor and returns measured distance
def get_distance():
    return f"Distance: {ultrasonic_sensor()}cm"

@app.get("/motor/set/{direction}")
# Sets the motor direction to forward, reverse, or stop
def set_motor_direction(direction: str):
    global motor_state
    motor_state = direction  # Update global motor direction state
    return {"message": f"Motor state set to '{direction}'"}

@app.get("/led_dance")
# Runs a playful light sequence with the LEDs
def led_dance():
    # Step 1: Turn off all LEDs
    for pin in LED_PINS:
        GPIO.output(pin, GPIO.LOW)
        time.sleep(0.1)
    
    # Step 2: Turn on LEDs in ascending order
    for pin in LED_PINS:
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.1)

    # Step 3: Turn off LEDs in reverse order
    for pin in reversed(LED_PINS):
        GPIO.output(pin, GPIO.LOW)
        time.sleep(0.1)

    # Step 4: Blink all LEDs together multiple times
    for _ in range(3):
        for pin in LED_PINS:
            GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.1)
        for pin in LED_PINS:
            GPIO.output(pin, GPIO.LOW)
        time.sleep(0.1)

    # Step 5: Restore LED states based on parking data
    update_leds_grouped()

    return {"message": "💃 LED dance completed and restored"}

@app.get("/simulate_state")
# Simulates and stores random parking spot availability
def trigger_simulated_state():
    simulate_random_state()       # Generate new random states
    update_leds_grouped()         # Reflect changes via LEDs
    return {"message": "Simulated state stored."}

def periodic_update():
    # Background loop that periodically simulates new parking data
    while True:
        time.sleep(3)             # Wait for 3 seconds between updates
        simulate_random_state()   # Generate new random states
        update_leds_grouped()     # Update LEDs to reflect current state
        
# ======== MAIN EXECUTION ========
if __name__ == "__main__":
    try:
        print("🟢 Starting parking monitoring system...")
        initialize_grouped_db()  # Initialize DB with base parking structure if needed

        # Start parking update simulation in a background thread
        update_thread = threading.Thread(target=periodic_update)
        update_thread.daemon = True
        update_thread.start()

        # Start motor control logic in a background thread
        motor_thread = threading.Thread(target=motor_control_loop)
        motor_thread.daemon = True
        motor_thread.start()

        # Launch FastAPI server to expose endpoints
        uvicorn.run(app, host="0.0.0.0", port=8000)

    except KeyboardInterrupt:
        # Graceful shutdown triggered manually
        print("\n🟥 Shutdown triggered by user.")
    finally:
        GPIO.cleanup()     # Clean up GPIO settings
        client.close()     # Close MongoDB connection
        print("✅ Resources released.")  # Log successful resource cleanup