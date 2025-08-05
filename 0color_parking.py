# ======== LIBRARY IMPORTS ========
import cv2                     # OpenCV for image and video processing
import numpy as np             # NumPy for numerical operations, especially image matrix manipulations
from collections import deque  # deque enables fixed-length queues for tracking slot history (smoothing logic)
from pymongo import MongoClient
import datetime
import time

# Connect to MongoDB cluster
MONGO_URI = "mongodb+srv://ANotRealName:54321@pypark.3exozxa.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["parking_monitor"]
collection = db["estados"]

# ======== CONFIGURATION ========
CAMERA_URL = "http://192.168.61.120:4747/video"  # URL for IP camera or phone stream (e.g., DroidCam)
MIN_AREA = 500         # Minimum contour area to consider movement as significant (likely a car)
HISTORY_LENGTH = 5     # Number of previous frames used to smooth occupancy detection
LOG_INTERVAL = 5

# ======== GLOBAL VARIABLES ========
drawing = False        # True when user is dragging the mouse to draw a parking slot
start_point = None     # Stores the initial (x, y) position of a drawn rectangle
slots = []             # List of tuples containing parking slot coordinates [(pt1, pt2), ...]
background = None      # Background image used to compare against live frames
slot_states = []       # Current status of each slot: True if occupied, False otherwise
slot_history = []      # History queue of each slot to perform stable, averaged detection

# ======== CAMERA INITIALIZATION FUNCTION ========
def init_camera():
    """Attempts to connect to the specified camera URL. Exits the program if connection fails."""
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("❌ Error connecting to camera")
        exit()
    return cap

# ======== BACKGROUND CALIBRATION FUNCTION ========
def calibrate_background(cap, num_frames=30):
    """
    Captures multiple frames to build an averaged background image.
    This helps reduce noise and improves object detection accuracy.
    """
    print("Calibrating background...")
    avg_frame = None

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert frame to grayscale and blur it to smooth out edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize or update background using weighted accumulation
        if avg_frame is None:
            avg_frame = gray.copy().astype("float")
        else:
            cv2.accumulateWeighted(gray, avg_frame, 0.5)

    # Finalize background frame and return it
    return cv2.convertScaleAbs(avg_frame) # type: ignore

# ======== MOUSE CALLBACK FUNCTION FOR DRAWING SLOTS ========
def mouse_callback(event, x, y, flags, param):
    """
    Allows the user to draw rectangles over the frozen video frame by clicking and dragging.
    Each rectangle defines a monitored parking slot.
    """
    global drawing, start_point, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:      # Mouse press: start drawing
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:      # Mouse release: finish drawing
        drawing = False
        end_point = (x, y)

        # Store slot coordinates and initialize state and history
        slots.append((start_point, end_point))
        slot_states.append(False)
        slot_history.append(deque([False]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH))

        # Draw visual feedback rectangle on the frozen frame
        cv2.rectangle(frame_copy, start_point, end_point, (0, 255, 0), 2) # type: ignore

# ======== OCCUPANCY DETECTION FUNCTION ========
def detect_occupation(roi, background_roi):
    """
    Detects whether there's significant motion (likely a car) in the given region by comparing it to the background.
    Steps:
      - Convert to grayscale
      - Apply blur
      - Compute absolute difference
      - Threshold and clean using morphology
      - Find contours and check area
    Returns True if motion (occupancy) is detected.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(background_roi, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    # Detect contours (connected white regions)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > MIN_AREA:
            return True  # occupied
    return False         # Empty

def log_slot_change(slot_states):
    """Inserts a document with the full current state of all slots."""
    doc = {
        "timestamp": datetime.datetime.utcnow(),
        "totalSpots": len(slot_states),
        "availableSpots": slot_states.count(False),
        "spots": [{"index": i, "status": "occupied" if state else "free"} for i, state in enumerate(slot_states)]
    }
    collection.insert_one(doc)
    print("📄 Full slot state inserted.")

# ======== MAIN MONITORING FUNCTION ========
def main():
    """
    Full workflow:
      - Connect to camera
      - Show preview until user confirms frame
      - Let user draw parking slots
      - Calibrate background
      - Continuously analyze frames and show slot occupancy visually
    """
    global frame_copy, background
    cap = init_camera()

    # === PREVIEW LIVE VIDEO FOR SLOT DRAWING ===
    print("Previewing area... Press 'c' to freeze and draw parking slots")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Live Preview ('c' to capture)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == 13:  # Confirm to freeze frame
            frame_copy = frame.copy()
            break
        elif key == ord('q'):            # Quit preview
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Live Preview ('c' to capture)")

    # === DRAW PARKING SLOTS OVER STATIC FRAME ===
    cv2.namedWindow("Draw Slots ('c' to continue)")
    cv2.setMouseCallback("Draw Slots ('c' to continue)", mouse_callback)

    while True:
        cv2.imshow("Draw Slots ('c' to continue)", frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == 13:  # Proceed to calibration
            break
        elif key == ord('q'):            # Quit slot drawing
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Draw Slots ('c' to continue)")

    if not slots:
        print("⚠️ No parking slots were defined.")
        cap.release()
        return

    # === BACKGROUND CALIBRATION (NO CARS IN SLOTS) ===
    print("Make sure all parking slots are empty during calibration...")
    background = calibrate_background(cap)
    
    previous_states = slot_states.copy()
    last_logged = time.time()

    # === START OCCUPANCY MONITORING LOOP ===
    print("Monitoring started... (Press 'q' to exit)")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        # Analyze each parking slot
        for idx, (pt1, pt2) in enumerate(slots):
            x1, y1 = pt1
            x2, y2 = pt2
            if x1 >= x2 or y1 >= y2:
                continue

            roi = frame[y1:y2, x1:x2]           # Extract current region from frame
            bg_roi = background[y1:y2, x1:x2]   # Extract corresponding region from background

            if roi.size == 0 or bg_roi.size == 0:
                continue

            # Determine if the slot is occupied
            occupied = detect_occupation(roi, bg_roi)

            # Smooth result using history buffer
            slot_history[idx].append(occupied)
            slot_states[idx] = sum(slot_history[idx]) / HISTORY_LENGTH > 0.6

            # Draw rectangle and status text
            color = (0, 0, 255) if slot_states[idx] else (0, 255, 0)  # Red = occupied, Green = free
            cv2.rectangle(frame, pt1, pt2, color, 2)
            status = "occupied" if slot_states[idx] else "free"
            cv2.putText(frame, status, (pt1[0], pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
        current_time = time.time()
        if current_time - last_logged >= LOG_INTERVAL:
            if slot_states != previous_states:
                log_slot_change(slot_states)
                previous_states = slot_states.copy()
            last_logged = current_time

        # Show updated frame with status
        cv2.imshow("Parking Status", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ======== SCRIPT ENTRYPOINT ========
if __name__ == "__main__":
    main()