# ======== LIBRARIES IMPORTED ========
import cv2                      # OpenCV for image and video frame processing
import numpy as np              # NumPy for efficient numerical operations
from collections import deque   # deque provides fast sliding windows for history tracking

# ======== CONFIGURATION VARIABLES ========
CAMERA_URL = "http://192.168.62.244:4747/video"  # Camera stream URL (can be phone or webcam over IP)
MIN_CORRELATION = 0.7     # Histogram correlation threshold: lower means more difference (indicates vehicle presence)
HISTORY_LENGTH = 5        # Number of frames stored to smooth detection results and prevent flickering

# ======== GLOBAL STATE ========
drawing = False                # Flag to indicate whether user is currently drawing a parking slot
start_point = None             # Stores the first mouse click position
slots = []                     # List of parking slot rectangles [(pt1, pt2), ...]
background = None              # Averaged empty background used for comparison
slot_states = []               # Current occupied/free state for each slot (True = occupied)
slot_history = []              # A deque (sliding window) for each slot, storing recent detection results

# ======== CAMERA INITIALIZATION FUNCTION ========
def init_camera():
    """
    Attempts to connect to the camera stream. If unsuccessful, exits the script.
    """
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("❌ Failed to connect to camera")
        exit()
    return cap

# ======== BACKGROUND CALIBRATION FUNCTION ========
def calibrate_background(cap, num_frames=30):
    """
    Builds a clean average background by capturing multiple frames while parking spots are empty.
    This background will be used to detect occupancy changes later.
    """
    print("Calibrating background...")
    avg_frame = None

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)             # Convert to grayscale
        gray = cv2.GaussianBlur(gray, (21, 21), 0)                 # Apply blur to reduce noise

        if avg_frame is None:
            avg_frame = gray.astype("float")                       # First frame becomes base
        else:
            cv2.accumulateWeighted(gray, avg_frame, 0.5)           # Smoothly accumulate frame

    return cv2.convertScaleAbs(avg_frame)                          # type: ignore # Finalize accumulated image

# ======== MOUSE DRAWING FUNCTION ========
def mouse_callback(event, x, y, flags, param):
    """
    Handles user interaction for drawing rectangles representing parking slots.
    Stores rectangle points and initializes state tracking variables.
    """
    global drawing, start_point, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

        # Save the slot and initialize its status/history
        slots.append((start_point, end_point))
        slot_states.append(False)
        slot_history.append(deque([False]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH))

        # Visually draw the slot on screen
        cv2.rectangle(frame_copy, start_point, end_point, (0, 255, 0), 2) # type: ignore

# ======== HISTOGRAM-BASED OCCUPANCY DETECTION ========
def detect_occupation_histogram(roi, background_roi):
    """
    Compares histograms of current slot region and its corresponding background region.
    A large deviation between histograms implies the presence of a vehicle.
    """
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(background_roi, cv2.COLOR_BGR2GRAY)

    roi_hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
    bg_hist = cv2.calcHist([bg_gray], [0], None, [256], [0, 256])

    cv2.normalize(roi_hist, roi_hist)
    cv2.normalize(bg_hist, bg_hist)

    correlation = cv2.compareHist(roi_hist, bg_hist, cv2.HISTCMP_CORREL)

    # Lower correlation indicates stronger difference (likely vehicle present)
    return correlation < MIN_CORRELATION

# ======== MAIN EXECUTION FUNCTION ========
def main():
    """
    Orchestrates the full workflow:
      - Connects to camera
      - Shows live preview for slot drawing
      - Calibrates background
      - Continuously monitors parking status
    """
    global frame_copy, background

    cap = init_camera()

    # === LIVE PREVIEW BEFORE SLOT SETUP ===
    print("Visualizing area... Press 'c' to freeze frame and draw parking slots")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Live Preview ('c' to capture)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == 13:
            frame_copy = frame.copy()
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Live Preview ('c' to capture)")

    # === SLOT DRAWING INTERFACE ===
    cv2.namedWindow("Draw Parking Slots ('c' to continue)")
    cv2.setMouseCallback("Draw Parking Slots ('c' to continue)", mouse_callback)

    while True:
        cv2.imshow("Draw Parking Slots ('c' to continue)", frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == 13:
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Draw Parking Slots ('c' to continue)")

    if not slots:
        print("⚠️ No parking slots were defined")
        cap.release()
        return

    # === BACKGROUND CALIBRATION ===
    print("Make sure all slots are empty during calibration...")
    background = calibrate_background(cap)

    # === START PARKING MONITORING LOOP ===
    print("Monitoring started... (Press 'q' to quit)")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        for idx, (pt1, pt2) in enumerate(slots):
            x1, y1 = pt1
            x2, y2 = pt2
            if x1 >= x2 or y1 >= y2:
                continue

            roi = frame[y1:y2, x1:x2]
            bg_roi = background[y1:y2, x1:x2]

            if roi.size == 0 or bg_roi.size == 0:
                continue

            # Detect occupancy using histogram comparison
            occupied = detect_occupation_histogram(roi, bg_roi)

            # Update history buffer and smooth output
            slot_history[idx].append(occupied)
            slot_states[idx] = slot_history[idx].count(True) >= 3

            # Draw current detection result
            color = (0, 0, 255) if slot_states[idx] else (0, 255, 0)
            cv2.rectangle(frame, pt1, pt2, color, 2)
            status = "Occupied" if slot_states[idx] else "Free"
            cv2.putText(frame, status, (pt1[0], pt1[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Parking Status", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ======== SCRIPT ENTRYPOINT ========
if __name__ == "__main__":
    main()
