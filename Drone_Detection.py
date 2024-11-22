import cv2
import torch
import warnings
import serial
import time
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'Drone-Detection\best.pt', source='github')


cap = cv2.VideoCapture(0)  # Default camera
confidence_threshold = 0.5

# Initialize serial communication with Arduino
arduino_port = 'COM3'  # Change to your actual Arduino port
baud_rate = 9600
arduino = serial.Serial(arduino_port, baud_rate)
time.sleep(2)  # Allow Arduino to initialize

if not cap.isOpened():
    print("Error: Could not open video source.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to RGB for YOLO
        img = Image.fromarray(frame[..., ::-1])
        
        # Run inference with YOLO model
        results = model(img, size=640)

        # Process detections
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result.tolist()
            
            # Check confidence and target class (class 0 for "drone")
            if conf > confidence_threshold and int(cls) == 0:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Display bounding box and coordinates
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"Coords: ({center_x}, {center_y})", (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Send coordinates to Arduino with less delay
                try:
                    arduino.write(f"{center_x},{center_y}\n".encode('utf-8'))
                except serial.SerialException as e:
                    print(f"Serial communication error: {e}")

        # Show the video feed with bounding boxes
        cv2.imshow('Drone Tracker', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    arduino.close()
    cv2.destroyAllWindows()
