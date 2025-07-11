import serial
import numpy as np
import cv2
import time

SERIAL_PORT = 'COM6'
BAUD_RATE = 115200
IMAGE_PATH = '5.jpg'
RESIZE_TO = (10, 10)

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    exit("Failed to load image")

flat_image = cv2.resize(img, RESIZE_TO, interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255.0

try:
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as ser:
        time.sleep(1.5)
        ser.write(flat_image.tobytes())
        print(f"Sent {len(flat_image)} float32 values")
except Exception as e:
    print("Serial error:", e)