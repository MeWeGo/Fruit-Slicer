import mss
import numpy as np
import mouse  # Daha hızlı fare kontrolü
import time
import cv2
import torch
from ultralytics import YOLO

# YOLO Modelini FP16 ile Yükle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('best.pt').to(device).half()

tracked_objects = {}

def capture_screen():
    """Hızlı ekran görüntüsü alır."""
    with mss.mss() as sct:
        monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}  
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)[:, :, :3]  # RGB
        return img

def detect_objects(image):
    """YOLO kullanarak nesneleri tespit eder."""
    small_img = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LINEAR)  # Daha hızlı resize

    with torch.cuda.amp.autocast():
        results = model(small_img)

    detected_objects = []
    scale_x = image.shape[1] / 1920
    scale_y = image.shape[0] / 1080 

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])  
            class_id = int(box.cls[0])  

            if confidence > 0.35:  
                center_x, center_y = int((x1 + x2) / 2 * scale_x), int((y1 + y2) / 2 * scale_y)
                detected_objects.append((center_x, center_y, class_id))

    return detected_objects

def slice_movement(x, y):
    """Daha hızlı slicing hareketi yapar."""
    angle_x = np.random.randint(-50, 50)  
    mouse.move(x + angle_x, y - 30, absolute=True, duration=0.01)  
    mouse.press(button='left')  
    mouse.move(x - angle_x, y + 100, absolute=True, duration=0.05)  
    mouse.release(button='left')  
    time.sleep(0.005)  # Küçük bir bekleme süresi

def process_objects(objects):
    """Hareketli nesneleri daha hızlı takip eder ve birden fazla nesneyi işler."""
    global tracked_objects

    for x, y, class_id in objects:
        if class_id == 0:  
            continue  

        obj_key = (x, y, class_id)

        # Takip algoritmasını daha hızlı hale getir (50px → 80px)
        for prev_key in list(tracked_objects.keys()):
            px, py, pclass = prev_key
            if pclass == class_id and abs(px - x) < 80 and abs(py - y) < 80:  
                if time.time() - tracked_objects[prev_key] < 0.1:
                    return  

        print(f"Slicing object at ({x}, {y}) - Class {class_id}")
        slice_movement(x, y)  

        tracked_objects[obj_key] = time.time()  

def main():
    while True:
        screen_img = capture_screen()
        objects = detect_objects(screen_img)
        process_objects(objects)

if __name__ == "__main__":
    main()
