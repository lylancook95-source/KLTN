import requests
from ultralytics import YOLO
from collections import Counter
import cv2
import time

import lgpio

LED_PIN = 26

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, LED_PIN)
lgpio.gpio_write(h, LED_PIN, 0) 

model = YOLO("/home/pi/best.pt")
SERVER_URL = "http://54.153.160.152:8000/read_camera"
DETECT_URL = "http://54.153.160.152:8000/detect"
UPLOAD_URL = "http://54.153.160.152:8000/upload_image"

def run_yolo():
    led_on()
    time.sleep(0.3)
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    led_off()
    if ret:
      results = model.predict(source=frame, device='cpu', conf=0.25, save=False)
      send_detect(results)

      for i, r in enumerate(results):
          boxes = r.boxes
          high_conf_boxes = boxes[boxes.conf > 0.5]
          r.boxes = high_conf_boxes
          img = r.plot()
          output_path = f"/home/pi/picture{i}.jpg"
          cv2.imwrite(output_path, img)
          print(f"Saved: {output_path}")
          upload_image(output_path)
    else:
        print("khong the doc anh")

def led_on():
    lgpio.gpio_write(h, LED_PIN, 1)

def led_off():
    lgpio.gpio_write(h, LED_PIN, 0)

def upload_image(image_path):
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            r = requests.post(UPLOAD_URL, files=files, timeout=10)

        if r.status_code == 200:
            print("Đã upload ảnh lên server:", r.json())
        else:
            print("Upload ảnh thất bại:", r.status_code)

    except Exception as e:
        print("Lỗi upload ảnh:", e)

def send_detect(results):
    objects = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        cls_ids = r.boxes.cls.tolist()
        for c in cls_ids:
            objects.append(r.names[int(c)])

    if not objects:
        payload = {"Khong phat hien": 1}
        try:
            requests.post(DETECT_URL, json=payload, timeout=5)
            print("YOLO: Không phát hiện vật thể → đã gửi server")
        except Exception as e:
            print("Lỗi gửi detect:", e)
        return


    counted = Counter(objects)
    payload = dict(counted)

    try:
        requests.post(DETECT_URL, json=payload, timeout=5)
        print("Đã gửi lên server:", payload)
    except Exception as e:
        print("Lỗi gửi detect:", e)

def main():
    last_command = None
    print("Waiting server")
    while True:
      try:
        response = requests.get(SERVER_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            command = data.get("data",{}).get("camera")
            if command != last_command:
               if command == 1:
                  print("Nhận lệnh từ server -> Thực thi YOLO")
                  run_yolo()
               else:
                  last_command = command
        time.sleep(0.1)
      except requests.RequestException as e:
        print("Lỗi kết nối server:", e)
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Thoát chương trình")
    finally:
        lgpio.gpiochip_close(h)

