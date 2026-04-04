from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")

model.info()

start = time.time()
results = model.train(data="data.yaml", epochs=100, imgsz=640, device="mps")

end = time.time()
print(f"{end-start}sec.")
