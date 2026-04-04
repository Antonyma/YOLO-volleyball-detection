import os
from ultralytics import YOLO
import cv2


VIDEOS_DIR = '/opt/anaconda3/envs/yolov8/data'

video_path = os.path.join(VIDEOS_DIR, 'Volleyball bronze Final.mp4')
video_path_out = '{}_out05n80.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Load a model
model = YOLO("./runs/detect/50epoch,s,train/weights/best.pt")  

target_classes = ["Ball", "Player"]  
color_map = {"PLAYER": (255, 0, 0), "BALL": (0, 255, 0)}

threshold = 0.5

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            class_name = results.names[int(class_id)].upper()
            box_color = color_map.get(class_name)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 4)
            cv2.putText(frame, class_name, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, box_color, 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()