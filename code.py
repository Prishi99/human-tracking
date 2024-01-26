import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import cvzone

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture("C:/Users/prish/PycharmProjects/opencvpython/pythonProject/.venv/Resources/vidp.mp4")

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker = Tracker()
counter_up = []
counter_down = []

cy1 = 194
cy2 = 220
offset = 6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []

    for index, row in px.iterrows():
        x1, y1, x2, y2, d = map(int, row[:5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

        if cy1 - offset < cy < cy1 + offset:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            if id not in counter_down:
                counter_down.append(id)

        if cy2 - offset < cy < cy2 + offset:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            if id not in counter_up:
                counter_up.append(id)

    cv2.line(frame, (3, cy1), (1018, cy1), (0, 255, 0), 2)
    cv2.line(frame, (5, cy1), (1019, cy1), (0, 255, 255), 2)

    down = len(counter_down)
    up = len(counter_up)

    cvzone.putTextRect(frame, f'Down: {down}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'Up: {up}', (50, 160), 2, 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
