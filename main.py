import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import cvzone

model = YOLO("yolov8s.pt")


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

cap = cv2.VideoCapture("D:\comvi\masjid\masjid1.mp4")


my_file = open("D:\comvi\masjid\coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)

count = 0

tracker = Tracker()

area1 = [(500, 300), (500, 280), (700, 280), (700, 300)]
area2 = [(500, 302), (500, 321), (750, 321), (750, 302)]

people_enter = {}
counter1 = []

people_exit = {}
counter2 = []

out = cv2.VideoWriter("people_counting_system_lab_uin.mp4", -1, 8.0, (1020, 500))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    #   print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    #    print(px)
    list = []
    for index, row in px.iterrows():
        #        print(row)

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if "person" in c:
            list.append([x1, y1, x2, y2])
    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        results = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
        if results > 0:
            people_exit[id] = (x4, y4)
        if id in people_exit:
            results1 = cv2.pointPolygonTest(
                np.array(area2, np.int32), ((x4, y4)), False
            )
            if results1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 1)
                cv2.circle(frame, (x4, y4), 4, (255, 0, 0), -1)
                cvzone.putTextRect(frame, f"{id}", (x3, y3), 1, 2)
                if counter2.count(id) == 0:
                    counter2.append(id)
        results2 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
        if results2 > 0:
            people_enter[id] = (x4, y4)
        if id in people_enter:
            results3 = cv2.pointPolygonTest(
                np.array(area1, np.int32), ((x4, y4)), False
            )
            if results3 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 1)
                cv2.circle(frame, (x4, y4), 4, (255, 0, 0), -1)
                cvzone.putTextRect(frame, f"{id}", (x3, y3), 1, 2)
                if counter1.count(id) == 0:
                    counter1.append(id)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 1)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 1)
    masuk = len(counter1)
    keluar = len(counter2)
    cvzone.putTextRect(frame, f"enter: {masuk}", (50, 50), 1, 2)
    cvzone.putTextRect(frame, f"exit:  {keluar}", (50, 100), 1, 2)
    cv2.putText(
        frame,
        "people counting",
        (450, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        "program by Dienna Aden Maskur",
        (650, 475),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )
    out.write(frame)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
