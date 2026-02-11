import face_recognition
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# ------------------ PATH ------------------
path = "/Users/ipsita/Downloads/faceattendance /images "

images = []
classNames = []

# Load images
myList = os.listdir(path)

for cl in myList:
    imgPath = os.path.join(path, cl)
    curImg = cv2.imread(imgPath)

    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

print("Registered users:", classNames)

# ------------------ ENCODING ------------------
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)
        if enc:
            encodeList.append(enc[0])
    return encodeList

# ------------------ ATTENDANCE ------------------
def markAttendance(name):
    if not os.path.exists("attendance.csv"):
        df = pd.DataFrame(columns=["Name", "Time"])
        df.to_csv("attendance.csv", index=False)

    df = pd.read_csv("attendance.csv")

    if name not in df["Name"].values:
        now = datetime.now()
        timeStr = now.strftime("%H:%M:%S")
        df.loc[len(df)] = [name, timeStr]
        df.to_csv("attendance.csv", index=False)

# Encode known faces
encodeListKnown = findEncodings(images)
print("Encoding completed")

# ------------------ CAMERA ------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible. Try changing VideoCapture(0) to (1)")
    exit()

# ------------------ MAIN LOOP ------------------
while True:
    success, img = cap.read()

    if not success:
        print("Failed to capture image")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.5)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        if len(faceDis) > 0:
            minDis = np.min(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                markAttendance(name)

                # GREEN box
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, name, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                # RED box - NOT REGISTERED
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(img, "NOT REGISTERED", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Face Recognition Attendance", img)

    # Press ENTER to exit
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
