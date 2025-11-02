import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime

# Path for dataset folder
path = 'dataset'
images = []
names = []

# Load all images
for file in os.listdir(path):
    curImg = cv2.imread(f'{path}/{file}')
    images.append(curImg)
    names.append(os.path.splitext(file)[0])

print(f"[INFO] Loaded {len(images)} known faces.")

# Function to encode known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)[0]
        encodeList.append(enc)
    return encodeList

# Create encodings
print("[INFO] Encoding faces...")
encodeListKnown = findEncodings(images)
print("[INFO] Encoding complete.")

# Load or create attendance file
excel_file = 'attendance.xlsx'
try:
    attendance = pd.read_excel(excel_file)
except FileNotFoundError:
    attendance = pd.DataFrame(columns=['Name', 'Time'])

# Function to mark attendance
def markAttendance(name):
    now = datetime.now()
    time_string = now.strftime('%H:%M:%S')
    if name not in attendance['Name'].values:
        attendance.loc[len(attendance)] = [name, time_string]
        attendance.to_excel(excel_file, index=False)
        print(f"[LOG] {name} marked present at {time_string}.")

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam...")

while True:
    success, img = cap.read()
    if not success:
        print("[ERROR] Camera not found!")
        break

    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgSmall)
    encodesCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = names[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Face Recognition Attendance', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Attendance session ended.")
