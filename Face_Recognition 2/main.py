import threading
import cv2
import os
from deepface import DeepFace

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

# 🔹 Load multiple reference images from a folder
reference_images = []
for file in os.listdir("references"):  # folder name = references
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(os.path.join("references", file))
        reference_images.append(img)

def check_face(frame):
    global face_match
    try:
        for ref_img in reference_images:
            result = DeepFace.verify(frame, ref_img.copy())
            if result['verified']:
                face_match = True
                return
        face_match = False
    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 38 == 8:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
