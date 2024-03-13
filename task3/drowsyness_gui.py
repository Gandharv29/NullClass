import cv2
import face_recognition
import numpy as np
from scipy.spatial import distance

def eye_aspect_ratio(eye):

    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.20
frame_check = 20

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

cap = cv2.VideoCapture(0)
flag = 0

while True:
    ret, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    for face_landmarks in face_landmarks_list:
        leftEye = np.array(face_landmarks["left_eye"])
        rightEye = np.array(face_landmarks["right_eye"])
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)
        if ear < thresh:
            flag += 1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "Drowsy", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print ("Drowsy")
        else:
            flag = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
