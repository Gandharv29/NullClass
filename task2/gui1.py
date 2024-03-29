import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
#from keras.preprocessing.image import load_img, img_to_array
from keras.models import  load_model
import matplotlib.pyplot as plt
import speech_recognition as sr
import tensorflow as tf
import os 


model_path="task2\model1.h5"

model = load_model(model_path)
face_haar_cascade = cv2.CascadeClassifier("task2\haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)


recognizer = sr.Recognizer()

while True:
    ret, test_img = cap.read() 
    if not ret:
        continue

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h] 
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = roi_gray.astype('float32')
        img_pixels /= 255.0

   
        predictions = model.predict(np.expand_dims(img_pixels, axis=0))
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

   
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Emotion Detector ', resized_img)

  
    with sr.Microphone() as source:
        print("Listening for voice input.")
        audio_data = recognizer.listen(source)

    try:
       
        text = recognizer.recognize_google(audio_data)
        print("Voice input:", text)

       
        if 'quit' in text.lower():
            break

    except sr.UnknownValueError:
        print("The audio cant be understood")
    except sr.RequestError as e:
        print("Could not request results from Speech Recognition ; {0}".format(e))

  
    if cv2.waitKey(10) == ord('q'):
        break


cap.release()