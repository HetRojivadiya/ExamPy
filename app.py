from flask import Flask, jsonify, request
from flask_cors import CORS
import time
import os
import numpy as np
import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import json


cred = credentials.Certificate("./exam.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://examproject-28db7-default-rtdb.firebaseio.com/'
    # Replace with your database URL
})

ref = db.reference('/')


app = Flask(__name__)
CORS(app)

global face_monitoring_active 
face_monitoring_active = True

@app.route('/turnOfFM', methods=['GET'])
def turnOfFM():
    global face_monitoring_active
    face_monitoring_active = False



@app.route('/start_face_monitoring', methods=['GET'])
def start_face_monitoring():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_cap = cv2.VideoCapture(0)
    face_detected = False
    count = 4
    last_face_detected_time = time.time()

    while True:
        ret, video_data = video_cap.read()
        gray_frame = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            if not face_detected:
                time.sleep(5)
                face_detected = True
            else:
                last_face_detected_time = time.time()
                
            for (x, y, w, h) in faces:
                cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            if face_detected and time.time() - last_face_detected_time >= 1:
                if count > 1:
                    face_detected = False
                    message = "Attention: Please focus on the quiz and do not look away."
                    print(message)
                    data = {
                        'indicator': 'True',
                    }
                    ref.child('exam').set(data)
                    print("Data added successfully")
                    print(message)
                    count -= 1
                else:
                    count -= 1

        if count == 0:
            return jsonify(message="stop")
        if(face_monitoring_active == False):
            break;
            
    video_cap.release()

    
reference_image_path = "./het2.jpg"
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

def authenticate_face(input_image):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(input_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    if len(faces) == 0:
        return False

    # Compare the detected face with the reference face
    reference_face = cv2.resize(reference_image, (input_image.shape[1], input_image.shape[0]))
    similarity_score = np.mean(np.abs(input_image - reference_face))

    similarity_threshold = 120

    print(similarity_score)
    
    if similarity_score > similarity_threshold:
        return True
    else:
        return False

@app.route('/capture_and_authenticate', methods=['GET'])
def capture_and_authenticate():
   
    video_cap = cv2.VideoCapture(0)
    
    if not video_cap.isOpened():
        return jsonify(message="Failed to open webcam")
    
    ret, video_data = video_cap.read()
    
    if not ret:
        return jsonify(message="Failed to capture video frame")
    
    gray_frame = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    authenticated = authenticate_face(gray_frame)
    
    if authenticated:
        message = "Authenticated"
    else:
        message = "Authentication failed"
    
    video_cap.release()
    
    return jsonify(message=message)

if __name__ == '__main__':
    app.run(debug=True)
    

    

