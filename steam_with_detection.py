# -*- coding: utf-8 -*-
"""
Created on Sun May 30 22:03:36 2021

@author: makki
"""
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
from imutils.video import VideoStream
import imutils

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
face_predictor = model_from_json(loaded_model_json)
# load weights into new model
face_predictor.load_weights("model.h5")

img_height = 480
img_width = 640

def mask_predictions(img):
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(imgGray,1.1,4)
    
    predict_imgs = []
    box_locs = []
    predictions = []
    for x, y, w, h in faces:
        
        face_img = img[y:y+h,x:x+w]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (224,224))
        face_img = img_to_array(face_img)
        face_img = preprocess_input(face_img)
        
        predict_imgs.append(face_img)
        
        box_locs.append([y,y+h,x,x+w])

    if len(faces) > 0:
        predict_imgs = np.array(predict_imgs, dtype="float32")
        predictions = face_predictor.predict(predict_imgs)
        
    return predictions, box_locs
        
cap = VideoStream(src=0).start()
        
while True:
    
    img = cap.read()
    img = imutils.resize(img,img_width,img_height)

    predictions, box_locs = mask_predictions(img)
    
    for preds, box in zip(predictions, box_locs):
        
        y1, y2, x1, x2 = box

        pred_val = np.argmax(preds)
        
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
        
        if pred_val == 0:
            label = 'No Mask'
        elif pred_val == 2:
            label = 'Mask Worn Incorrectly'
        else:
            label = 'Mask'
            
        label = label + f': {preds[pred_val]*100:.1f}%'
            
        cv2.putText(img, label, (x1,y1-20), cv2.FONT_HERSHEY_PLAIN, 1.25, (0,255,0),2)
        
        
    cv2.imshow("video", img)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

cv2.destroyAllWindows()