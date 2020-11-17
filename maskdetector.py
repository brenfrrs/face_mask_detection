import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pandas as pd
import time
from datetime import datetime

df = pd.DataFrame(columns=['mask','no_mask','time', 'image', 'prediction_confidence'])
cascPath = "models/haarcascade_upperbody.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
model = load_model("models/alt_mask_recog_ver2.h5")

def add_mask():
    global df
    now=datetime.now()
    current_time = now.strftime("%H:%M:%S")
    img_path = #'PATH/with_mask/hasmask{}.jpg'.format(str(datetime.now()))
    pred = "{:.2f}%".format(max(mask, withoutMask) * 100)
    new_row={'mask':1, 'no_mask':0, 'time':current_time, 'image':img_path, 'prediction_confidence':pred}
    cv2.imwrite(img_path,frame)

    df=df.append(new_row, ignore_index=True)
    df.to_csv('maskdata.csv', index = True)

def add_nomask():
    global df
    now=datetime.now()
    current_time = now.strftime("%H:%M:%S")
    img_path = #'PATH/without_mask/hasmask{}.jpg'.format(str(datetime.now()))
    pred = "{:.2f}%".format(max(mask, withoutMask) * 100)
    new_row={'mask':0, 'no_mask':1, 'time':current_time, 'image':img_path, 'prediction_confidence':pred}
    cv2.imwrite(img_path,frame)
    df=df.append(new_row, ignore_index=True)
    df.to_csv('maskdata.csv', index = True)

video_capture = cv2.VideoCapture('http://184.153.36.118:82/mjpg/video.mjpg')


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    faces_list=[]
    preds=[]
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h,x:x+w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame =  preprocess_input(face_frame)
        faces_list.append(face_frame)
        if len(faces_list)>0:
            preds = model.predict(faces_list)
        for pred in preds:
            (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        if mask > withoutMask:
            print('MASK')
            add_mask()
        else:
            print('NO MASK')
            add_nomask()

        cv2.putText(frame, label, (x, y- 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
