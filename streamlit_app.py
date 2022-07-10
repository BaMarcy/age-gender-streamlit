# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 22:46:13 2022

@author: Deepworker
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import time
import av
from torchvision import transforms, models
import torch 
import numpy as np
import torch.nn as nn
import cvlib as cv
import cv2

st.title('Real-Time Age and Gender Estimation with PyTorch')
st.write("[deepworker.net](http://deepworker.net/)")
st.write("[GitHub](https://github.com/BaMarcy/age_gender_predictor)")
st.write("This app runs in a Docker container and hosted on Heroku.")

class ageGenderClassifier(nn.Module):
    """
    ANN multi-task learning class for VGG16, RESNET18 and RESNET34
    """
    def __init__(self, num_units=512, num_age_classes=9, dropout=0.1):

        super(ageGenderClassifier, self).__init__()

        self.age_classifier = nn.Sequential(
            nn.Linear(num_units,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128, num_age_classes)
        )
        self.gender_classifier = nn.Sequential(
            nn.Linear(num_units,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        #print(x.shape)
        age = self.age_classifier(x)
        gender = self.gender_classifier(x)
        return gender, age

class GenderAgeModel():
   
    def __init__(self, im_size=224, tfms=None): 
        self.im_size = im_size 
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Imagenet standards
        self.dict_age = {'0-2': 0,
                         '3-9': 1,
                         '10-19': 2,
                         '20-29': 3,
                         '30-39': 4,
                         '40-49': 5,
                         '50-59': 6,
                         '60-69': 7,
                         'more than 70': 8} 
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = models.resnet34(pretrained=True)
        self.model.fc = ageGenderClassifier()
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load("resnet34.pt", map_location=torch.device('cpu')))
        self.model.eval()
        
    def get_age(self, age):
        key_list = list(self.dict_age.keys())
        val_list = list(self.dict_age.values())
        position = val_list.index(age)
        return key_list[position]
    
    def preprocess_image(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.im_size, self.im_size))
        
        im = torch.tensor(im).permute(2,0,1)
        im = self.normalize(im/255.)
        return im[None]
    
    def predict(self, im):
        X = self.preprocess_image(im).to(self.device)
        gender, age = self.model(X)
        pred_gender = gender.cpu().detach().numpy()
        pred_age = torch.argmax(age, dim=1).cpu().detach().numpy()
        return (np.where(pred_gender[0][0] < 0.5, 'Male', 'Female'), self.get_age(pred_age))
    
model = GenderAgeModel()


def process(image):
    img_h, img_w, img_c = image.shape
    #image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    start = time.time()
    
    try:
        faces, _ = cv.detect_face(image)
        for f in faces:
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 1)
            
            gender, age = model.predict(image[startY:endY, startX:endX])
                        
            cv2.putText(image, str(gender), (endX, startY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, str(age), (endX, startY+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    except:
        pass

    cv2.putText(image, "by Marcell Balogh", (img_w-100, img_h-20), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
    
    
    end = time.time()
    totalTime = end - start
    try:
        fps = 1 / totalTime
        cv2.putText(image, f'FPS: {int(fps)}', (img_w-80, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    except:
        pass

    
            
           
    return image




class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process(img)  
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    
webrtc_ctx = webrtc_streamer(
    key="head pose",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)