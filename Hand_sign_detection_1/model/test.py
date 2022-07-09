import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow
import time
offset = 20
imageSize = 300
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5","labels.txt")

#this part is for saving our images in our folders with presing the S

folder = "Data/c"
counter = 0
labels = ["A","B","C"]
while True :
    success,img = cap.read()
    hands,img =detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand["bbox"]

        imgWhite = np.ones((imageSize,imageSize,3),np.uint8) * 255 # create picture

        imgCrop = img[y-offset:y +h + offset,x-offset:x+w + offset]

        imgCropshape = imgCrop.shape


        aspectRation = h/w

        if aspectRation > 1 :
            k = imageSize / h
            wCal =math.ceil (k * w)
            imgResize = cv2.resize(imgCrop,(wCal,imageSize))
            imgResizeShape = imgResize.shape
            # put bbox in center of white image
            wGap = math.ceil((imageSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction,index = classifier.getPrediction(img)
            print(prediction,index)


        else:
            k = imageSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imageSize, hCal))
            imgResizeShape = imgResize.shape
            # put bbox in center of white image
            hGap = math.ceil((imageSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image",img)
    cv2.waitKey(1)
