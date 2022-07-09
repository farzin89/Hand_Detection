import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

offset = 20
imageSize = 300
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True :
    success,img = cap.read()
    hands,img =detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand["bbox"]

        imgWhite = np.ones((imageSize,imageSize,3),np.uint8) * 255 # create picture

        imgCrop = img[y-offset:y +h + offset,x-offset:x+w + offset]

        imgCropshape = imgCrop.shape

        imgWhite[0:imgCropshape[0],0:imgCropshape[1]] = imgCrop

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image",img)
    cv2.waitKey(1)