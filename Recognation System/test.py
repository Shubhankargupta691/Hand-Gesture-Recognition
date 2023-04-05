import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow


cap = cv2.VideoCapture(0)  ## id number

detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
imgSize = 300
counter  = 0

labels = ["A","B","C"]

folder = 'DATA/C'

try:
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x,y,w,h = hand['bbox']          #  bounding box

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y - offset: y + h+ offset, x-offset:x + w + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h/w

            if aspectRatio > 1:
                k = imgSize/h
                wcal = math.ceil(k*w)
                imgResize =  cv2.resize(imgCrop, (wcal, imgSize))
                imgResizeShape = imgResize.shape

                wGap = math.ceil((imgSize - wcal)/2)
                imgWhite[:, wGap: wcal+wGap ] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw = False)
                print(prediction,index)

            else:
                k = imgSize / w
                hcal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hcal))
                imgResizeShape = imgResize.shape

                hGap = math.ceil((imgSize - hcal) / 2)
                imgWhite[hGap: hcal + hGap,:] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw = False)

            cv2.rectangle(imgOutput, (x - offset, y - offset-50), (x-offset+90, y - offset-50+50), (255, 255, 255), cv2.FILLED)
            cv2.putText(imgOutput,labels[index],(x,y-26),cv2.FONT_HERSHEY_SIMPLEX,1.7,(255,0,255),2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w+offset, y + h+offset), (255, 0, 255), 4)


            cv2.imshow("ImageCrop",imgCrop)
            cv2.imshow("ImageWhite", imgWhite)


        cv2.imshow("Image",imgOutput)     # to show the image
        cv2.waitKey(1)              # delay of 1 ms

except:
    print("Check your Code")