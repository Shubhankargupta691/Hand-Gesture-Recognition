import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)  ## id number

detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300
counter  = 0

folder = 'DATA/C'

try:
    while True:
        success, img = cap.read()
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
            else:
                k = imgSize / w
                hcal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hcal))
                imgResizeShape = imgResize.shape

                hGap = math.ceil((imgSize - hcal) / 2)
                imgWhite[hGap: hcal + hGap,:] = imgResize


            cv2.imshow("ImageCrop",imgCrop)
            cv2.imshow("ImageWhite", imgWhite)


        cv2.imshow("Image",img)     # to show the image
        key = cv2.waitKey(1)              # delay of 1 ms

        if key == ord("s"):
            counter+=1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
            print(counter)
except:
    print("Check your Code")