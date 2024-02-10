import cv2
import time
import os
import HandTrackingModule as htm
import random

def RPSmechanism(p1, p2):
    #p1 is user, p2 is computer
    result = ''
    if p1 == p2:
        result = 'Tie'
    elif p1 == 0 and p2 == 5:
        result = 'Computer'
    elif p1 == 5 and p2 == 0:
        result = 'User'
    elif p1 > p2:
        result = 'Computer'
    elif p2 > p1:
        result = 'User'
        
    return result

wCam, hCam = 1500, 1080
pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'fingers'
myLst = os.listdir(folderPath)
overlayList = []
for imPath in myLst:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

detector = htm.handDetector(detectionCon=0.75)

RPS = [0, 2, 5]
waitKey = 0
result = None

while True:
    success, img = cap.read()
    
    img = detector.findHand(img)
    lmList = detector.findPosition(img, draw=False)
    
    userFinger = detector.fingerCounter(img, lmList)
    
    if waitKey % 50 == 0:
        if userFinger == 0 or userFinger == 2 or userFinger == 5:
            computerFinger = random.choice(RPS)
            h, w, c = overlayList[computerFinger - 1].shape
            img[0:h, 0:w] = overlayList[computerFinger - 1]
            result = RPSmechanism(userFinger, computerFinger)
            cv2.putText(img, result, (200, 700),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 15)

    if result != None:
        cv2.putText(img, result, (200, 700),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 15)
        img[0:h, 0:w] = overlayList[computerFinger - 1]
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    cv2.imshow('Image', img)
    cv2.waitKey(1)
    waitKey += 1