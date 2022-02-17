import cv2
import numpy as np
import handTracking as ht
import time
import autopy

##############################
wCam, hCam = 600, 400
frameR = 100         #frame reduction
smoothening = 7

##############################
pTime = 0
plocx, plocy = 0,0
clocx, clocy = 0,0

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

detector = ht.handDetector(maxHands=1,detectionCon=0.7)
wScr, hScr = autopy.screen.size()
#print(wScr,hScr)
while True:
    # 1.Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList= detector.findPosition(img)

    #2. Get the tip of the index finger and middle fingers
    if len(lmList)!=0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        #print(x1,y1,x2,y2)


        #3. check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)
        #4. only Index finger : moving mode
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255, 0, 255), 2)
        if fingers[1]==1 and fingers[2]==0:
            #5. Convert Coordinates

            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))


            #6. Smoothing values
            clocx = plocx+(x3-plocx)/smoothening
            clocy = plocy+(y3-plocy)/smoothening

            #7. Move Mouse
            autopy.mouse.move(wScr-clocx,clocy)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocx,plocy = clocx, clocy
            length=detector.dragPos(6,8)
            if length<40:
                autopy.mouse.toggle(down=True)
            else:
                autopy.mouse.toggle(down=False)




        #8. Both Index & middle fingers are up: clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance  betn the fingers (middle & index)
            length, img, lineInfo = detector.findDistance(8,12,img)



            # 10. Click mouse if Distance is short
            if length<30:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()




    #11. Frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    #12. Dispaly
    cv2.imshow("AIMouse",img)
    if cv2.waitKey(10)==ord('q'):
        break
