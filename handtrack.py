import math

import mediapipe as mp
import cv2
import time
import numpy as np
import autopy

wCam, hCam = 640, 480
frameR = 100
plocX, plocY = 0, 0
clocX, clocY = 0, 0
wScr, hScr = autopy.screen.size()
mp_hand = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
with mp_hand.Hands(
        min_tracking_confidence=0.5,
        min_detection_confidence=0.5,
) as hands:
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print('no image')
            continue

        image = cv2.cvtColor(cv2.flip(img,1), cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lmlist = []
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image, hand_landmarks, mp_hand.HAND_CONNECTIONS
                )
        tip = [4, 8, 12, 16, 20]
        total = 0
        # if finger is up
        if len(lmlist) != 0:
            fingers = []
            if lmlist[tip[0]][1] < lmlist[tip[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):
                if lmlist[tip[id]][2] < lmlist[tip[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            total = fingers.count(1)
            x1, y1 = lmlist[8][1:]
            x2, y2 = lmlist[12][1:]
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                clocX = plocX + (x3 - plocX) / 5
                clocY = plocY + (y3 - plocY) / 5

                autopy.mouse.move(wScr - clocX, clocY)
                cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocY = clocY
                plocX = clocX
            if fingers[1] == 1 and fingers[2] == 1:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                length = math.hypot(x2 - x1, y2 - y1)
                if length < 40:
                    autopy.mouse.click()
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'{total}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 3)
        cv2.putText(image, f'fps: {fps}', (400, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 3)
        cv2.imshow('hand', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()