import cv2
import numpy as np
import time
import autopy
import HandTrackingModule as htm

# ================= SETTINGS =================
wCam, hCam = 640, 480
frameR = 80
smoothening = 5
clickDelay = 0.25
# ============================================
# Camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Hand Detector
detector = htm.HandDetector(maxHands=1, detectionCon=0.7, trackCon=0.7)

# Screen Size
wScr, hScr = autopy.screen.size()

# Variables
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
prevClickTime = 0

while True:

    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)

    # Detect Hands
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    # Draw Active Region
    cv2.rectangle(img, (frameR, frameR),
                  (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)

    if len(lmList) != 0:

        fingers = detector.fingersUp()

        if len(fingers) != 0:

            # Get Index finger tip
            x1, y1 = lmList[8][1], lmList[8][2]
            # Get Middle finger tip
            x2, y2 = lmList[12][1], lmList[12][2]

            # ================= MOVE MODE =================
            if fingers[1] == 1 and fingers[2] == 0:

                # Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # Smoothen
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # Move Mouse
                autopy.mouse.move(clocX, clocY)

                # Update previous
                plocX, plocY = clocX, clocY

                # Draw
                cv2.circle(img, (x1, y1), 12,
                           (255, 0, 255), cv2.FILLED)

            # ================= CLICK MODE =================
            if fingers[1] == 1 and fingers[2] == 1:

                length, img, lineInfo = detector.findDistance(8, 12, img)

                currentTime = time.time()

                if length < 35 and (currentTime - prevClickTime) > clickDelay:

                    cv2.circle(img,
                               (lineInfo[4], lineInfo[5]),
                               15, (0, 255, 0), cv2.FILLED)

                    autopy.mouse.click()
                    prevClickTime = currentTime

    # ================= FPS =================
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}',
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
