from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import numpy.random
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
coordx = []
coordy = []
while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False) # without draw
    print(hands)
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        coordx.append(hand1['bbox'][0])
        coordy.append(hand1['bbox'][1])
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        # fingers1 = detector.fingersUp(hand1)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            # lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            fingers2 = detector.fingersUp(hand2)

            # Find Distance between two Landmarks. Could be same hand or different hands
            # length, info, img = detector.findDistance(
            # lmList1[8], lmList2[8], img)  # with draw
            # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
    # Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
heatmap, xedges, yedges = np.histogram2d(coordx, coordy, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()
