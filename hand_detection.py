from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm


cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)
# to capture coordinates of left and right hands
coordLx = []
coordLy = []
coordRx = []
coordRy = []
while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False) # without draw
    # print(hands)
    if hands:
        handType1 = hands[0]["type"]  # Handtype Left or Right
        if (handType1 == 'Left'):
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            for i in range(0, 1000):
                coordLx.append(hand1['bbox'][0])
                coordLy.append(hand1['bbox'][1])
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            # print("Left hand")

        # fingers1 = detector.fingersUp(hand1)

        if (len(hands) == 2 or handType1 == 'Right'):
            # Hand 2
            if (len(hands) == 2):
                hand2 = hands[1]
            else:
                hand2 = hands[0]
            # lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"
            for i in range(0, 1000):
                coordRx.append(hand2['bbox'][0])
                coordRy.append(hand2['bbox'][1])
            fingers2 = detector.fingersUp(hand2)
            # print("Right hand")
            # Find Distance between two Landmarks. Could be same hand or different hands
            # length, info, img = detector.findDistance(
            # lmList1[8], lmList2[8], img)  # with draw
            # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
    # Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

#################


def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def generateFeedback(x, y):
    minX = 10000
    maxX = -10000
    minY = 10000
    maxY = -10000
    difX = 0
    difY = 0
    for i in (x):
        minX = min(minX, i)
        maxX = max(maxX, i)
    for j in (y):
        minY = min(minY, i)
        maxY = max(maxY, j)

    print("Min and max of x coords is %d ", " %d ", minX, maxY)
    print("Min and max of y coords is %d ", " %d ", minY, maxY)

    difX += maxX-minX
    difY += maxY-minY
    text = "The hotter (red color region ) shows that your hand was placed in the region for longer region"
    if (difX > 150 and difY > 100):
        print("Their was a good movement of hand  ")
    else:
        print("Try moving your hand much frequently ")


def plotInit(x, y, hand):
    fig, axs = plt.subplots(1, 2)
    # plt.title(title)
    titleScatter = "Scatter plot"+hand
    titleHeat = "Heat map "+hand
    sigmas = [0, 64]
    # ct = 0
    for ax, s in zip(axs.flatten(), sigmas):
        if s == 0:
            ax.plot(x, y, 'k.', markersize=5)
            ax.set_title(titleScatter)
        else:
            img, extent = myplot(x, y, s)
            ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
            ax.set_title(titleHeat)
            # ax.set_title("Heatmap of Right hand movements")

    #  displaying feedback
    generateFeedback(x, y)
    plt.show()
#################


cap.release()
cv2.destroyAllWindows()
plotInit(coordLx, coordLy, "Left hand")
plotInit(coordRx, coordRy, "Right Hand")
# print(coordLx)
# print(coordLy)
