import cv2
import numpy as np
from keras.models import model_from_json
from cvzone.HandTrackingModule import HandDetector
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm


detector = HandDetector(detectionCon=0.8, maxHands=2)
coordLx = []
coordLy = []
coordRx = []
coordRy = []
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
# cap = cv2.VideoCapture('C:/Users/user/Videos/Captures/production_ID_5198164')

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    hands, img = detector.findHands(frame)  # with draw
    # hands = detector.findHands(img, draw=False) # without draw
    print(hands)
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

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
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


count_emotion = {"Angry": 0, "Disgusted": 0, "Happy": 0,
                 "Neutral": 0, "Sad": 0, "Surprised": 0, "Fearful": 0}


def analyze():

    labels = []
    sizes = []

    for x, y in count_emotion.items():
        labels.append(x)
        sizes.append(y)

    # Plot
    plt.pie(sizes, labels=labels)

    plt.axis('equal')
    plt.show()
    plt.bar(list(count_emotion.keys()), count_emotion.values(), color='g')
    plt.show()
#################


cap.release()
cv2.destroyAllWindows()
plotInit(coordLx, coordLy, "Left hand")
plotInit(coordRx, coordRy, "Right Hand")
# analyze()
