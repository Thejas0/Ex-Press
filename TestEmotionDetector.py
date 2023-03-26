import cv2
from matplotlib.figure import Figure
import numpy as np
from keras.models import model_from_json
from cvzone.HandTrackingModule import HandDetector
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

detector = HandDetector(detectionCon=0.8, maxHands=2)
coordLx = []
coordLy = []
coordRx = []
coordRy = []
count_emotion = {"Angry": 0, "Disgusted": 0, "Happy": 0,
                 "Neutral": 0, "Sad": 0, "Surprised": 0, "Fearful": 0}
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


def capture():
    global minnX
    minnX = 1000
    global maxxX
    maxxX = -1000
    global minnY
    minnY = 1000
    global maxxY
    maxxY = -1000
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

    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, 720))
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
            print("Prediction   : ", emotion_prediction, maxindex)
            exp = emotion_dict[maxindex]
            print(exp, " : ", emotion_dict[maxindex])
            if (exp == 'Fearful'):
                exp = 'Neutral'
                maxindex = 4

            count_emotion[emotion_dict[maxindex]] += 1

            cv2.putText(frame, exp, (x+5, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        hands, img = detector.findHands(frame)  # with draw
        # hands = detector.findHands(img, draw=False) # without draw
        print(hands)
        if hands:
            handType1 = hands[0]["type"]  # Handtype Left or Right
            if (handType1 == 'Left'):
                # Hand 1
                hand1 = hands[0]
                lmList1 = hand1["lmList"]
                bbox1 = hand1["bbox"]
                coordLx.append(hand1['bbox'][0])
                coordLy.append(hand1['bbox'][1])
                minnX = min(minnX, hand1['bbox'][0])
                maxxX = max(maxxX, hand1['bbox'][1])

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
                coordLx.append(hand2['bbox'][0])
                coordLy.append(hand2['bbox'][1])
                minnY = min(minnY, hand2['bbox'][0])
                maxxY = max(maxxY, hand2['bbox'][1])
                fingers2 = detector.fingersUp(hand2)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    MaxX = maxxX
    MinX = minnX
    print("msajdnaskndaks   ", MinX, MaxX)
    print("Min max", minnX, maxxX)
#################


def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def generateFeedback():
    print("Min and max of x coords is %d ", " %d ", minnX, minnY)
    print("Min and max of y coords is %d ", " %d ", minnY, maxxY)

    difX = maxxX-minnX
    difY = maxxY-minnY
    text = "The hotter (red color region ) shows that your hand was placed in the region for longer region"
    if (difX > 150 and difY > 100):
        print("There was a good movement of hand  ")
        return True
    else:
        print("Try moving your hand much frequently ")
    return False


def plotInit(x, y):

    # root = tk.Tk()
    # root.geometry("500x600")
    # frameChartsLT = tk.Frame(root)
    # frameChartsLT.pack()
    fig, axs = plt.subplots(1, 2)
    # plt.title(title)
    titleScatter = "Scatter plot"
    titleHeat = "Heat map "
    sigmas = [0, 64]
    # ct = 0
    for ax, s in zip(axs.flatten(), sigmas):
        if s == 0:
            ax.plot(x, y, 'k.', markersize=5)
            ax.set_title(titleScatter)
        else:
            img, extent = myplot(x, y, s)

            ax.imshow(img, extent=extent,
                      origin='lower', cmap=cm.jet)
            ax.set_title(titleHeat)
            # ax.set_title("Heatmap of Right hand movements")

    # chart1 = FigureCanvasTkAgg(fig, frameChartsLT)
    # chart1.get_tk_widget().pack()
    #  displaying feedback
    # generateFeedback()
    plt.show()


def analyze():

    labels1 = []
    sizes = []

    for x, y in count_emotion.items():
        labels1.append(x)
        sizes.append(y)

    root = tk.Tk()
    root.geometry("500x600")
    explode = [0.03, 0, 0.1, 0, 0, 0, 0]
    frameChartsLT = tk.Frame(root)
    frameChartsLT.pack()
    fig = Figure(figsize=(4, 2), dpi=100)  # create a figure object
    ax = fig.add_subplot(111)  # add an Axes to the figure
    textprops = {"fontsize": 8}
    ax.pie(sizes, radius=1, labels=labels1, shadow=True,
           autopct="%0.2f%%", explode=explode, textprops=textprops)

    chart1 = FigureCanvasTkAgg(fig, frameChartsLT)
    chart1.get_tk_widget().pack()

    ###
    f = Figure(figsize=(4, 2), dpi=100)  # Graph
    ax = f.add_subplot(111)
    ll = tk.Label(root, text="No of times the expression was displayed")
    ll.config(font=("Courier", 14))
    ll.pack()

    # the x locations for the groups
    width = .5

    rects1 = ax.barh(list(count_emotion.keys()), count_emotion.values(), width)

    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()
    T = tk.Text(root, height=5, width=52)

    max = 0
    expr = ""
    for x, y in count_emotion.items():
        if (y > max):
            max = y
            expr = x

# Create label
    l = tk.Label(root, text="Analysis ")
    l.config(font=("Courier", 14))
    Fact = expr
    Fact1 = ""
    # {"Angry": 0, "Disgusted": 0, "Happy": 0,
    #              "Neutral": 0, "Sad": 0, "Surprised": 0, "Fearful": 0}
    if (Fact == 'Angry'):
        Fact1 = "The dominant expression in your presentation is Angry.You Appeared To Be Triggered From The Audience,Try To Focus On Yourself And Try To Connect With The Audience On An Emotional Level."
    if (Fact == 'Disgusted'):
        Fact1 = "The dominant expression in your presentation is Disgusted."
    if (Fact == 'Happy'):
        Fact1 = "The dominant expression in your presentation is Happy.You Appeared To Be Happy During Your Presentation,That's Great.It May Make You More Likeable Among The Audience And The Presentation Could Be More Engaging. "
    if (Fact == 'Neutral'):
        Fact1 = "The dominant expression in your presentation is Neutral.Being Pretty Neutral Is Fine,But Try Not To Mix Up Emotions And Confuse Your Audience"
    if (Fact == 'Sad'):
        Fact1 = "The dominant expression in your presentation is Sad.This Might Have A Negative Impact On The Audience,So Try To Be More Lively And Confident While Speaking."
    if (Fact == 'Surprised'):
        Fact1 = "The dominant expression in your presentation is Surprised."
    if (Fact == 'Fearful'):
        Fact1 = "The dominant expression in your presentation is Fearful.In Some Sections You Depicted Fear,So We Feel That You Are Not Confident In Some Of The Topics,Try To Work On Those Weak Spots,Remember Confidence Is Key"
    Fact = Fact1
    l.pack()
    T.pack()
    T.insert(tk.END, Fact)
#################


# capture()

# plotInit(coordLx, coordLy, "Left hand")
# plotInit(coordRx, coordRy, "Right Hand")
# analyze()
