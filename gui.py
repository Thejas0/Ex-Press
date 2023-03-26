import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from functools import partial

# Import the PhotoImage class from the tkinter module
from tkinter import PhotoImage, Toplevel
import os
import TestEmotionDetector as te
import time
# import TestEmotionDetector as TE
# Create the main window

master = Tk()
master.title("EX-Press")
master.geometry("700x800")
# window = tk.Tk()
# window.title("EX-Press")
# window.geometry("600x455")
# window.config(bg="white")

# image = PhotoImage(file="GUI/LOGO.png")  # Load the image file
# Resize the image while preserving the aspect ratio
# image = image.subsample(1, 1)
# label = tk.Label(image=image)  # Create an image label widget
# Pack the image label widget to the left of the parent widget
# label.pack(side="left")

# Create a label widget to display the time
time_label = tk.Label(master, text="", font=("Arial", 18, "bold"))
time_label.pack()


# Create a button widget
# button = tk.Button(window, image=image, bg="white")

# Update the time label every second


def update_time():
    # Get the current time in hh:mm:ss AM/PM format
    current_time = time.strftime("%I:%M:%S %p")
    time_label.configure(text=current_time)
    # Run this function again after 1000 milliseconds (1 second)
    master.after(1000, update_time)


update_time()  # Initialize the time label

# Define a function for each button that runs the corresponding Python file


def run_file_1():
    # exec(open('Basic/actions.py').read())
    print("hii")


def run_file_2():

    te.plotInit(te.coordLx, te.coordLy)
    # te.plotInit(te.coordRx, te.coordRy, "Right Hand")
    # window1.mainloop()


def run_file_3():

    te.analyze()
    print("hii")


def run_file_4():
    # exec(open('Version 1.1/app.py').read())
    print("hii")


def run_file_5():
    # exec(open('User Guide/GIT ReadMe.py').read())
    root = Tk()

    # specify size of window.
    root.geometry("500x600")

    # Create text widget and specify size.
    T = Text(root, height=10, width=60)

    # Create label
    l = Label(root, text="Suggestion")
    l.config(font=("Courier", 14))

    positive = te.generateFeedback()

    Fact = "From the graph the hotter region (red color region) refers to the coordinates where the hand was placed on the longer duration of time."
    print(positive)
    Fact1 = """"""
    if (positive):
        Fact1 = "As you can see in the plot , you have moved your hands a lot , they appeared in really many places and that is a good sign on having a good body language "
    else:
        Fact1 = "There wasn't much of hand movements in your presentation .Using your hand as a story telling tool is how you can best explain yourself  Improve your hand movements for better impact of your presentaion"
    # Create button for next text.
    Fact += Fact1
    print(Fact)
    l.pack()
    T.pack()

    # Insert The Fact.
    T.insert(tk.END, Fact)
    # T.insert(tk.END, Fact1)

    tk.mainloop()
# Create the buttons


button1 = tk.Button(text="Capture", bg="beige",
                    activebackground="light blue", command=te.capture)
button1.configure(height=7, width=25)
button1.pack(pady=4, padx=4)
button1.config(font=("Playfair", 12, "bold"))
button1.config(borderwidth=2, relief="groove")

button2 = tk.Button(text="Hand Gesture Analysis", bg="beige",
                    activebackground="light blue", command=run_file_2)
# partial(te.plotInit, te.coordLx, te.coordLy, "Left Hand")
button2.configure(height=7, width=25)
button2.pack(pady=4, padx=4)
button2.config(font=("Playfair", 12, "bold"))
button2.config(borderwidth=2, relief="groove")

button5 = tk.Button(text="Hand Gesture Suggestion", bg="beige",
                    activebackground="light blue", command=run_file_5)
button5.configure(height=7, width=25)
button5.pack(pady=4, padx=4)
button5.config(font=("Playfair", 12, "bold"))
button5.config(borderwidth=2, relief="groove")

button3 = tk.Button(text="Facial Expression Analysis", bg="beige",
                    activebackground="light blue", command=run_file_3)
button3.configure(height=7, width=25)
button3.pack(pady=4, padx=4)
button3.config(font=("Playfair", 12, "bold"))
button3.config(borderwidth=2, relief="groove")

button4 = tk.Button(text="Speech Analysis", bg="beige",
                    activebackground="light blue", command=run_file_4)
button4.configure(height=7, width=25)
button4.pack(pady=4, padx=4)
button4.config(font=("Playfair", 12, "bold"))
button4.config(borderwidth=2, relief="groove")


# Pack the buttons into the window
# button1.pack()
# button2.pack()
# button3.pack()
# button4.pack()
# Run the main loop
# window.mainloop()
mainloop()
