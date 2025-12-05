import numpy as np
import math
import cv2 as cv
from matplotlib import pyplot as plt
import os
import glob
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure



#Chosen input video
nr = 1
target = "input/target%d.jpg" % nr
video = "input/kct%d.mp4" % nr

#initialize variable keeping track of frame number
frameNr = 1

#Capturing the frames from the video
frame = cv.Mat 
cv.namedWindow("Displayed Video")
cap = cv.VideoCapture(video)


def createVideo(name):
    return cv.VideoWriter('output/%s.mp4' % name, 
                     cv.VideoWriter_fourcc(*'MP4V'),
                     30, (int(cap.get(3)), int(cap.get(4))))

#Saving edited frames to new file
vid = createVideo("vid")

while( cap.isOpened()):#and (frameNr < 400)
    ret, frame = cap.read()
    if (ret == False):
        break

    framehight, frameWidth = frame.shape[:2]
    canvasSize = (frameWidth, framehight)

    # if frame is read 
    # correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    time = cap.get(cv.CAP_PROP_POS_MSEC)
    time = round(time, 2)

    ## Vis 1
    vid.write(frame)


    #Delete frame file and iterate to next frame
    if os.path.exists("frames/frame%d.jpg" % frameNr):
        os.remove("frames/frame%d.jpg" % frameNr)
    frameNr += 1
    print(frameNr)

    #Show the video on screen
    cv.imshow("Displayed Video", frame)
    
    if cv.waitKey(1) == ord('q'): break

#Clean frames folder
files = glob.glob('/frames/*')
for f in files:
    os.remove(f)

vid.release()
cv.destroyAllWindows()