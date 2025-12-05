import numpy as np
import math
import cv2 as cv
from matplotlib import pyplot as plt
import os
import glob
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def drawTargetAndAim(frame, aimpointPos, targetTopLeftPos, targetBottomRightPos):
    #Draw rectangle around target and circle on aimpoint
    green = (0,255,0)
    red = (0,0,255)
    radius = 2
    thickness = 2
    if(aimpointPos[0] > targetTopLeftPos[0] and aimpointPos[0] < targetBottomRightPos[0] and aimpointPos[1] > targetTopLeftPos[1] and aimpointPos[1] < targetBottomRightPos[1]):
        cv.rectangle(frame,targetTopLeftPos, targetBottomRightPos, green, thickness)
        cv.circle(frame, aimpointPos, radius, green, thickness)
    else:
        cv.rectangle(frame,targetTopLeftPos, targetBottomRightPos, red, thickness)
        cv.circle(frame, aimpointPos, radius, red, thickness)
    return frame

def drawPlot(frame, dataArray, plotPos, plotType):
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.plot(dataArray)
    
    if(plotType == "accuracy" or plotType == "precision"):
        ax.legend(['aim-point'], loc='lower left')
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda y, pos: round(y/30, 2) 
                )
            )
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Percent %")
        ax.grid()
        ax.set_ylim(-5, 105)
        if(plotType == "accuracy"):
            fig.suptitle('Accuracy Last Second (10 cm Radius)')
        else:
            fig.suptitle('Precision Last Second (10 cm Radius)')

    elif(plotType == "eucAcceleration" or plotType == "eucVelocity" or plotType == "eucDist" or plotType == "offsetVelocity" or plotType == "offsetMean" or plotType == "offset"):
        if(plotType == "eucAcceleration" or plotType == "eucVelocity" or plotType == "eucDist"):
            ax.legend(['aim-point'], loc='lower left')
        elif(plotType == "offsetVelocity" or plotType == "offsetMean" or plotType == "offset"):
            ax.legend(['X-axis', 'Y-axis inverted'], loc='lower left')
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, pos: int(x/(97/40)) 
                )
            )
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda y, pos: round(y/30, 2) 
                )
            )
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Centimeter")
        ax.grid()
        if(plotType == "eucAcceleration"):
            ax.set_ylim(-40*(97/40), 40*(97/40))
            fig.suptitle('Aim-point Acceleration')
        elif(plotType == "eucVelocity"):
            ax.set_ylim(-40*(97/40), 40*(97/40))
            fig.suptitle('Aim-point Velocity')
        elif(plotType == "eucDist"):
            fig.suptitle('Aim-point Offset Euclidian Distance')
            ax.set_ylim(-5*(97/40), 80*(97/40))
        elif(plotType == "offsetVelocity"):
            ax.set_ylim(-40*(97/40), 40*(97/40))
            fig.suptitle('Aim-point offset velocity')
        elif(plotType == "offsetMean"):
            ax.set_ylim(-16*(97/40), 16*(97/40))
            fig.suptitle('Mean Aim-point Offset')
        elif(plotType == "offset"):
            ax.set_ylim(-40*(97/40), 40*(97/40))
            fig.suptitle('Aim-point offset')
   
    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    plotAsArray = np.asarray(buf)
    # convert to a frame
    plotAsFrame = cv.cvtColor(plotAsArray, cv.COLOR_RGB2BGR)
    # add plot frame on top of original frame
    frame[plotPos[1]:plotPos[1]+plotAsFrame.shape[0], plotPos[0]:plotPos[0]+plotAsFrame.shape[1]] = plotAsFrame
    return frame

def drawPolarPlot(frame, aimpointOffsetArray, plotPos):
    aimOffset = []
    for point in aimpointOffsetArray:
        euclidianDistance = math.sqrt((point[0]**2) + (point[1]**2))
        aimOffset.append(euclidianDistance/(97/40))
    aimAngle = []
    for point in aimpointOffsetArray:
        angle = math.atan2(-point[1], point[0])
        aimAngle.append(angle)

    ## make a Figure and attach it to a canvas.
    fig = Figure(figsize=(5, 5), dpi=100)
    canvas = FigureCanvasAgg(fig)
    
    ## Do some plotting here
    ax = fig.subplots(subplot_kw={'projection': 'polar'})

    if len(aimOffset) > 90:
        r = aimOffset[0:-89]
        theta = aimAngle[0:-89]
        ax.plot(theta, r, '--', color='#d3d3d3', linewidth=1)
    if len(aimOffset) > 60:
        r = aimOffset[-90:-59]
        theta = aimAngle[-90:-59]
        ax.plot(theta, r, color='#3b3bff', linewidth=1)
    if len(aimOffset) > 30:
        r = aimOffset[-60:-29]
        theta = aimAngle[-60:-29]
        ax.plot(theta, r, color='#03925e', linewidth=1)
    
    r = aimOffset[-30:]
    theta = aimAngle[-30:]
    ax.plot(theta, r, color='k')
    ax.plot(theta[-1], r[-1], "-ro")
    ax.set_rmax(50)
    ax.set_rticks([10, 20, 30, 40, 50], ["10","20","30","40","50 cm"])  # Less radial ticks
    ax.set_rlabel_position(180-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    ax.set_title("Aim-point path on a polar axis", va='bottom')
    
    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    plotAsArray = np.asarray(buf)
    # convert to a frame
    plotAsFrame = cv.cvtColor(plotAsArray, cv.COLOR_RGB2BGR)
    # add plot frame on top of original frame
    frame[plotPos[1]:plotPos[1]+plotAsFrame.shape[0], plotPos[0]:plotPos[0]+plotAsFrame.shape[1]] = plotAsFrame

    return frame
#Calculate and draw cumulative mean offset in x and y for each frame
def drawMeanOffset(aimpointOffsetMeanArray, frame, pos):
    meanOffsetX = aimpointOffsetMeanArray[-1][0]
    meanOffsetY = aimpointOffsetMeanArray[-1][1]
    #Draw mean offset as text on video
    cv.putText(frame, ("Mean X Offset =" + str(meanOffsetX)), (pos[0],pos[1]), cv.FONT_HERSHEY_PLAIN, 3.0, (0,0,0), 3)
    cv.putText(frame, ("Mean Y Offset =" + str(-meanOffsetY)), (pos[0],pos[1]+50), cv.FONT_HERSHEY_PLAIN, 3.0, (0,0,0), 3)
    return frame

# Draw time on frame
def drawTime(time, frame, pos):
    cv.putText(frame, ("Time = " + str(time)), pos, cv.FONT_HERSHEY_PLAIN, 3.0, (0,0,0), 3)
    return frame

# Draw Layout
def drawLayout(frame, centroid):
    maxHight, maxWidth = frame.shape[:2]
    boxHigh = centroid[1]-150
    boxLow = centroid[1]+150
    boxRight = centroid[0]+150
    boxLeft = centroid[0]-150

    #Layout Background
    backgroundColor = (128,192,242)
    cv.rectangle(frame,(0,0), (boxLeft,maxHight), backgroundColor, -1)
    cv.rectangle(frame,(0,0), (maxWidth,boxHigh), backgroundColor, -1)
    cv.rectangle(frame,(boxRight,0), (maxWidth,maxHight), backgroundColor, -1)  
    cv.rectangle(frame,(0,boxLow), (maxWidth,maxHight), backgroundColor, -1)

    #Box edge
    edgeColor = (0,0,0)
    edgeSize = 10
    cv.rectangle(frame,(boxLeft-edgeSize,boxHigh-edgeSize), (boxLeft,boxLow+edgeSize), edgeColor, -1)
    cv.rectangle(frame,(boxLeft-edgeSize,boxHigh-edgeSize), (boxRight+edgeSize,boxHigh), edgeColor, -1)
    cv.rectangle(frame,(boxLeft-edgeSize,boxLow), (boxRight+edgeSize,boxLow+edgeSize), edgeColor, -1)
    cv.rectangle(frame,(boxRight,boxHigh-edgeSize), (boxRight+edgeSize,boxLow+edgeSize), edgeColor, -1)
    return frame

# Draw Overlay
def drawOverlay(frame):
    maxHight, maxWidth = frame.shape[:2]

    #Layout Background
    backgroundColor = (128,192,242)
    cv.rectangle(frame,(0,0), (maxWidth,maxHight), backgroundColor, -1)
    return frame

def drawStatistics(frame, statisticsArray, pos):
    precision = statisticsArray[0][-1]
    accuracy = statisticsArray[1][-1]
    cv.putText(frame, ("Precision =" + str(precision) + "%"), pos, cv.FONT_HERSHEY_PLAIN, 3.0, (0,0,0), 3)
    cv.putText(frame, ("Accuracy =" + str(accuracy) + "%"), (pos[0],pos[1]+50), cv.FONT_HERSHEY_PLAIN, 3.0, (0,0,0), 3)
    return frame

def zoom_at(frame, zoom=2, angle=0, coord=None):
    
    cy, cx = [ i/2 for i in frame.shape[:-1] ] if coord is None else coord[::-1]
    
    rot_mat = cv.getRotationMatrix2D((cx,cy), angle, zoom)
    frame = cv.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv.INTER_LINEAR)
    
    return frame

def drawVis(frame, time, centroid, aimpointOffsetMeanArray, statisticsArray):
    #Draw Layout
    frame = drawLayout(frame, centroid)

    frame = zoom_at(frame)

    #Draw time as text on video
    frame = drawTime(time, frame, (50,450))
    #Draw mean aimpoint offset on choosen video
    frame = drawMeanOffset(aimpointOffsetMeanArray, frame, (50,500))
    #Draw precision and accuracy
    frame = drawStatistics(frame, statisticsArray, (50,600))
    return frame

def drawEdge(frame, plotCentre, plotSize):
    boxHigh = plotCentre[1]-int(plotSize[1]/2)
    boxLow = plotCentre[1]+int(plotSize[1]/2)
    boxRight = plotCentre[0]+int(plotSize[0]/2)
    boxLeft = plotCentre[0]-int(plotSize[0]/2)

    #Box edge
    edgeColor = (0,0,0)
    edgeSize = 10
    cv.rectangle(frame,(boxLeft-edgeSize,boxHigh-edgeSize), (boxLeft,boxLow+edgeSize), edgeColor, -1)
    cv.rectangle(frame,(boxLeft-edgeSize,boxHigh-edgeSize), (boxRight+edgeSize,boxHigh), edgeColor, -1)
    cv.rectangle(frame,(boxLeft-edgeSize,boxLow), (boxRight+edgeSize,boxLow+edgeSize), edgeColor, -1)
    cv.rectangle(frame,(boxRight,boxHigh-edgeSize), (boxRight+edgeSize,boxLow+edgeSize), edgeColor, -1)
    return frame

def drawFinalLayout(frame, polarPlotPos, accelerationPos, precisionPos, accuracyPos):
    #Draw Layout
    maxHight, maxWidth = frame.shape[:2]

    #Define colors
    backgroundColor = (128,192,242)
    edgeColor = (0,0,0)

    #Background
    cv.rectangle(frame,(0,0), (int(maxWidth*2/3),maxHight), backgroundColor, -1)

    #Edges
    cv.rectangle(frame,(0,0), (maxWidth,10), edgeColor, -1)
    cv.rectangle(frame,(0,0), (10,maxHight), edgeColor, -1)
    cv.rectangle(frame,(maxWidth-10,0), (maxWidth,maxHight), edgeColor, -1)
    cv.rectangle(frame,(0,maxHight-10), (maxWidth,maxHight), edgeColor, -1)
    cv.rectangle(frame,(int(maxWidth*2/3)-5,0), (int(maxWidth*2/3)+5,maxHight), edgeColor, -1)

    #Define Plot sizes
    polarPlotSize = (500, 500)
    accelerationSize = (500, 400)


    #Calculate plot centres
    polarPlotCentre = (polarPlotPos[0]+int(polarPlotSize[0]/2), polarPlotPos[1]+int(polarPlotSize[1]/2))
    accelerationCentre = (accelerationPos[0]+int(accelerationSize[0]/2), accelerationPos[1]+int(accelerationSize[1]/2))
    precisionCentre = (precisionPos[0]+int(accelerationSize[0]/2), precisionPos[1]+int(accelerationSize[1]/2))
    accuracyCentre = (accuracyPos[0]+int(accelerationSize[0]/2), accuracyPos[1]+int(accelerationSize[1]/2))

    #Draw edges around plots
    frame = drawEdge(frame, polarPlotCentre, polarPlotSize)
    frame = drawEdge(frame, accelerationCentre, accelerationSize)
    frame = drawEdge(frame, precisionCentre, accelerationSize)
    frame = drawEdge(frame, accuracyCentre, accelerationSize)

    return frame

def drawAimPath(frame, aimpointOffsetArray, aimpoint, frameNr, pathLength):
    #Showing aim point path on each frame by using previously stored data
    prevpoint = (0,0)
    framesDrawn = 1
    for x in aimpointOffsetArray:
        if(framesDrawn >= frameNr-pathLength):
            if(frameNr != 1):
                frame = cv.line(frame, (aimpoint[0]+prevpoint[0], aimpoint[1]+prevpoint[1]), (aimpoint[0]+x[0], aimpoint[1]+x[1]), (166,166,166), 1)
            cv.circle(frame, (aimpoint[0]+x[0], aimpoint[1]+x[1]), 2, (0,0,0), 1)
        framesDrawn += 1
        prevpoint = x
    return frame

def warpFunc(frame, centroid, warpXY): 
    framehight, frameWidth = frame.shape[:2]
    canvasSize = (frameWidth, framehight)

    targetoffset = (warpXY[0]-centroid[0], warpXY[1]-centroid[1])
    #Warping image to set centroid in the chosen coordinates
    translation_matrix = np.float32([ [1,0,targetoffset[0]], [0,1,targetoffset[1]] ])
    frame = cv.warpAffine(frame, translation_matrix, canvasSize)
    return frame


# createEditedVid = Warped target with Layout, data and aim-point path
def createEditedVid(frame, centroid, aimpoint, aimpointOffsetArray, frameNr, pathLength, time, aimpointOffsetMeanArray, statisticsArray):
    #Warping image to get target centre in video centre
    frame = warpFunc(frame, centroid, aimpoint)
    #draw aim-path
    frame = drawAimPath(frame, aimpointOffsetArray, aimpoint, frameNr, pathLength)
    #draw visualization
    frame = drawVis(frame, time, aimpoint, aimpointOffsetMeanArray, statisticsArray)
    return frame

def drawEdgeWithBorder(frame, dataArray, plotPos, plotModel, plotSize):
    frame = drawEdge(frame, (plotPos[0]+int(plotSize[0]/2), plotPos[1]+int(plotSize[1]/2)), plotSize)
    frame = drawPlot(frame, dataArray, plotPos, plotModel)
    return frame

# createVidGraph = Frame of plots
def createVidGraph(frame, aimpointOffsetArray, aimpointOffsetMeanArray, statisticsArray, aimpointEuclidianDistanceArray, aimpointEuclidianVelocityArray, aimpointEuclidianAccelerationArray):
    #Draw Layout
    frame = drawOverlay(frame)
    plotSize = (500,400)

    #Draw edges around plots
    drawEdgeWithBorder(frame, aimpointOffsetArray, (10,10), "offset", plotSize)
    drawEdgeWithBorder(frame, aimpointOffsetMeanArray, (520,10), "offsetMean", plotSize)
    drawEdgeWithBorder(frame, statisticsArray[0], (1030,10), "precision", plotSize)
    drawEdgeWithBorder(frame, statisticsArray[1], (1030,420), "accuracy", plotSize)
    drawEdgeWithBorder(frame, aimpointEuclidianDistanceArray, (10,420), "eucDist", plotSize)
    #drawEdgeWithBorder(frame, aimpointEuclidianVelocityArray, (465,500), "eucVelocity", plotSize)
    drawEdgeWithBorder(frame, aimpointEuclidianAccelerationArray, (520,420), "eucAcceleration", plotSize)
    return frame

def createPolarPlot(frame, aimpointOffsetArray):
    frame = drawPolarPlot(frame, aimpointOffsetArray, (0,0))
    return frame

def createFinalVid(frame, aimpoint, aimpointOffsetArray, aimpointEuclidianAccelerationArray, statisticsArray):
    #Warping image to get target centre in a set position
    frame = warpFunc(frame, aimpoint, (1600,aimpoint[1]))
    
    polarPlotPos = (100,50)
    accelerationPos = (100,600)
    precisionPos = (650, 100)
    accuracyPos = (650, 600)

    frame = drawFinalLayout(frame, polarPlotPos, accelerationPos, precisionPos, accuracyPos)

    frame = drawPolarPlot(frame, aimpointOffsetArray, polarPlotPos)
    frame = drawPlot(frame, aimpointEuclidianAccelerationArray, accelerationPos, "eucAcceleration")

    precisionArray = statisticsArray[0]
    accuracyArray = statisticsArray[1]
    frame = drawPlot(frame, accuracyArray, precisionPos, "accuracy")
    frame = drawPlot(frame, precisionArray, accuracyPos, "precision")

    return frame


#Different template matching algorithms
#methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

#Chosen template matching algorithm
method = eval('cv.TM_CCOEFF_NORMED')

#initialize array
aimpointOffsetArray = []
aimpointOffsetMeanArray = []
precisionArray = []
accuracyArray = []
aimpointOffsetVelocityArray = []
aimpointEuclidianDistanceArray = []
aimpointEuclidianVelocityArray = []
aimpointEuclidianAccelerationArray = []

#Chosen input video
nr = 4
target = "input/target%d.jpg" % nr
video = "input/kct%d.mp4" % nr

#initialize variable keeping track of frame number
frameNr = 1

#Set aim-point path length
pathLength = 10

#Capturing the frames from the video
frame = cv.Mat 
cv.namedWindow("Displayed Video")
cap = cv.VideoCapture(video)


def createVideo(name):
    return cv.VideoWriter('output/%s.mp4' % name, 
                     cv.VideoWriter_fourcc(*'MP4V'),
                     30, (int(cap.get(3)), int(cap.get(4))))

def createPolarPlotVideo(name):
    return cv.VideoWriter('output/%s.mp4' % name, 
                     cv.VideoWriter_fourcc(*'MP4V'),
                     30, (500, 500))

def createGraphVideo(name):
    return cv.VideoWriter('output/%s.mp4' % name, 
                     cv.VideoWriter_fourcc(*'MP4V'),
                     30, (1540, 830))

#Saving edited frames to new file
editedVid = createVideo("editedVid")
graphVid = createGraphVideo("graphVid")
finalVid = createVideo("finalVid")
polarPlot = createPolarPlotVideo("polarPlot")

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
    time = round(frameNr/30, 2)
    
    # save frame as JPEG file
    cv.imwrite("frames/frame%d.jpg" % frameNr, frame)     
    
    #Read frame frome file
    img = cv.imread("frames/frame%d.jpg" % frameNr,0)
    template = cv.imread(target,0)
    templateWidth, templateHight = template.shape[::-1]

    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    targetTopLeftPos = max_loc
    targetBottomRightPos = (targetTopLeftPos[0] + templateWidth, targetTopLeftPos[1] + templateHight)
    centroid = (targetTopLeftPos[0] + int(templateWidth/2), targetTopLeftPos[1] + int(templateHight/2))
    if(frameNr == 1):
        aimpointPos = centroid
        firstimage = frame.copy()

    #Calculate aimpoint offset
    aimpointOffset = (((aimpointPos[0])-centroid[0]),((aimpointPos[1])-centroid[1]))
    aimpointOffsetArray.append(aimpointOffset)

    #Calculate Euclidian Distance
    aimpointEuclidianDistanceArray.append(math.sqrt((aimpointOffsetArray[-1][0]**2) + (aimpointOffsetArray[-1][1]**2)))

    #Calculate Derivate of the euclidian distance to get velocity
    if(len(aimpointEuclidianDistanceArray) > 1):
        aimpointEuclidianVelocityArray.append(aimpointEuclidianDistanceArray[-1]-aimpointEuclidianDistanceArray[-2])
    
    #Calculate Derivate of the velocity to get acceleration
    if(len(aimpointEuclidianAccelerationArray) < 2):
        aimpointEuclidianAccelerationArray.append(0)
    if(len(aimpointEuclidianVelocityArray) > 1 ):
        aimpointEuclidianAccelerationArray.append(aimpointEuclidianVelocityArray[-1]-aimpointEuclidianVelocityArray[-2])

    #Calculate aimpoint offset mean
    meanOffsetX = 0 
    meanOffsetY = 0
    nrofpoints = 0
    for x in aimpointOffsetArray:
        meanOffsetX += x[0]
        meanOffsetY += x[1]
        nrofpoints += 1
    meanOffsetX = round(meanOffsetX/nrofpoints, 2)
    meanOffsetY = round(meanOffsetY/nrofpoints, 2)
    aimpointOffsetMeanArray.append((meanOffsetX, meanOffsetY))

    #Calculate precision and accuracy
    checkFrameAmount = 30
    frameNr = 1
    accuratePoints = 0 
    precisePoints = 0 
    checkedPoints = 0
    for point in aimpointOffsetArray:
        if(frameNr > len(aimpointOffsetArray) - checkFrameAmount):
            if(math.sqrt(((aimpointOffsetArray[-1][0]-point[0])**2)+((aimpointOffsetArray[-1][1]-point[1])**2)) <= 10*(97/40)):
                precisePoints += 1
            if(math.sqrt((point[0]**2)+(point[1]**2)) <= 10*(97/40)):
                accuratePoints += 1
            checkedPoints += 1
        frameNr += 1
    #append (precision, accuracy)
    precisionArray.append(round(100*(precisePoints/checkedPoints), 1))
    accuracyArray.append(round(100*(accuratePoints/checkedPoints), 1))
    statisticsArray = (precisionArray, accuracyArray)

    #Calculate Derivate of the offset
    if(len(aimpointOffsetArray) > 1):
        aimpointOffsetVelocityArray.append((aimpointOffsetArray[-1][0]-aimpointOffsetArray[-2][0],aimpointOffsetArray[-1][1]-aimpointOffsetArray[-2][1]))

    #Draw Target and Aimpoint on frame
    frameWithTargetAndAim = drawTargetAndAim(frame.copy(), aimpointPos, targetTopLeftPos, targetBottomRightPos)

    # Video with Overlay and data
    warpPos = (int(frameWidth/2), int(framehight/2))
    #frameEdited = createEditedVid(frameWithTargetAndAim.copy(), centroid, warpPos, aimpointOffsetArray, frameNr, pathLength, time, aimpointOffsetMeanArray, statisticsArray)
    #editedVid.write(frameEdited)

    # Vis with Graphs
    test_image = np.zeros((830,1540,3), np.uint8)
    frameGraph = createVidGraph(test_image, aimpointOffsetArray, aimpointOffsetMeanArray, statisticsArray, aimpointEuclidianDistanceArray, aimpointEuclidianVelocityArray, aimpointEuclidianAccelerationArray)
    graphVid.write(frameGraph)

    # Vis Polar Plot
    blank_image = np.zeros((500,500,3), np.uint8)
    #framePolarPlot = createPolarPlot(blank_image, aimpointOffsetArray)
    #polarPlot.write(framePolarPlot)

    # Final Vid #(1440,540)
    #frameFinalVid = createFinalVid(frameWithTargetAndAim.copy(), aimpointPos, aimpointOffsetArray, aimpointEuclidianAccelerationArray, statisticsArray)
    #finalVid.write(frameFinalVid)


    #Delete frame file and iterate to next frame
    if os.path.exists("frames/frame%d.jpg" % frameNr):
        os.remove("frames/frame%d.jpg" % frameNr)
    frameNr += 1

    #Show the video on screen
    cv.imshow("Displayed Video", frameGraph)
    
    if cv.waitKey(1) == ord('q'): break

#Clean frames folder
files = glob.glob('/frames/*')
for f in files:
    os.remove(f)

editedVid.release()
graphVid.release()
polarPlot.release()
finalVid.release()
cap.release()
cv.destroyAllWindows()