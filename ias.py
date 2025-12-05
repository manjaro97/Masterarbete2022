def drawAccuracyPlot(frame, accuracyArray, plotPos):
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.plot(accuracyArray)
    ax.legend(['aim-point'], loc='lower left')
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "$\\frac{%d}{30}$"%(x)))
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda y, pos: round(y/30, 2) 
            )
        )
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Percent %")
    ax.grid()
    ax.set_ylim(-5, 105)
    fig.suptitle('Accuracy Last Second (10 cm Radius)')

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

def drawPrecisionPlot(frame, precisionArray, plotPos):
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.plot(precisionArray)
    ax.legend(['aim-point'], loc='lower left')
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "$\\frac{%d}{30}$"%(x)))
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda y, pos: round(y/30, 2) 
            )
        )
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Percent %")
    ax.grid()
    ax.set_ylim(-5, 105)
    fig.suptitle('Precision Last Second (10 cm Radius)')

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

    #fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
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
    plt.show()
    

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

def drawAimpointEucAccelerationGraph(frame, aimpointEuclidianAccelerationArray, plotPos):
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.plot(aimpointEuclidianAccelerationArray)
    ax.legend(['aim-point'], loc='lower left')
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "$\\frac{%d}{30}$"%(x)))
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
    #ax.set_xlim(0, 10)
    ax.set_ylim(-40*(97/40), 40*(97/40))
    fig.suptitle('Aim-point Acceleration')

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
 
def drawAimpointEucVelocityGraph(frame, aimpointEuclidianVelocityArray, plotPos):
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.plot(aimpointEuclidianVelocityArray)
    ax.legend(['aim-point'], loc='lower left')
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
    ax.set_ylim(-40*(97/40), 40*(97/40))
    fig.suptitle('Aim-point Velocity')

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

def drawAimpointEucDistGraph(frame, aimpointEuclidianDistanceArray, plotPos):
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.plot(aimpointEuclidianDistanceArray)
    ax.legend(['aim-point'], loc='lower left')
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "$\\frac{%d}{30}$"%(x)))
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
    #ax.set_xlim(0, 10)
    #ax.set_ylim(-40*(97/40), 40*(97/40))
    fig.suptitle('Aim-point Offset Euclidian Distance')

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

def drawOffsetVelocityGraph(frame, aimpointOffsetVelocityArray, plotPos):
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.plot(aimpointOffsetVelocityArray)
    ax.legend(['X-axis', 'Y-axis inverted'], loc='lower left')
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "$\\frac{%d}{30}$"%(x)))
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
    #ax.set_xlim(0, 10)
    ax.set_ylim(-40*(97/40), 40*(97/40))
    fig.suptitle('Aim-point offset velocity')

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

def drawStatisticsGraph(frame, statisticsArray, plotPos):
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.plot(statisticsArray[0])
    ax.plot(statisticsArray[1])
    ax.legend(['Precision', 'Accuracy'], loc='lower left')
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda y, pos: round(y/30, 2) 
            )
        )
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Percentage %")
    ax.grid()
    ax.set_ylim(-5, 105)
    fig.suptitle('Precision and Accuracy Last Second (10 cm Radius)')

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

def drawOffsetMeanGraph(frame, aimpointOffsetMeanArray, plotPos):
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.plot(aimpointOffsetMeanArray)
    ax.legend(['X-axis', 'Y-axis inverted'], loc='lower left')
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "$\\frac{%d}{30}$"%(x)))
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
    #ax.set_xlim(0, 10)
    ax.set_ylim(-16*(97/40), 16*(97/40))
    fig.suptitle('Mean Aim-point Offset')

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

def drawOffsetGraph(frame, aimpointOffsetArray, plotPos):
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.plot(aimpointOffsetArray)
    ax.legend(['X-axis', 'Y-axis inverted'], loc='lower left')
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: "$\\frac{%d}{30}$"%(x)))
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
    #ax.set_xlim(0, 10)
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