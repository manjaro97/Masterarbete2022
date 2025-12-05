import numpy as np
import math
import cv2 as cv
from matplotlib import pyplot as plt
import os
import glob
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

for i in range(10):
    r = i/10
    theta = 1.57

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r,'-o')
    ax.set_rmax(40)
    ax.set_rticks([10, 20, 30, 40])  # Less radial ticks
    ax.set_rlabel_position(180-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    ax.set_title("A line plot on a polar axis", va='bottom')
    plt.show()