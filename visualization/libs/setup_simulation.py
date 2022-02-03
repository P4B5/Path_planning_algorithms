from ctypes import sizeof
from math import sqrt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rand
import cv2 as cv
import csv
import queue
import statistics
import time

# from kinematic_model import KinematicBicycleModel
from libs.dynamic_model import Car
from matplotlib.animation import FuncAnimation
from libs.car_description import Description
from libs.cubic_spline_interpolator import generate_cubic_spline
import matplotlib.animation as animation

# own libraries
from libs.generate_roadmaps import binary_threshold_image
from libs.gui import *
from libs.generate_path import *



#############################################################
#
#                   ----- PARAMETERS ---
#
#############################################################

simulation_time = 1000.0             #time after that the simulation stopps [s]


#############################################################
#
#                   ----- CLASSES----
#
#############################################################

class Simulation:

    def __init__(self):
        fps = 50.0
        self.dt = 1/fps
        self.map_size_x = 90.0
        self.map_size_y = 60.0
        self.frames = int(simulation_time*fps)
        self.loop = False

class Path:

    def __init__(self):
       
        # Get path to waypoints.csv
        relative_path = "../../visualization/data/waypoints.csv"       # Path to data relative to script
        dirname = os.path.dirname(__file__)
        file_path = os.path.join(dirname, relative_path)

        df = pd.read_csv(file_path)

        x = df['X-axis'].values
        y = df['Y-axis'].values

        self.points_x = x
        self.points_y = y
        ds = 0.05
        self.px, self.py, self.pyaw, _ = generate_cubic_spline(x, y, ds)




#############################################################
#
#           ----- GET CSV FILE FOR PATH POINTS ----
#
#############################################################

def get_csv(path):

    relative_path = "../../visualization/data/waypoints.csv"       # Path to data relative to script
    dirname = os.path.dirname(__file__)
    file_path = os.path.join(dirname, relative_path)
    print(file_path)
    header = ['X-axis','Y-axis']

    with open(file_path, "w", encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(path)):
            writer.writerow([str(float(path[i][0])), str(float(path[i][1]))])




#############################################################
#
#             ----- FILTER THE POINTS IN THE PATH ----
#
#############################################################


      
def simplify_path(path, filter_points=15):
    l = len(path)
    for i in range(l):
        if i%filter_points != 0 & i != l-1:
            path.remove(path[l-1-i])

    return path


