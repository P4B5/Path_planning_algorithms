from tracemalloc import stop
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from libs.generate_roadmaps import binary_threshold_image
import os

# =========================================================
# FUNCTION TO SELECT POINTS ON THE DESIRED ROADMAP INTERACTIVELY.
# Parameters: 
#   - img_path:str, absolute path to the input_image. Initial None.
#   - input_image:np.ndarray, the preloaded input image. Initial None.
# Returns:
#   - start_pt, stop_pt, List, start and stop point of the desired path
def interactive_select_start_end(img_path:str = None, input_img:np.ndarray = None):
    # Assert that input is given
    if input_img is None: 
        assert img_path != None, "Input not defined for fetching endpoints"
        # Read image you want to converge to a roadmap
        orig_image = cv.imread(img_path)

    if img_path is None: 
        assert input_img.all() != None, "Input not defined for fetching endpoints"
        # Use input image
        orig_image = input_img

    # Calculate the binary image of the road:
    bin_img = binary_threshold_image(input_img=orig_image)
    plt.title('select two points in the road', loc='right')
    plt.imshow(orig_image, cmap='gray')


    # Select points until good endpoints are chosen
    ok_points = False
    while not ok_points: 
        # Get user input
        pts = plt.ginput(2)
        start_pt = [int(pts[0][0]), int(pts[0][1])]
        stop_pt = [int(pts[1][0]), int(pts[1][1])]

        # IF ONE EVER NEEDS TO TROUBLESHOOT POINT-SELECTION: uncomment
        #fig1, ax1 = plt.subplots()
        #ax1.imshow(orig_image)
        #ax1.plot(start_pt[0], start_pt[1], marker="o", markersize=2)
        #ax1.plot(stop_pt[0], stop_pt[1], marker="o", markersize=2)

        # Make sure two points were chosen
        if start_pt is None or stop_pt is None: 
            print("Select both a start and end point")
            continue

        # Make sure points are located on roads. Note that x/y is inverted in np arrays
        if bin_img[start_pt[1],start_pt[0]] != 255: 
            print("Start point must be on a road")
            continue

        if bin_img[stop_pt[1],stop_pt[0]] != 255: 
            print("Stop point must be on a road")
            continue

        ok_points = True

    plt.close()
    # Return points
    return start_pt, stop_pt

# FUNCTION PLOTTING THE CALCULATED PATH ON A MAP OF CHOICE WITH START/END-POINTS.
# Parameters: 
#   - input_map:np.ndarray, input map.
#   - path:List, the path from p_start to p_stop.
#   - p_start:tuple, starting point. 
#   - p_stop:tuple, endpoint. 
# Returns:
#   - None 
def display_path(input_map, path, p_start, p_stop):
    # Initialize a subplot to plot everything together
    fig1, ax1 = plt.subplots()
    # plot the desired map
    
    ax1.imshow(input_map)
   
    # plot the start/endpoint
    ax1.scatter(x=p_start[0],y=p_start[1])
    ax1.scatter(x=p_stop[0], y=p_stop[1])
    # plot the path
    ax1.scatter(*zip(*path), c='r', s=1)
    plt.xlabel('xlabel')
    plt.show()

# =========================================================
# FUNCTION TO CHANGE THE COORDINATES FROM IMAGE FRAME TO WORLD FRAME
# Parameters: 
#   - image frame coordinates
# Returns:
#   - world frame coordinates

def image_to_world_coordinates(coordinates):
    WFC_X = 433 # origin x coordinate of the world frame in the picture
    WFC_Y = 194 # origin y coordinate of the world frame in the picture
    gps_x = (coordinates[0] - WFC_X) # relative x coodinate in world frame -> gps x
    gps_y = -(coordinates[1] - WFC_Y) # relative y coodinate in world frame -> gps y
    return [gps_x, gps_y]

# =========================================================
# FUNCTION TO CHANGE THE COORDINATES FROM WORLD FRAME TO IMAGE FRAME
# Parameters: 
#   - world frame coordinates
# Returns:
#   - image frame coordinates

def world_to_image_coordinates(coordinates):
    WFC_X = 433 # origin x coordinate of the world frame in the picture
    WFC_Y = 194 # origin y coordinate of the world frame in the picture
    img_x = (coordinates[0] + WFC_X) # relative x coodinate in world frame -> gps x
    img_y = -(coordinates[1] - WFC_Y) # relative y coodinate in world frame -> gps y
    return [img_x, img_y]


def get_gps_coordinates(coordinates):
    return image_to_world_coordinates(coordinates)


# ==============================================================
# ==================JUST A LIL TEST AREA========================
# ==============================================================

# Load data and calculate absolute path
# relative_path = "/home/pabs/ROB_autonomous_car/navigation/data/IST_grey.PNG"        # Path to data relative to script
# dirname = os.path.dirname(__file__)
# image_path = os.path.join(dirname, relative_path)

# # Define original image
# orig_image = cv.imread(image_path)

# # Running code for test

# start_pt, stop_pt = interactive_select_start_end(img_path=image_path)
# start_pt_wf = image_to_world_coordinates(start_pt)
# stop_pt_wf = image_to_world_coordinates(stop_pt)
# print("IMAGE FRAME COORDINATES: ", start_pt, stop_pt)
# print("WORLD FRAME COORDINATES: ", start_pt_wf, stop_pt_wf)
