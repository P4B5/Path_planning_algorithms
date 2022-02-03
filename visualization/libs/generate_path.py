from webbrowser import get
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import math
from queue import PriorityQueue
from typing import List

from scipy import rand
from libs.RDP import rdp              # Copyright (c) 2019 Sean Bleier
import random
import time

# =========================================================
# FUNCTION TO TRANSFORM A BINARY MAP INTO A GRADIENT MAP. THAT IS, ALL ROADS ON THE MAP
# WILL BE EVALUATED: THE INSIDE OF THE ROAD IS RATED HIGHER THAN THE OUTSIDE. USED WITH
# PATH-PLANNING ALGORITHM TO AVOID CURBES ECT.
# Parameters: 
#   - bin_img:np.ndarray, input map. Initial None.
#   - morphology_kernel:np.ndarray, kernel used for erosion. Initial np.ones((3,3), np.uint8).
# Returns:
#   - gradient_map:np.ndarray, map with graded roads.
def generate_gradient_map(bin_image:np.ndarray, curbe_width:int = 5,morphology_kernel:np.ndarray = np.ones((3,3), np.uint8)):
    gradient_map = np.zeros(shape=bin_image.shape, dtype=np.uint8)
    
    lost_roads = bin_image.copy()
    erosion_table = []
    erosion_amount = curbe_width
    while erosion_amount >= 0:
        eroded_img = cv.erode(lost_roads, morphology_kernel, iterations = erosion_amount)
        reconstructed_img = cv.dilate(eroded_img, morphology_kernel, iterations = erosion_amount)

        lost_roads = cv.subtract(lost_roads, reconstructed_img)
        erosion_table.append(reconstructed_img)
        erosion_amount -= 1

    for i in range(len(erosion_table)):
        prev_road_map, curr_road_map = erosion_table[i], erosion_table[i]
        for j in range(curbe_width-i):
            curr_road_map = cv.erode(curr_road_map, morphology_kernel, iterations = 1)
            edge = cv.subtract(prev_road_map, curr_road_map)
            gradient_map = cv.add(gradient_map, edge*((curbe_width-i)-j))

            prev_road_map = curr_road_map

        gradient_map = cv.add(gradient_map, curr_road_map)

    return gradient_map

# =========================================================
# FUNCTION CONVERTING A 2D-ARRAY TO A WEIGHTED GRAPH OF THE NON-ZERO ELEMENTS IN THE GRID. EACH NODE
# KEEPS TRACK OF IN WHICH OF 8 DIRECTIONS ANOTHER NODE IS PRESENT. 
# Parameters: 
#   - input_map:np.ndarray, input map. Initial None.
# Returns:
#   - graph:dict, dictionary with keys = pixels input_map and value = neighbors of node.
#   - grid:List, list of nonzero points in map. Essentially list of nodes in the graph. 
def generate_graph_and_grid(input_map:np.ndarray):
    # Initialize graph and possible directions
    graph = {}
    direction_mapping = [['NW', 'N', 'NE'], 
                        ['W', 'ME', 'E'],
                        ['SW', 'S', 'SE']]
    # Find all nonzero indexes in the map:
    non_zero_x, non_zero_y = np.nonzero(input_map)

    # Pad image with 0s for neighborhood-detection
    padded_map = cv.copyMakeBorder(input_map.copy(), 1, 1, 1, 1, cv.BORDER_CONSTANT, 0)

    # Iterate thorugh all the non-zero points
    for elem in range(0, len(non_zero_x)):
        row = non_zero_x[elem]
        col = non_zero_y[elem]
        # define possible directions a neighbor can exist from a node
        neighbors = {'NW':0, 'N':0, 'NE':0, 'W':0, 'E':0, 'SW':0, 'S':0, 'SE':0}

        # find which neighboring pixels exist in a 3x3 grid
        for x in range(-1, 2):
            for y in range(-1, 2):
                # nodes ignore detecting themselves as neighbors
                if (x == 0 and y == 0): 
                    continue
                # if neighboring element is nonzero in the padded map: add as a neighbor
                if padded_map[1+row+x,1+col+y] != 0:
                    neighbors[direction_mapping[x+1][y+1]] = 1

        graph[(row, col)] = neighbors

    # generate the grid by extracting the keys of the graph    
    grid = graph.keys()
    return graph, grid

# =========================================================
# THE HEURISTIC OF THE A* ALGORITHM.  
# Parameters: 
#   - p1/p2:tuples, input points.
# Returns:
#   - manhattan-distance between p1 and p2. Easy, quick distance-estimate.
def heuristic(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    # return abs(x1-x2) + abs(y1-y2) #manhattan distance
    return math.sqrt((x1-x2)**2 + (y1-y2)**2) #euclidean distance

# IMPLEMENTATION OF THE A* ALGORITHM, FINDING THE SHORTEST PATH BETWEEN TWO POINTS IN A GRAPH
# EFFICIENTLY. 
# Parameters: 
#   - graph:dict, input graph, consisting of pixel_values and possible neighbors. 
#   - grid:List, input grid, list of all points in the graph. 
#   - p_start:tuple, desired starting point of the path.
#   - p_stop:tuple, desired goal-point of the path. 
# Returns:
#   - path:List, a list of points spanning the optimal path from p_start to p_stop
def a_star(graph, grid, gradient_map, p_start, p_stop):
    # Format the input_points, to match numpy row/col-conventions
    p_start = (p_start[1], p_start[0])
    p_stop = (p_stop[1], p_stop[0])

    # Declare initial g and f scores
    g_score = {cell:float('inf') for cell in grid}
    g_score[p_start] = 0
    f_score = {cell:float('inf') for cell in grid}
    f_score[p_start] = heuristic(p_start, p_stop)

    # Generate priority queue and put the startpoint in there
    open = PriorityQueue()
    open.put((heuristic(p_start,p_stop), heuristic(p_start,p_stop), p_start))

    # Generate dictionary to keep track of the last traversed from at each node. Traversed 
    # backwards to find the path in the end. 
    last = {}

    # Iterate through all children of the starting point until the priority_queue is empty
    # or p_stop is reached
    while not open.empty():
        currCell = open.get()[2]
        if currCell == p_stop:
            break

        # iterate through all potential children (neighbors) of the current cell
        for d in ['NW', 'N', 'NE', 'W','E' ,'SW', 'S', 'SE']:
            if graph[currCell][d] == 1:
                if d == 'NW':
                    childCell = (currCell[0]-1, currCell[1]-1)
                if d == 'N':
                    childCell = (currCell[0]-1, currCell[1])
                if d == 'NE':
                    childCell = (currCell[0]-1, currCell[1]+1)
                if d == 'W':
                    childCell = (currCell[0], currCell[1]-1)
                if d == 'E':
                    childCell = (currCell[0], currCell[1]+1)
                if d == 'SW':
                    childCell = (currCell[0]+1, currCell[1]-1)
                if d == 'S':
                    childCell = (currCell[0]+1, currCell[1])
                if d == 'SE':
                    childCell = (currCell[0]+1, currCell[1]+1)
                
                # calculate the g- and f-scores of the child
                temp_g_score = g_score[currCell]+1
                #temp_f_score = temp_g_score + heuristic(childCell,p_stop)
                weight_child = gradient_map[childCell[0],childCell[1]]
                temp_f_score = temp_g_score + weight_child*heuristic(childCell,p_stop)

                # if the scores are better than the existing values, replace them. 
                if temp_f_score < f_score[childCell]:
                    g_score[childCell] = temp_g_score
                    f_score[childCell] = temp_f_score
                    #open.put((temp_f_score, heuristic(childCell, p_stop), childCell))
                    open.put((temp_f_score, weight_child*heuristic(childCell, p_stop), childCell))
                    last[childCell] = currCell

    # Generate the path by iterating through last backwards
    path = {}
    cell = p_stop
    while cell != p_start:
        # again, invert points for ease of implementation with numpy/matplotlib later

        try:
            path[last[cell]] = (cell[1], cell[0])
            cell = last[cell]
        except KeyError:
            print("ERROR: IMPOSSIBLE TO JOIN THE SELECTED POINTS")
            return None
        

    # return the path as a list of points
    return list(path.values())

# FUNCTION SIGNIFICANTLY SIMPLIFYING THE PATH BY UTILIZING THE RDP ALGORITHM 
# (https://www.wikiwand.com/en/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm). FUNCITON
# EXISTS FOR READABILITY
# Parameters: 
#   - path:List, the complete, unfiltered path.
#   - thresh:float, how much one can deviate from the road. Higher = cruder road.
# Returns:
#   - path:List, the simplified path with only necessary points. 
def simplify_path(input_path:List, thresh:float = 1.0):
    path = rdp(input_path, thresh)
    return path

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
    ax1.scatter(x=p_start[0],y=p_start[1],s=80)
    ax1.scatter(x=p_stop[0], y=p_stop[1], s=80)
    # plot the path
    ax1.scatter(*zip(*path), c='r', s=1)

    plt.show()






###################################################################
#
#
#                           RTT
# 
#               Rapidly-exploring random trees
# 
# 
# 
# ##################################################################




# =========================================================
# RTT  - Rapidly-exploring random tree
# Parameters: 
#   - graph:dict, input graph, consisting of pixel_values and possible neighbors. 
#   - grid:List, input grid, list of all points in the graph. 
#   - p_start:tuple, desired starting point of the path.
#   - p_stop:tuple, desired goal-point of the path. 
# Returns:
#   - tree: a graph with all the nodes and conections which reaches the final position


def display_path(point):
    relative_path = "..\data\IST_grey_2.PNG"        # Path to data relative to script
    dirname = os.path.dirname(__file__)
    image_path = os.path.join(dirname, relative_path)

    # Define original image
    orig_image = cv.imread(image_path)

    # Initialize a subplot to plot everything together
    fig1, ax1 = plt.subplots()
    # plot the desired map
    ax1.imshow(orig_image)
    # plot the start/endpoint
    ax1.scatter(x=point[0],y=point[1],s=80)
    # ax1.scatter(x=p_stop[0], y=p_stop[1], s=80)
    # plot the path
    # ax1.scatter(*zip(*path), c='r', s=1)

    plt.show()
    


def get_random_postion(max_x, max_y):
    print("max x: ", max_x)
    print("max y: ", max_y)
    return (int(random.uniform(0, max_x)), int(random.uniform(0, max_y)))


def get_random_position_intelligent(p_stop):
    sample_x = np.random.normal(p_stop[0], 60, 1)
    sample_y = np.random.normal(p_stop[1], 60, 1)
    
    return (int(sample_x[0]), int(sample_y[0]))
  

def near_neighbour(tree, rand_pos):
    pos = [0,-1]
    for i in tree.keys():
        h = math.sqrt((rand_pos[0] - i[0])**2 + (rand_pos[1] - i[1])**2 )
        if h <= pos[1] or pos[1] == -1 and pos[1] != 0:
            pos[0] = i
            pos[1] = h
    return pos[0]


def select_input(rand_pos, near_pos, step=2):
    next_node = 0

    ang = math.atan2(near_pos[0] - rand_pos[0],near_pos[1]-rand_pos[1])
    ang = ang * 180 / math.pi

    
    if  ang < 67.5 and ang > 22.5: #'NE'
        next_node = (near_pos[0]-step, near_pos[1]-step)
    
    if ang < 22.5 and ang > -22.5: #E
        next_node = (near_pos[0], near_pos[1]-step)

    if  ang < -22.5 and ang > -67.5: #'SE'
        next_node = (near_pos[0]+step, near_pos[1]-step)

    if  ang < 112.5 and ang > 67.5: #'N'
        next_node = (near_pos[0]-step, near_pos[1])
        
    if  ang < -67.5 and ang > -112.5: #'S'
        next_node = (near_pos[0]+step, near_pos[1])

    if  ang < 157.5 and ang > 112.5: #'NW'
        next_node = (near_pos[0]-step, near_pos[1]+step)
        
    if  ang < -157.5 or ang > 157.5: #'W':
        next_node = (near_pos[0], near_pos[1]+step)
        
    if  ang < -112.5 and ang > -157.5: #'SW':
        next_node = (near_pos[0]+step, near_pos[1]+step)
        
    return next_node



def new_state():


    return None




def init_tree(graph):
    tree= {}
    neighbors = {'NW':0, 'N':0, 'NE':0, 'W':0, 'E':0, 'SW':0, 'S':0, 'SE':0}
    for i in graph.keys():
        tree[i] = neighbors

    return tree


def extend_tree(img, gradient_map, tree, rand_pos, step=5):


    near_pos = near_neighbour(tree, rand_pos) #get near pos
    next_node = select_input(rand_pos, near_pos, step)
    

    if gradient_map[next_node[1]][next_node[0]] != 0:
        tree[near_pos].append(next_node)
        tree[next_node] = []

        cv.line(img, near_pos, next_node, (0,0,0), thickness=1, lineType=8)
        img[next_node[1], next_node[0]]=[0, 255, 0]

    return tree
    # new_pos = new_state(near_pos, u)
    # if free_path(near_pos, new_pos):
    #     tree.add_node

    # return tree_graph


def rtt(img, graph, grid, gradient_map, p_start, p_stop):

    # grid -> all positions
    # graph -> postions + pixels orientations

    # start and end points
    p_start =  (p_start[0], p_start[1])
    p_stop  =  (p_stop[0], p_stop[1])
    # size of the gradient 
    max_img_x = gradient_map.shape[1]
    max_img_y = gradient_map.shape[0]
 


    # visualize init and goal positions
    cv.circle(img, p_start, 5,(0,255,0),thickness=3, lineType=8)
    cv.circle(img, p_stop, 5,(0,0,255),thickness=3, lineType=8)
    

    # --- tree inicialization
    tree = {}
    tree[p_start] =  []
    print(tree)

    
    K = 0 # number of iterations of the random tree

    while True:
        if K < 2000:
            rand_pos = get_random_postion(max_img_x, max_img_y)
            tree = extend_tree(img, gradient_map,tree, rand_pos, step=9)
        elif K>=2000:
            rand_pos = get_random_position_intelligent(p_stop)
            tree = extend_tree(img, gradient_map,tree, rand_pos, step=2)
        print("rand pos: ",rand_pos)
        cv.circle(img, rand_pos, 1,(255,0,0),thickness=1, lineType=1)
        # exit(rand_pos)

      
        cv.imshow("image",img)
        cv.waitKey(1)
        K += 1
        # time.sleep(0.01)
        
        
        # cv.circle(img, u, 1,(0,255,0),thickness=1, lineType=8)

      
  
    
      
    return tree



















###################################################################
#
#
#                           RTT*
# 
#               Rapidly-exploring random trees
# 
#       CHARACTERISTICS:
# 
# ##################################################################

# =========================================================
# RTT  - Rapidly-exploring random tree star
# Parameters: 
#   - graph:dict, input graph, consisting of pixel_values and possible neighbors. 
#   - grid:List, input grid, list of all points in the graph. 
#   - p_start:tuple, desired starting point of the path.
#   - p_stop:tuple, desired goal-point of the path. 
# Returns:
#   - path:List


def rtt_star(graph, grid, gradient_map, p_start, p_stop):



    return None



###################################################################
#
#
#                       UNINFORMED  RTT*
# 
#               Rapidly-exploring random trees
# 
#       CHARACTERISTICS:
# 
# ##################################################################

# =========================================================
# RTT  - Rapidly-exploring random tree star
# Parameters: 
#   - graph:dict, input graph, consisting of pixel_values and possible neighbors. 
#   - grid:List, input grid, list of all points in the graph. 
#   - p_start:tuple, desired starting point of the path.
#   - p_stop:tuple, desired goal-point of the path. 
# Returns:
#   - path:List