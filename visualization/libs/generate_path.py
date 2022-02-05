from inspect import _void
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
from sklearn.linear_model import LinearRegression

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


def get_random_postion(max_x, max_y):
    return (int(random.uniform(0, max_x)), int(random.uniform(0, max_y)))

def get_random_position_gaussian(p_stop, d=50):
    sample_x = np.random.normal(p_stop[0], d, 1)
    sample_y = np.random.normal(p_stop[1], d, 1)
    
    return (int(sample_x[0]), int(sample_y[0]))
  
def near_neighbour(tree, rand_pos):
    pos = [0,-1]
    for i in tree.keys():
        h = math.sqrt((rand_pos[0] - i[0])**2 + (rand_pos[1] - i[1])**2 )
        if h <= pos[1] or pos[1] == -1 and pos[1] != 0:
            pos[0] = i
            pos[1] = h
    return pos[0],pos[1]

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


def is_crossing_gree_area(near_pos, next_node, bin_map):
    crossing = True

    mid_point = (int((near_pos[0] + next_node[0])/2), int((near_pos[1] + next_node[1])/2))
    if next_node[1] < bin_map.shape[0] and next_node[0] < bin_map.shape[1] and bin_map[mid_point[1]][mid_point[0]] != 0:
        crossing = False

    return crossing

def extend_tree(img, bin_map, tree, rand_pos, step=5):

    near_pos, d = near_neighbour(tree, rand_pos) #get near pos
    next_node = select_input(rand_pos, near_pos, step)
 
    crossing = is_crossing_gree_area(near_pos, next_node, bin_map)
    if next_node[1] < bin_map.shape[0] and next_node[0] < bin_map.shape[1] and \
        next_node not in tree.keys() and bin_map[next_node[1]][next_node[0]] != 0 and crossing is False:
        tree[next_node] = [near_pos,d]
        #------------ visualization ---------------------------
        cv.line(img, near_pos, next_node, (0,0,0), thickness=1, lineType=8) 
        img[next_node[1], next_node[0]]=[0, 255, 0]
        #------------------------------------------------------

    return tree


def get_path_tree(img, tree, p_start,p_stop):

    path = []
    pos, d = near_neighbour(tree, p_stop)

    path.append(p_stop)
    while len(tree[pos]) != 0:
        path.append(tree[pos][0])
        pos = tree[pos][0]
    path.append(p_start)
    
    return path


def rrt(img, bin_map, p_start, p_stop):

   # start and end points
    p_start =  (p_start[0], p_start[1])
    p_stop  =  (p_stop[0], p_stop[1])
    # size of the gradient 
    max_img_x = bin_map.shape[1]
    max_img_y = bin_map.shape[0]
 
    # visualize init and goal positions
    cv.circle(img, p_start, 5,(0,255,0),thickness=3, lineType=8)
    cv.circle(img, p_stop, 5,(0,0,255),thickness=3, lineType=8)
    
    # --- tree inicialization
    tree = {}
    tree[p_start] =  []
    # print(tree)

    ts1 = time.time()    
    goal = False

    # ----- PARAMETERS ------

    limit = 3000
    limit_max_step = 700
   
    step = 10
    error_goal = 8

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)


    tm = []
    n_nodes = []
    iter = []


    i = 0
    while goal == False:
  
        rand_pos = get_random_postion(max_img_x, max_img_y)
        tree= extend_tree(img, bin_map,tree, rand_pos, step=step)
        k = list(tree.keys())[-1]
        distance_to_goal = math.sqrt((k[0] - p_stop[0])**2 + (k[1] - p_stop[1])**2)

        # get time stamp
        ts2 = time.time()
        tf = ts2 - ts1

        # visualization
        # cv.circle(img, rand_pos, 1,(255,0,0),thickness=1, lineType=1)
        cv.imshow("image",img)
        cv.waitKey(1)

        if i%2500 == 0 and step != 1:
            step-=1
        if distance_to_goal <= error_goal:
            goal = True

        print("iterations: ", i, " -- step: ", step, "-- number of nodes: ", len(tree))

        iter.append(i)
        n_nodes.append(len(tree))
        tm.append(tf)

        i+=1

    fig, axs = plt.subplots(2)
    fig.suptitle('BASIC RRT')
    axs[0].plot(tm, n_nodes)
    axs[0].set_ylabel("number of nodes")
    axs[0].set_xlabel("time")
    axs[1].plot(iter, n_nodes)
    axs[1].set_ylabel("number of nodes")
    axs[1].set_xlabel("iterations")
    


    print(len(iter))
    print(len(n_nodes))
    print("---------------------------")
    print("GOAL REACHED!!")
    print("limit = {} \n \
    limit_max_step = {} \n \
    step =  {} \n \
    error_goal = ".format(limit,limit_max_step, error_goal))
    print("---> time to compute the path: {}".format(tf))
    print("==================================================================================")

    path = get_path_tree(img, tree, p_start, p_stop)
    print(path)
    return path


def rrt_gaussian(img, bin_map, p_start, p_stop):

    # start and end points
    p_start =  (p_start[0], p_start[1])
    p_stop  =  (p_stop[0], p_stop[1])
    # size of the gradient 
    max_img_x = bin_map.shape[1]
    max_img_y = bin_map.shape[0]
 
    # visualize init and goal positions
    cv.circle(img, p_start, 5,(0,255,0),thickness=3, lineType=8)
    cv.circle(img, p_stop, 5,(0,0,255),thickness=3, lineType=8)
    
    # --- tree inicialization
    tree = {}
    tree[p_start] =  []
    # print(tree)

   
    ts1 = time.time()    
    goal = False



    error_goal = 8
    step = 10
    sigma = 250

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)

    tm = []
    n_nodes = []
    iter = []

    i = 0 # number of iterations of the random tree
    while goal == False:
        # if i < limit:
        if i<1000: 
            rand_pos = get_random_postion(max_img_x, max_img_y)
            tree= extend_tree(img, bin_map,tree, rand_pos, step=step)
        if i>=1000: 
            rand_pos = get_random_position_gaussian(p_stop, d=sigma)
            tree= extend_tree(img, bin_map,tree, rand_pos, step=step)
        # if i>=2000 and i<3000: 
        #     rand_pos = get_random_position_gaussian(p_stop, d=100)
        #     tree= extend_tree(img, bin_map,tree, rand_pos, step=step)
        # if i>3000: 
        #     rand_pos = get_random_position_gaussian(p_stop, d=50)
        #     tree= extend_tree(img, bin_map,tree, rand_pos, step=step)


        if i%1000 == 0 and step != 1:
            step-=1
        if i%50 == 0 and sigma != 1 and i>=1000:
            sigma-=1
            
        # print("distance to goal: ", distance_to_goal)
        k = list(tree.keys())[-1]
        distance_to_goal = math.sqrt((k[0] - p_stop[0])**2 + (k[1] - p_stop[1])**2)
        # print(distance_to_goal)

        ts2 = time.time()
        tf = ts2 - ts1

        iter.append(i)
        n_nodes.append(len(tree))
        tm.append(tf)
        print("iterations: ", i, " -- step: ", step, "-- number of nodes: ", len(tree))


        if distance_to_goal <= error_goal:
            break

        cv.circle(img, rand_pos, 1,(255,0,0),thickness=1, lineType=1)
        # exit(rand_pos)

        cv.imshow("image",img)
        cv.waitKey(1)
        i += 1


    fig, axs = plt.subplots(3)
    fig.suptitle('IMPROVED RRT')


    # define a model
    x = np.array(tm).reshape(-1, 1)
    y = np.array(n_nodes)
    model = LinearRegression().fit(x, y)
    axs[0].plot(x, model.predict(x), '-r', label='lr')
    axs[0].scatter(tm, n_nodes, s=0.1)
    axs[0].set_ylabel("number of nodes")
    axs[0].set_xlabel("time")

    x = np.array(iter).reshape(-1, 1)
    y = np.array(n_nodes)
    model = LinearRegression().fit(x, y)
    axs[1].plot(x, model.predict(x), '-r', label='lr')
    axs[1].scatter(iter, n_nodes, s=0.1)
    axs[1].set_ylabel("number of nodes")
    axs[1].set_xlabel("iterations")
    

    x = np.array(tm).reshape(-1, 1)
    y = np.array(iter)
    model = LinearRegression().fit(x, y)
    axs[2].plot(x, model.predict(x), '-r', label='lr')
    axs[2].scatter(tm,iter , s=0.2)
    axs[2].set_ylabel("iterations")
    axs[2].set_xlabel("time")
    





    # print(len(iter))
    # print(len(n_nodes))
    print("---------------------------")
    print("GOAL REACHED!!")
    # print("limit = {} \n \
    # limit_max_step = {} \n \
    # step =  {} \n \
    # error_goal = ".format(limit,limit_max_step, error_goal))
    print("---> time to compute the path: {}".format(tf))
    print("==================================================================================")


    path = get_path_tree(img, tree, p_start, p_stop)
    return path

















# def rrt(img, graph, grid, gradient_map, p_start, p_stop):

#     # grid -> all positions
#     # graph -> postions + pixels orientations

#     # start and end points
#     p_start =  (p_start[0], p_start[1])
#     p_stop  =  (p_stop[0], p_stop[1])
#     # size of the gradient 
#     max_img_x = gradient_map.shape[1]
#     max_img_y = gradient_map.shape[0]
 


#     # visualize init and goal positions
#     cv.circle(img, p_start, 5,(0,255,0),thickness=3, lineType=8)
#     cv.circle(img, p_stop, 5,(0,0,255),thickness=3, lineType=8)
    

#     # --- tree inicialization
#     tree = {}
#     tree[p_start] =  []
#     # print(tree)

    
#     K = 0 # number of iterations of the random tree
#     ts1 = time.time()    
#     goal = False


#     limit = 3000
#     limit_max_step = 700
#     step1 = 9
#     step2 = 4
#     step3 = 3
#     final_step = 2
#     error_goal = 5

#     t = time.localtime()
#     current_time = time.strftime("%H:%M:%S", t)
#     print(current_time)

#     while goal == False:
#         if K < limit:
#             rand_pos = get_random_postion(max_img_x, max_img_y)
#             if K<limit_max_step: 
#                 tree= extend_tree(img, gradient_map,tree, rand_pos, step=step1)
#             if K%2 == 0: 
#                 tree= extend_tree(img, gradient_map,tree, rand_pos, step=step2)
#             else:
#                 tree= extend_tree(img, gradient_map,tree, rand_pos, step=step3)
#         elif K>=limit:
#             rand_pos = get_random_position_gaussian(p_stop)
#             tree = extend_tree(img, gradient_map,tree, rand_pos, step=final_step)
        
#         # print("distance to goal: ", distance_to_goal)
#         k = list(tree.keys())[-1]
#         distance_to_goal = math.sqrt((k[0] - p_stop[0])**2 + (k[1] - p_stop[1])**2)
#         # print(distance_to_goal)

#         ts2 = time.time()
#         tf = ts2 - ts1

#         if distance_to_goal <= error_goal:
#             break

#         # elif tf > 20:
#         #     print("EXECUTION EXCEDED ADMISIBLE TIME")
#         #     print("==================================================================================")
#         #     exit(0)
            
       

#         # cv.circle(img, rand_pos, 1,(255,0,0),thickness=1, lineType=1)
#         # exit(rand_pos)

#         cv.imshow("image",img)
#         cv.waitKey(1)
#         K += 1
#         # time.sleep(0.01)
#         # cv.circle(img, u, 1,(0,255,0),thickness=1, lineType=8)


#     print("---------------------------")
#     print("GOAL REACHED!!")
#     print("limit = {} \n \
#     limit_max_step = {} \n \
#     step1 = {} \n \
#     step2 = {} \n \
#     step3 =  {} \n \
#     final_step = {} \n \
#     error_goal = ".format(limit,limit_max_step, step1, step2, step3, final_step, error_goal))
#     print("---> time to compute the path: {}".format(tf))
#     print("==================================================================================")
    

#     exit(0)


#     path = get_path_tree()
      
#     return path


