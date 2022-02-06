# Comparison of search strategies for path planning ofmobile robots


## Authors
Jonas Brodmann  -  Karlsruhe Institute of Technology  
Valentin Fuchs - Karlsruhe Institute of Technology  
Pablo Castellanos  - Rey Juan Carlos University

## Abstract
This repository is relative to the paper which deals with the comparison of search algorithm for path planning of autonomous movingvehicles.  It aims at giving an overview on different popular search strategies and shows exemplary ina simulation environment the strength and weaknesses of A* and Rapidly-exploring random tree algo-rithms inside the campus of Instituto Superior Tecnico in Lisbon, Portugal.

## Map
<img src="https://github.com/P4B5/Path_planning_algorithms/docs/evaluation_map.png" width="400" height="200"/>


## Performance of the Trees

### Basic RRT

<img src="https://github.com/P4B5/Path_planning_algorithms/docs/random_points_improved.png" width="400" height="200"/>

### Improved RRT
<img src="https://github.com/P4B5/Path_planning_algorithms/docs/random_points_basic.png" width="400" height="200"/>


## Results

### A*

<img src="https://github.com/P4B5/Path_planning_algorithms/docs/plot_Astar_complexity.png" width="400" height="200"/>


### RRT

<img src="https://github.com/P4B5/Path_planning_algorithms/docs/plot_RRT_basic_complexity.png" width="400" height="200"/>


### Improved RRT

<img src="https://github.com/P4B5/Path_planning_algorithms/docs/plot_Astar_complexity.png" width="400" height="200"/>


## Setup and run the environment
- install requeriments: `pip install -r requirements.txt`
- execute the program: `python3 animate.py`
