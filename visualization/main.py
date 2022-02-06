

###############################################################################
#
#                       ---   AUTONOMOUS CAR  ---
#
# 
#              LAB 2 - ROBOTICS COURSE 2021/2022 -  IST LISBON
# 
#                   Valentin Fuchs    -- ist1101370
#                   Jonas Brodmann    -- ist1101416 
#                   Pablo Castellanos -- ist1101777
#                   Maxime Roedele    -- ist1101520
#                   Aron Berner       -- ist1101793
#       
#
#       Install requeriments:
#       pip install -r requirements.txt
# 
#       Execution:
#       python3 main.py      
#                   
#       simulation visualization based on the open-source repository: 
#       https://github.com/winstxnhdw/KinematicBicycleModel
#
#                                              
###############################################################################

from libs.setup_simulation import *
# from visualization.libs.RDP import distance


# Simulation paramters
car_velocity = 10                   #max velocity [m/s], 10 m/s is default and path is optimized for that
use_real_sensors = False            #use real sensor model (false = ideal position data is used to control the car, otherwise GPS etc. is used) -- this function is currently nor working
energy_contingent = 36*(10**3)       #energy contingent [kJ]
distance_factor = 0.56              #distance in meters for each pixel



#############################################################
#
#            ----- MAIN ANIMATION PROGRAM ---
#
#############################################################

def main():
    
    sim = Simulation()
    path = Path()

    car = Car(path.px[0], path.py[0], path.pyaw[0], path.px, path.py, path.pyaw, sim.dt, energy_contingent, car_velocity)
    desc = Description(car.overall_length, car.overall_width, car.rear_overhang, car.tyre_diameter, car.tyre_width, car.axle_track, car.wheelbase)

    interval = sim.dt * 10**3

    fig = plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')
    
    relative_path = "../visualization/data/IST_grey_2.PNG"        # Path to data relative to script
    dirname = os.path.dirname(__file__)
    image_path = os.path.join(dirname, relative_path)

    orig_image = cv.imread(image_path)
    ax.imshow(orig_image)
 
    # uncoment this line to plot a trajectory
    ax.plot(path.px, path.py, '--', color='gold')

    annotation = ax.annotate(f'{car.x:.1f}, {car.y:.1f}', xy=(car.x, car.y + 5), color='black', annotation_clip=False)
    target, = ax.plot([], [], '+r')

    outline, = ax.plot([], [], color=car.colour)
    fr, = ax.plot([], [], color=car.colour)
    rr, = ax.plot([], [], color=car.colour)
    fl, = ax.plot([], [], color=car.colour)
    rl, = ax.plot([], [], color=car.colour)
    rear_axle, = ax.plot(car.x, car.y, '+', color=car.colour, markersize=2)

    plt.grid()

  
    def animate(frame):

        # Camera tracks car
        ax.set_xlim(car.x - sim.map_size_x, car.x + sim.map_size_x)
        ax.set_ylim(car.y + sim.map_size_y, car.y - sim.map_size_y)

        # Drive and draw car
        car.drive(use_real_sensors)
        outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = desc.plot_car(car.x, car.y, car.yaw, car.delta)
        outline.set_data(*outline_plot)
        fr.set_data(*fr_plot)
        rr.set_data(*rr_plot)
        fl.set_data(*fl_plot)
        rl.set_data(*rl_plot)
        rear_axle.set_data(car.x, car.y)

        # Show car's target
        target.set_data(path.px[car.target_id], path.py[car.target_id])

        # Annotate car's coordinate above car
        car.x_m= car.x*distance_factor
        car.y_m = car.y*distance_factor
        annotation.set_text(f'{car.x_m:.1f}, {car.y_m:.1f}')
        annotation.set_position((car.x, car.y + 5))

        # labels for simulation information
        plt.title(f'Simulation time: {sim.dt*frame:.2f}s', loc='right')
        
        distance_to_target = (math.sqrt((car.x-path.px[-1])**2 + (car.y-path.py[-1])**2))* distance_factor
        plt.xlabel(f'Speed: {car.velocity_x:.2f} m/s {car.velocity_x*3.6:.2f} km/h --- Energy left: {car.energy_left:.2f} kJ --- Energy used: {car.consumed_energy_since_start:.2f} kJ --- Distance to target: {distance_to_target:.2f} m' , loc='left')
        
        if distance_to_target <= 0.9:
            print("---> SUCCESS: THE CAR REACHED THE FINAL POSITION")
            time.sleep(5)
            exit(0)

        return outline, fr, rr, fl, rl, rear_axle, target,

    _ = FuncAnimation(fig, animate, frames=sim.frames, interval=interval, repeat=sim.loop)

    plt.show()



#############################################################
#
#           ----- GET THE PATH TO DRIVE  -----
#
#############################################################

def get_path():          

   # Load data and calculate absolute path
    print("LOADING DATA:")
    relative_path = "../visualization/data/IST_grey_2.PNG"        # Path to data relative to script
    dirname = os.path.dirname(__file__)
    image_path = os.path.join(dirname, relative_path)

    # Define original image
    orig_image = cv.imread(image_path)

    # Get binary image
    print("PREPROCESSING OF DATA IN PROGRESS...")
    bin_image = binary_threshold_image(input_img=orig_image)
    bin_image[bin_image == 255] = 1
    print("original image size main: ", orig_image.shape)

    # Gradient erosion
    gradient_map = generate_gradient_map(bin_image)
   
    print("PREPROCESSING COMPLETE!\n================================\n")

    # Convert map to dictionary
    graph, grid = generate_graph_and_grid(gradient_map)

    path = None
    #selects the points until the path can be calculated
    while path is None:
        # Take in start and stop_points
        print("SELECT START/END-POINTS")
        p_start, p_stop = interactive_select_start_end(input_img=orig_image)
        print("start:", p_start, " stop:",  p_stop)
        # Try to calculate path
        print("CALCULATING PATH...")
        
        # path = rrt(orig_image,bin_image, p_start, p_stop)
        path = rrt_gaussian(orig_image,bin_image, p_start, p_stop)
        # path = a_star(graph, grid,gradient_map, p_start, p_stop)
        # exit(0)

    #reverse the path points to go from initial to final point
    path.reverse()

    # Simplify the path 
    print("SIMPLIFYING PATH")
    path = simplify_path(path, filter_points=4)
   
    # --- uncomment to visualize the path ---
    display_path(orig_image, path, p_start, p_stop)
    get_csv(path) #save the path points in a csv file




#############################################################
#
#                   ----- MAIN ----
#
#############################################################


if __name__ == '__main__':
    get_path() #-- get the shortest path from point A to point B
    main()     #-- main program
