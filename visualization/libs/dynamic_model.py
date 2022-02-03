#!/usr/bin/env python

from cmath import sqrt
import math
import numpy as np
import os
from libs.stanley_controller import StanleyController
from libs.sensor_models import sensor_models
# import kinematic_model
from libs.kinematic_model import KinematicBicycleModel

class Car():

    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, dt, energy_contingent = 36*10**6, max_velocity=10):

        # Model parameters
        self.x = init_x                                 #init x position [m]
        self.y = init_y                                 #init y position [m]
        self.yaw = init_yaw                             #init yaw angle (orientation of the car) [rad]
        self.velocity_x = 0.1                           #init velocity of the car [m/s] (shouldn't be zero, as the model equtions create singularities there)
        self.velocity_y = 0.0                           #velocity orthagonal to the cars moving direction (moving direction: along x) [m/s]
        self.delta = 0.00                               #init steering angle [rad]
        self.wished_delta = 0.00                        #steering angle asked for by the controller [rad]
        self.Lf = 1.1                                   #distance center of front tire to center of gravity [m],                                                value according to task Lab 2
        self.Lr = 1.1                                   #distance center of rear tire to center of gravity [m],                                                 value according to task Lab 2
        self.wheelbase = self.Lr + self.Lf
        self.max_steer = np.deg2rad(33)                 #max steering angle [rad],                                                                              value according to http://street.umn.edu/VehControl/javahelp/HTML/Definition_of_Vehicle_Heading_and_Steeing_Angle.htm
        self.dt = dt                                    #time interval between controller and model update points
        self.c_r = 0.01                                 #friction coefficient carmodel,                                                                         according to https://www.engineeringtoolbox.com/rolling-friction-resistance-d_1303.html 
        self.c_a = 0.25                                 #aero coefficient (cw-value),                                                                           according to https://en.wikipedia.org/wiki/Automobile_drag_coefficient 
        self.Iz = 2250                                  #inertia (Trägheit) yaw,                                                                                according to several sources like http://archive.sciendo.com/MECDC/mecdc.2013.11.issue-1/mecdc-2013-0003/mecdc-2013-0003.pdf
        self.mass = 810                                 #mass car [N], according to Lab 2 
        self.omega = 0.0                                #steering angle velocity [rad/s] (also called steering rate)
        self.c_alphafront = 65100                       #tires cornering stiffness coefficient front,                                                           according to https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4791051
        self.c_alpharear = 54100                        #tires cornering stiffness coefficient rear
        self.frontExtension = 1.55 * 1.5                #front area of the car (hight is assumed to be 1.5 m), used to calculate the aero resistance of the car [m^2]
        self.max_velocity_x = max_velocity              #velocity the car should reach in the simulation                                                
        self.efficiency = 35                            #efficiency of the car engine [%],                                                                      according to https://www.aaa.com/autorepair/articles/how-efficient-is-your-cars-engine                                                                                                                                              
        self.max_acc = 3                                #max acceleration of the car [m/s^2],                                                                   according to https://hypertextbook.com/facts/2001/MeredithBarricella.shtml                                                                         
        self.max_braking = 0.47 * 9.81                  #max decerlation while braking [m/s^2],                                                                 according to https://copradar.com/chapts/references/acceleration.html
        self.max_steering_velocity = 13                 #max steering velocity [rad/s],                                                                         avarage according to https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwipn--kzs_1AhUBahQKHW9CCGcQFnoECAoQAw&url=https%3A%2F%2Fwww-nrd.nhtsa.dot.gov%2Fpdf%2Fesv%2Fesv16%2F98s2w35.pdf&usg=AOvVaw2YpeUli4OmkRFSYlffe92A
        self.consumed_energy_cycle = 0                  #consumed energy in last simulation cycle [J]
        self.consumed_energy_since_start = 0            #consumed energy since start of the simulation [J]
        self.energy_contingent = energy_contingent      #energy to perform the task [kJ]                                                                        given in the beginning of the task, 36 MJ = 1 L gasoline
        self.energy_left = 0.0                          #energy left since start [kJ]
        self.no_gas = False                             #becomes true if car rans out of gas
        self.car_reached_endpoint = False
        self.car_stopped = False

        # Tracker parameters
        self.px = px                                    #created path point x-coodinate [m]
        self.py = py                                    #created path point y-coodinate [m]                                    
        self.pyaw = pyaw                                #created path orientation [m]
        self.k = 3.75                                   #obtained from https://github.com/winstxnhdw/KinematicBicycleModel, implemented for a kinematic model there. Parameter were slightly modified to fit the dynamic model
        self.ksoft = 1.0
        self.kyaw = 0.01
        self.ksteer = 0.1
        self.crosstrack_error = None
        self.target_id = None

        # Description parameters
        self.overall_length = 3.332                         #according to lab 2 description
        self.overall_width = 1.508                          #according to lab 2 description
        self.tyre_diameter = 0.5136                         #according to lab 2 description
        self.tyre_width = 0.2032                            #assumed according to https://github.com/winstxnhdw/KinematicBicycleModel
        self.axle_track = 1.662                             #assumed 
        self.rear_overhang = (self.overall_length - self.wheelbase) / 2
        self.colour = 'black'

        #Initialize Stanley Controller
        self.tracker = StanleyController(self.k, self.ksoft, self.kyaw, self.ksteer, self.max_steer, self.wheelbase, self.px, self.py, self.pyaw)
        #Initialize Dynamic Bicycle Model
        self.dbm = DynamicBicycleModel(self.Lr, self.Lf, self.max_steer, self.dt, self.c_r, self.c_a, self.c_alphafront, self.c_alpharear, self.Iz, self.mass, self.frontExtension, self.efficiency, self.max_acc)
    	#Initialize Kinematic Bicycle Model
        self.dkm = KinematicBicycleModel(self.wheelbase, self.max_steer, self.dt, self.max_acc)

    #Function to execute the car model
    def drive(self, use_real_sensors = False):

        #Use real sensors (or in this case: modeled sensors, that add an accuracy to the control)
        if(use_real_sensors == True):
            #get sensor data from GPS, steering wheel encoder and velocity encoder
            x_GPS, y_GPS, v_GPS, heading_GPS = sensor_models.sensor_models.model_GPS(self.x, self.y, self.velocity_x, self.yaw)
            delta_encoder = sensor_models.sensor_models.model_steering_encoder(self.delta)
            #here: velocity can be used directly because of model. Normally (so for a real car) it is necessary to obtain the velocity from a encoder, that measures the angular velocity. This can be done using v=2⋅π⋅rT
            velocity_encoder = sensor_models.sensor_models.model_velocity_encoder(self.velocity_x)

            #Use the kinematic model to get the new velocitiesand position using the given sensor data (velocity and steering angle)
            x_sensor, y_sensor, yaw_sensor = KinematicBicycleModel.kinematic_model(x_GPS, y_GPS, heading_GPS, self.max_velocity_x, velocity_encoder, delta_encoder)

            #Filter and merge given data

                
            #Use data as input for the model control
            self.wished_delta, self.target_id, self.crosstrack_error = self.tracker.stanley_control(x_sensor, y_sensor, yaw_sensor, velocity_encoder, delta_encoder)

        #else: use ideal values, so without uncertainty
        else:
            #Find control values according to Stanley Controller
            self.wished_delta, self.target_id, self.crosstrack_error = self.tracker.stanley_control(self.x, self.y, self.yaw, self.velocity_x, self.delta)

        sum_distance = 0.0
        distance_to_target = 0.0
        deceleration_way = 0.0
        avrg_dist_pthpts = 0.0

        #reduce velocity if car gets close to endpoint 
        #1. get the avarage distance between two planned points (take last 5 points of path)
        number_points = 5
        i = len(self.px)-(number_points+2)

        if i<0:
            print("Path is too short to accelerate correctly")
            ride_slow = True
            

        while i < len(self.px)-2:
            sum_distance += np.sqrt(np.square(self.px[i]-self.px[i+1])+np.square(self.py[i]-self.py[i+1]))
            i += 1

        avrg_dist_pthpts = sum_distance/number_points

        #2. Calculate deceleration way (v0-1/2*a*t^2)
        deceleration_time = self.velocity_x/self.max_braking
        deceleration_way = -1/2 * self.max_braking * np.square(deceleration_time) + self.velocity_x * deceleration_time
        braking_point = np.ceil(deceleration_way/avrg_dist_pthpts)

        distance_to_target = np.sqrt(np.square(self.px[len(self.px)-1]-self.x)+np.square(self.py[len(self.py)-1]-self.y))

        braking = False
        #brake down to minimal velocity (we could also let the car roll out but here i think it is best to brake as late as possible to complete the path in a reasonable time)
        #also: in curves with higher velocities that might lead to issues if the car is passing the target point before reaching it, as only target distance is evaluated
        if((distance_to_target <= (deceleration_way + 5)) & (self.velocity_x > 1.0)):
            #brake
            print("braking")
            braking = True
            #self.max_velocity_x = self.max_velocity_x - self.max_braking * self.dt
            self.max_velocity_x = 1.0
    
        if(distance_to_target <= 1):
            self.velocity_x = 0.0
            self.max_velocity_x = 0.0
            self.car_reached_endpoint = True

        if(self.car_reached_endpoint == False & self.car_stopped == False & (self.velocity_x >= 0.1) & (self.max_velocity_x >= 0.1)):
            if(self.no_gas != True):
                #Execute car model drive for checking left energy (-> check if left energy is enough to move the car onece more)
                _, _, _, _, _, _, _, self.consumed_energy_cycle = self.dbm.dynamic_model(self.x, self.y, self.yaw, self.velocity_x, self.velocity_y, self.max_velocity_x, self.delta, self.wished_delta, self.omega)
                #Add up consumed energy
                self.consumed_energy_since_start = self.consumed_energy_since_start + self.consumed_energy_cycle/1000                       #/1000 to get kJ
                consumption_liter = self.consumed_energy_since_start/(36*10^6)                                                              #one liter of gasolin has about 36 MJ energy
                #check how much energy is left
                self.energy_left = self.energy_contingent-self.consumed_energy_since_start
            if(self.energy_left > 0):#execute car model if there is enough energy to udate velocity and position
                self.x, self.y, self.yaw, self.velocity_x, self.velocity_y, self.last_delta, self.omega, _ = self.dbm.dynamic_model(self.x, self.y, self.yaw, self.velocity_x, self.velocity_y, self.max_velocity_x, self.delta, self.wished_delta, self.omega)
            else:#car ran out of gas, stop the car
                if(braking == False & self.car_stopped == False):
                    #If not braking, let car roll out if out of gas
                    friction = self.velocity_x * (self.c_r + self.c_a * self.velocity_x)
                    self.max_velocity_x = self.velocity_x - self.dt * friction
                    self.x, self.y, self.yaw, self.velocity_x, self.velocity_y, self.last_delta, self.omega, _ = self.dbm.dynamic_model(self.x, self.y, self.yaw, self.velocity_x, self.velocity_y, self.max_velocity_x, self.delta, self.wished_delta, self.omega)
                self.energy_left = 0.0
                self.consumed_energy_since_start = self.energy_contingent                   #overwrite these values when car is out of energy and can not perform any more move
                print("Car ran out of gas")
                self.no_gas = True
                if(self.velocity_x < 1.4):
                    self.velocity_x = 0.0
                    self.car_stopped = True


            print("energy consumed since start:",self.consumed_energy_since_start)
            #os.system('cls' if os.name=='nt' else 'clear')
            print(f"Cross-track term: {self.crosstrack_error}")
            
            



class DynamicBicycleModel():

    def __init__(self, Lr = 0.566, Lf =  0.566, max_steer=0.7, dt=0.05, c_r=0.0, c_a=0.0, c_alphafront= 90000.0, c_alpharear = 80000.0, Iz=1.0, mass = 810, car_FrontExtension = 1.55*1.5, efficiency = 35, max_acc = 0.3 * 9.81, max_dec=0.47 * 9.81, max_steering_velocity= 13):
        """
        2D Dynamic Bicycle Model

        This car model is based based on the dynamic bicycle model described e.g. in: https://www.cs.cmu.edu/~motionplanning/reading/PlanningforDynamicVeh-1.pdf and in https://www.coursera.org/lecture/intro-self-driving-cars/lesson-5-lateral-dynamics-of-bicycle-model-1Opvo
        The code is based on the implementation of the kinematic model: https://github.com/winstxnhdw/KinematicBicycleModel and was modified for the dynamic model

        At initialisation
        :param Lf:                  (float) distance front tire center to center of gravity [m]
        :param Lr:                  (float) distance rear tire center to center of gravity [m]
        :param max_steer:           (float) vehicle's steering limits [rad]
        :param dt:                  (float) discrete time period [s]
        :param c_r:                 (float) vehicle's coefficient of resistance 
        :param c_a:                 (float) vehicle's aerodynamic coefficient
        :param c_alphafront:        (float) corner stiffness coefficient front tire        //these values should be around 80000 N/rad according to the literature. In the code it only works with 80.
        :param c_alpharear:         (float) corner stiffness coefficient rear tire
        :param Iz:                  (float) yaw inertia (Trägheit)
        :param m:                   (float) mass car [kg]
        :param frontExtension:      (float) area of the car front (used to compute the aerodynamical resistance of the car) [m^2]
        :param efficiency:          (float) efficiency of the car engine [%]
        :param max_acc:             (float) maximum acceleration [m/s^2]
        :param max_dec:             (float) maximum braking deceleration [m/s^2]
        :param max_omega:           (float) maximum steering velocity [rad/s]
    
        At every time step  
        :param x:                   (float) vehicle's x-coordinate [m]
        :param y:                   (float) vehicle's y-coordinate [m]
        :param yaw:                 (float) vehicle's heading [rad]
        :param velocity_x:          (float) vehicle's velocity in the x-axis [m/s] (lonitudinal speed, relative to car model, if x is the moving direction of the car)
        :param velocity_y:          (float) vehicle's velocity in the x-axis [m/s] (lateral speed)
        :param max_velocity_X:      (float) vehicle's max velocity [m/s] (wished for by the controller/user, internally checked if possible with given acc/dec)
        :param delta:               (float) vehicle's steering angle [rad]
        :param omega:               (float) vehicle's yaw rate [rad/s]
    
        :return new_x:              (float) vehicle's x-coordinate [m]
        :return new_y:              (float) vehicle's y-coordinate [m]
        :return new_yaw:            (float) vehicle's heading [rad]
        :return last_velocity_x:    (float) vehicle's velocity in the x-axis [m/s]
        :return new_velocity_y:     (float) vehicle's velocity in the y-axis [m/s]
        :return steering_angle:     (float) vehicle's steering angle [rad]
        :return new_omega:          (float) vehicle's angular velocity [rad/s]
        :return consumed_energy     (float) consumed energy [J]
        """

        self.dt = dt
        self.Lf = Lf
        self.Lr = Lr
        self.max_steer = max_steer
        self.c_r = c_r
        self.c_a = c_a
        self.c_alphafront = c_alphafront
        self.c_alpharear = c_alpharear
        self.mass = mass
        self.Iz = Iz
        self.car_frontExtension = car_FrontExtension
        self.efficiency = efficiency
        self.max_acc = max_acc
        self.max_dec = max_dec
        self.max_omega = max_steering_velocity


    def dynamic_model(self, x, y, yaw, velocity_x, velocity_y, max_velocity_x, last_steering_angle, steering_angle, omega):

        # Limit the acceleration value to max_acc of the car
        acceleration = (max_velocity_x-velocity_x)/self.dt
        if(acceleration > self.max_acc):
            acceleration = self.max_acc
    
        #Limit deceleration
        if(acceleration < self.max_dec *(-1)):
            acceleration = self.max_dec *(-1)

        # Limit steering angle to physical vehicle limits
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)

        #Limit steering velocity to realisitc values
        steering_rate = (steering_angle-last_steering_angle)/self.dt
        if(steering_rate > self.max_omega):
            steering_angle = self.max_omega * self.dt 
        elif(steering_rate < self.max_omega *(-1)):
            steering_angle = self.max_omega * self.dt *(-1)

        #Update statespace model
        new_x = x + velocity_x * math.cos(yaw) * self.dt - velocity_y * math.sin(yaw) * self.dt
        new_y = y + velocity_x* math.sin(yaw) * self.dt + velocity_y * math.cos(yaw) * self.dt
        yaw = yaw + omega * self.dt
        new_yaw = normalize_angle(yaw)
        Ffy = -self.c_alphafront * math.atan2(((velocity_y + self.Lf * omega) / velocity_x - steering_angle), 1.0)
        Fry = -self.c_alpharear * math.atan2((velocity_y - self.Lr * omega) / velocity_x, 1.0)
        new_velocity_x = velocity_x + (acceleration - Ffy * math.sin(steering_angle)/self.mass + velocity_y * omega) * self.dt
        new_velocity_y = velocity_y + (Fry / self.mass + Ffy * math.cos(steering_angle) / self.mass - velocity_x * omega) * self.dt
        new_omega = omega + (Ffy * self.Lf * math.cos(steering_angle) - Fry * self.Lr) / self.Iz * self.dt

        #Compute distance travelled 
        path_travelled = np.sqrt(np.square(new_x-x)+np.square(new_y-y))                          # estimate path travelled in m
        
        #Compute friction losses
        friction_loss = self.c_r * self.mass * 9.81 * path_travelled                    # lost energy due to friction
        aero_loss = np.square(velocity_x)*self.c_a*1.255*self.car_frontExtension           # according to https://www.energie-lexikon.info/luftwiderstand.html, 1,255kg/m^3 is the density of air
        acc_energy = self.mass * acceleration *path_travelled                           # energy needed for the acceleration (only appliead if acceleration is positiv)
        idle_energy = 0.8 * 36 *10**6 * self.dt/3600                                    # according to https://www.sueddeutsche.de/wirtschaft/auto-wie-viel-sprit-verbraucht-ein-motor-im-leerlauf-dpa.urn-newsml-dpa-com-20090101-150217-99-02124 citing a the ADAC study: a combustion engine needs 0.8-1.5 l/h idle 

        #Compute used energy with resepct ot the effeciency of a car engine. Test showed that the car needed approx. 6Liter/100km by driving 30 km/h which was considered realistic given the efficiency of the car and the mass
        if(acc_energy > 0):
            consumed_energy = (aero_loss + friction_loss + acc_energy) * 100/self.efficiency + idle_energy          # 35% is assumed to be the efficiency of a combustion engine
        else: 
            consumed_energy = (aero_loss + friction_loss) * 100/self.efficiency + idle_energy
        

        #Update values
        return new_x, new_y, new_yaw, new_velocity_x, new_velocity_y, steering_angle, new_omega, consumed_energy

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

def main():

    print("This script is not meant to be executable, and should be used as a library.")

if __name__ == "__main__":
    main()
