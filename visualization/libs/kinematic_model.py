#!/usr/bin/env python

from numpy import cos, sin, tan, clip
from libs.normalise_angle import normalise_angle

class KinematicBicycleModel():

    def __init__(self, wheelbase=1.0, max_steer=0.7, dt=0.05, max_acc = 3):
        """
        2D Kinematic Bicycle Model

        The kinematic model is based on https://github.com/winstxnhdw/KinematicBicycleModel and was slightly modified 


        At initialisation
        :param wheelbase:           (float) vehicle's wheelbase [m]
        :param max_steer:           (float) vehicle's steering limits [rad]
        :param dt:                  (float) discrete time period [s]
        :max acc:                   (float) maximum car acceleration [m/s^2]
    
        At every time step  
        :param x:                   (float) vehicle's x-coordinate [m]
        :param y:                   (float) vehicle's y-coordinate [m]
        :param yaw:                 (float) vehicle's heading [rad]
        :param velocity:            (float) vehicle's velocity in the x-axis [m/s]
        :param throttle:            (float) vehicle's accleration [m/s^2]
        :param delta:               (float) vehicle's steering angle [rad]
    
        :return new_x:              (float) vehicle's x-coordinate [m]
        :return new_y:              (float) vehicle's y-coordinate [m]
        :return new_yaw:            (float) vehicle's heading [rad]
        :return new_velocity:       (float) vehicle's velocity in the x-axis [m/s]
        :return steering_angle:     (float) vehicle's steering angle [rad]
        :return angular_velocity:   (float) vehicle's angular velocity [rad/s]
        """

        self.dt = dt
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.max_acc = max_acc

    def kinematic_model(self, x, y, yaw, max_velocity_x, velocity_x, steering_angle):

        # Limit the acceleration value to max_acc of the car
        acceleration = (max_velocity_x-velocity_x)/self.dt
        if(acceleration > self.max_acc):
            new_velocity = velocity_x + self.max_acc * self.dt

        # Limit steering angle to physical vehicle limits
        steering_angle = clip(steering_angle, -self.max_steer, self.max_steer)

        # Compute the angular velocity
        angular_velocity = new_velocity * tan(steering_angle) / self.wheelbase

        # Compute the final state using the discrete time model
        new_x = x + new_velocity * cos(yaw) * self.dt
        new_y = y + new_velocity * sin(yaw) * self.dt
        new_yaw = normalise_angle(yaw + angular_velocity * self.dt)
        
        return new_x, new_y, new_yaw

def main():

    print("This script is not meant to be executable, and should be used as a library.")

if __name__ == "__main__":
    main()
