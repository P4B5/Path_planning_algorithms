#In this file the sensors for the steering angle encoder, the wheel velocity sensor and the GNSS-receiver are modelled  

import numpy

class sensor_models():
    def model_GPS(x, y, velocity, heading):

        #GPS noise setup
        mean=0
        std_pos = 1         #GNSS receiver with DGPS like SBAS assumed
        std_velo = 0.2

        #apply noise on position signal
        x_GPS = x + numpy.random.normal(mean, std_pos, size=1)
        y_GPS = y + numpy.random.normal(mean, std_pos, size=1)
        velocity_GPS = velocity + numpy.random.normal(mean, std_velo, size=1)

        heading_GPS = heading

        return x_GPS, y_GPS, velocity_GPS, heading_GPS

    def model_steering_encoder(delta):

        #steering noise setup
        mean=0
        std = 5.23*10**(-3)         #assumed a encoder accuracy of 0.3 rad according to https://www.dynapar.com/knowledge/encoder_resolution_encoder_accuracy_repeatability/ high accuracy

        #apply noise on steering signal
        delta_encoder = delta + numpy.random.normal(mean, std, size=1)

        return delta_encoder
    
    def model_velocity_encoder(velocity):
        
        #velocity encoder noise setup
        mean=0
        std = 0.01 

        #apply noise on velocity encoder signal
        velocity_encoder = velocity + numpy.random.normal(mean, std, size=1)*velocity         #assumed, there was no reliable information found on accuracy of self-driving cars wheel encoders

        return velocity_encoder
    

