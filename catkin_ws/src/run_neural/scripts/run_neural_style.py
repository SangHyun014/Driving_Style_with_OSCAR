#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import threading 
import cv2
import time
import rospy
import numpy as np
from std_msgs.msg import Int32, Float64
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import math

import sys
import os

import const
from image_converter import ImageConverter
from drive_run import DriveRun
from config import Config
from image_process import ImageProcess
from tf.transformations import euler_from_quaternion, quaternion_from_euler

if Config.data_collection['vehicle_name'] == 'fusion':
    from fusion.msg import Control
elif Config.data_collection['vehicle_name'] == 'rover':
    from geometry_msgs.msg import Twist
    from rover.msg import Control
else:
    exit(Config.data_collection['vehicle_name'] + 'not supported vehicle.')


config = Config.neural_net
velocity = 0.0
g_steer = 0.0
goal_velocity = 0.0
style = 0

class NeuralControl:
    def __init__(self, weight_file_name, base_weight_name=None):
        rospy.init_node('run_neural')
        self.ic = ImageConverter()
        self.image_process = ImageProcess()
        self.rate = rospy.Rate(30)
        self.drive= DriveRun(weight_file_name, base_weight_name)
        rospy.Subscriber(Config.data_collection['camera_image_topic'], Image, self._controller_cb)
        self.image = None
        self.image_processed = False
        #self.config = Config()
        self.braking = False
        self.lstm_image = []
        self.lstm_vel = []
        self.term_count = 0

    def _controller_cb(self, image): 
        img = self.ic.imgmsg_to_opencv(image)
        cropped = img[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                      Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
        
        # base_model_cropped = img[Config.data_collection['base_image_crop_y1']:Config.data_collection['base_image_crop_y2'],
        #               Config.data_collection['base_image_crop_x1']:Config.data_collection['base_image_crop_x2']]
                      
        img = cv2.resize(cropped, (config['input_image_width'],
                                   config['input_image_height']))
        # base_model_img = cv2.resize(base_model_cropped, (config['input_image_width'],
        #                            config['input_image_height']))
                                  
        self.image = self.image_process.process(img)
        # self.base_model_image = self.image_process.process(base_model_img)
        
        # image NaN 디버깅
        """ 
        try:
            arr = self.image
            finite = np.isfinite(arr).all()
            # 모든 값이 NaN이면 nanmin/max에서 경고가 나므로 보호
            if np.isnan(arr).all():
                mn, mx = float('nan'), float('nan')
            else:
                mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
            # 0.5초에 한 번만 로그
            now = time.time()
            if now - self._dbg_last_log > 0.5:
                rospy.loginfo("image_proc: finite=%s min=%.3f max=%.3f shape=%s dtype=%s",
                              str(finite), mn, mx, str(getattr(arr, 'shape', None)), str(arr.dtype))
                self._dbg_last_log = now
        except Exception as e:
            rospy.logwarn("image_proc: exception in NaN check: %s", str(e))
        """

        ## this is for CNN-LSTM net models
        if config['lstm'] is True:
            if self.term_count % Config.run_neural['lstm_dataterm'] == 0:
                self.lstm_image.append(self.image)
                if len(self.lstm_image) > config['lstm_timestep'] :
                    del self.lstm_image[0]
                if config['num_inputs'] == 2:
                    self.lstm_vel.append(velocity)
                    if len(self.lstm_vel) > config['lstm_timestep']:
                        del self.lstm_vel[0]
            self.term_count += 1
                    
        self.image_processed = True
        
    def _timer_cb(self):
        self.braking = False

    def apply_brake(self):
        self.braking = True
        timer = threading.Timer(Config.run_neural['brake_apply_sec'], self._timer_cb) 
        timer.start()

      
def pos_vel_cb(value):
    global velocity

    vel_x = value.twist.twist.linear.x 
    vel_y = value.twist.twist.linear.y
    vel_z = value.twist.twist.linear.z
    
    velocity = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    
def goal_vel_cb(value):
    global goal_velocity
    goal_velocity = value.data

def steer_cb(value):
    global g_steer
    g_steer = value.data

def style_cb(value):
    global style
    style = value.data

def main(weight_file_name, base_weight_name=None):

    # ready for neural network
    neural_control = NeuralControl(weight_file_name, base_weight_name)
    
    rospy.Subscriber(Config.data_collection['base_pose_topic'], Odometry, pos_vel_cb)
    rospy.Subscriber(Config.data_collection['vehicle_steer_topic'], Float64, steer_cb)
    rospy.Subscriber(Config.data_collection['goal_velocity'], Float64, goal_vel_cb)
    rospy.Subscriber(Config.data_collection['style'], Int32, style_cb)

    # ready for /bolt topic publisher
    joy_pub = rospy.Publisher(Config.data_collection['vehicle_control_topic'], Control, queue_size = 10)
    joy_data = Control()

    if Config.data_collection['vehicle_name'] == 'rover':
        joy_pub4mavros = rospy.Publisher(Config.config['mavros_cmd_vel_topic'], Twist, queue_size=20)

    print('\nStart running. Vroom. Vroom. Vroooooom......')
    print('steer \tthrt: \tbrake \tvel \tgoal_v \tHz')

    use_predicted_throttle = True if config['num_outputs'] == 2 else False
    
    while not rospy.is_shutdown():

        if neural_control.image_processed is False:
            neural_control.rate.sleep()
            continue
        
        start = time.time()

        # predicted steering angle from an input image
        if config['num_inputs'] == 3:
            steer, throttle, brake = neural_control.drive.run((neural_control.image, velocity, float(goal_velocity-velocity)))
            # steer, throttle, brake = neural_control.drive.run((neural_control.image, neural_control.base_model_image, velocity, goal_velocity))
                      
            joy_data.steer = steer
            joy_data.throttle = throttle
            joy_data.brake = brake
        
        if config['num_inputs'] == 4:

            # NaN 디버깅용
            # feats, y = neural_control.drive.debug_cvae_outputs(neural_control.image, velocity, float(goal_velocity-velocity), style)            # 예: cond 최종만 확인
            # cond = feats.get('cvae_cond_fc')  # shape: (1, 256)
            # rospy.loginfo("cond finite=%s min=%.3f max=%.3f y_mean=%s",
            #            np.isfinite(cond).all(), np.nanmin(cond), np.nanmax(cond),
            #            np.array2string(y[0], precision=3))
            
            steer, throttle, brake = neural_control.drive.run((neural_control.image, velocity, float(goal_velocity-velocity), style))
            joy_data.steer = steer
            joy_data.throttle = throttle
            joy_data.brake = brake

        end = time.time()

        #############################
        ## very very simple controller
        ## 
        # is_sharp_turn = False
        # # if brake is not already applied and sharp turn
        # if neural_control.braking is False: 
        #     if velocity < Config.run_neural['velocity_0']: # too slow then no braking
        #         joy_data.throttle = Config.run_neural['throttle_default'] # apply default throttle
        #         joy_data.brake = 0
        #     elif abs(joy_data.steer) > Config.run_neural['sharp_turn_min']:
        #         is_sharp_turn = True
            
        #     if is_sharp_turn or velocity > Config.run_neural['max_vel']: 
        #         joy_data.throttle = Config.run_neural['throttle_sharp_turn']
        #         joy_data.brake = Config.run_neural['brake_val']
        #         neural_control.apply_brake()
        #     else:
        #         if use_predicted_throttle is False:
        #             joy_data.throttle = Config.run_neural['throttle_default']
        #         joy_data.brake = 0
                

        
        ##############################    
        ## publish mavros control topic
        joy_pub.publish(joy_data)
        # if Config.data_collection['vehicle_name'] == 'rover':
        #     joy_data4mavros = Twist()
        #     if neural_control.braking is True:
        #         joy_data4mavros.linear.x = 0
        #         joy_data4mavros.linear.y = 0
        #     else: 
        #         joy_data4mavros.linear.x = joy_data.throttle*Config.run_neural['scale_factor_throttle']
        #         joy_data4mavros.linear.y = joy_data.steer*Config.run_neural['scale_factor_steering']

        #     joy_pub4mavros.publish(joy_data4mavros)


        ## print out
        # print(joy_data.steer, joy_data.throttle, joy_data.brake, velocity)
        hz = (1.0/(end-start)) if end>start else 0.0
        cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f} \t{4:.3f} \t{5:.1f}\r'.format(joy_data.steer, joy_data.throttle, joy_data.brake, velocity, goal_velocity, hz)

        sys.stdout.write(cur_output)
        sys.stdout.flush()
            
        
        ## ready for processing a new input image
        neural_control.image_processed = False
        neural_control.rate.sleep()



if __name__ == "__main__":
    try:
        if config['style_run'] is True:
            if len(sys.argv) != 3:
                exit('Usage:\n$ rosrun run_neural run_neural.py style_weight_name base_weight_name')
            main(sys.argv[1], sys.argv[2])
                
        else:
            if len(sys.argv) != 2:
                exit('Usage:\n$ rosrun run_neural run_neural.py weight_file_name')
            main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
        
