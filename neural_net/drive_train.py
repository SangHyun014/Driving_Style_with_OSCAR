#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np
#import keras
import sklearn
from sklearn.model_selection import train_test_split

import const
from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess
from data_augmentation import DataAugmentation
from progressbar import ProgressBar
from keras import backend as K
import tensorflow as tf
from keras.callbacks import Callback

config = Config.neural_net

class KLAnneal(Callback):  # ✅ keras.callbacks.Callback 상속
    def __init__(self, kl_var, warmup_epochs=10, start=0.0, end=1.0):
        super(KLAnneal, self).__init__()  # ✅ Py2 스타일
        self.kl_var = kl_var
        self.warmup_epochs = warmup_epochs
        self.start = float(start)
        self.end = float(end)

    def on_epoch_begin(self, epoch, logs=None):
        ratio = min(1.0, epoch / float(self.warmup_epochs))
        val = self.start + (self.end - self.start) * ratio
        K.set_value(self.kl_var, val)
        print("\n[KLAnneal] epoch {}: kl_w = {:.3f}".format(epoch, val))

    # (선택) keras는 기본적으로 on_batch_begin이 있지만, 안전하게 더미 메서드 추가해도 됨
    def on_batch_begin(self, batch, logs=None):
        pass

###############################################################################
#
class DriveTrain:
    
    ###########################################################################
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56/'
    def __init__(self, data_path, base_model_path=None):
        
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            model_name = data_path[loc_slash + 1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            model_name = data_path
        csv_path = data_path + '/' + model_name + const.DATA_EXT  # use it for csv file name 
        
        self.csv_path = csv_path
        self.train_generator = None
        self.valid_generator = None
        self.train_hist = None
        self.data = None
        
        #self.config = Config() #model_name)
        
        #self.model_name = model_name
        
        self.model_name = data_path + '_' + Config.neural_net_yaml_name \
            + '_N' + str(config['network_type'])
        self.model_ckpt_name = self.model_name + '_ckpt'

        if config['data_split'] is True:
            self.data = DriveData(self.csv_path)
            self.data_path = data_path
        else:
            self.t_data = DriveData(data_path+'/train/'+ model_name+'/'+ model_name + const.DATA_EXT)
            self.v_data = DriveData(data_path+'/valid/'+ model_name+'/'+ model_name + const.DATA_EXT)
            self.t_data_path = data_path+'/train/'+ model_name
            self.v_data_path = data_path+'/valid/'+ model_name
        self.net_model = NetModel(data_path, base_model_path=base_model_path)
        self.image_process = ImageProcess()
        self.data_aug = DataAugmentation()
        
        
    ###########################################################################
    #
    def _prepare_data(self):
        
        if config['data_split'] is True:
            self.data.read()
            # put velocities regardless we use them or not for simplicity.
            samples = list(zip(self.data.image_names, self.data.velocities, self.data.measurements))
            if config['lstm'] is True:
                self.train_data, self.valid_data = self._prepare_lstm_data(samples)
            else:    
                self.train_data, self.valid_data = train_test_split(samples, 
                                        test_size=config['validation_rate'])
        else:
            self.t_data.read()
            self.v_data.read()
            # put velocities regardless we use them or not for simplicity.
            if config['num_inputs'] == 3 and config['only_thr_brk'] is False:
                train_samples = list(zip(self.t_data.image_names, self.t_data.velocities, self.t_data.measurements, self.t_data.goal_velocities))
                valid_samples = list(zip(self.v_data.image_names, self.v_data.velocities, self.v_data.measurements, self.v_data.goal_velocities))
            elif config['num_inputs'] == 4:
                train_samples = list(zip(self.t_data.image_names, self.t_data.velocities, self.t_data.measurements, self.t_data.goal_velocities, self.t_data.styles))
                valid_samples = list(zip(self.v_data.image_names, self.v_data.velocities, self.v_data.measurements, self.v_data.goal_velocities, self.v_data.styles))
            elif config['only_thr_brk'] is True:
                train_samples = list(zip(self.t_data.image_names, self.t_data.velocities, self.t_data.measurements, self.t_data.goal_velocities))
                valid_samples = list(zip(self.v_data.image_names, self.v_data.velocities, self.v_data.measurements, self.v_data.goal_velocities))
            else:
                train_samples = list(zip(self.t_data.image_names, self.t_data.velocities, self.t_data.measurements))
                valid_samples = list(zip(self.v_data.image_names, self.v_data.velocities, self.v_data.measurements))
                
            if config['lstm'] is True:
                self.train_data,_ = self._prepare_lstm_data(train_samples)
                self.valid_data,_ = self._prepare_lstm_data(valid_samples)
            else:
                self.train_data = train_samples
                self.valid_data = valid_samples
            
        self.num_train_samples = len(self.train_data)
        self.num_valid_samples = len(self.valid_data)
        
        print('Train samples: ', self.num_train_samples)
        print('Valid samples: ', self.num_valid_samples)
    
                                          
    ###########################################################################
    # group the samples by the number of timesteps
    def _prepare_lstm_data(self, samples):
        num_samples = len(samples)

        # get the last index number
        last_index = (num_samples - config['lstm_timestep']*config['lstm_dataterm'])
        
        image_names = []
        measurements = []
        velocities = []
        for i in range(0, last_index):
            timestep_samples = samples[ i : i+config['lstm_timestep']*config['lstm_dataterm'] :config['lstm_dataterm']]
            is_same_dataset = 0
            timestep_image_names = []
            timestep_measurements = []
            timestep_velocities = []
            timestep_image_times = []
            for image_name, velocity, measurment in timestep_samples:
                timestep_image_names.append(image_name)
                timestep_measurements.append(measurment)
                timestep_velocities.append(velocity)
                hour = int(str(image_name.split('.')[0].split('-')[-4:-3][0]))
                miniute = int(str(image_name.split('.')[0].split('-')[-3:-2][0]))
                second = int(str(image_name.split('.')[0].split('-')[-2:-1][0]))
                timestep_image_times.append(hour*3600 + miniute*60 + second)
            prev_time = timestep_image_times[0]
            for i in range(1, len(timestep_image_times)):
                if abs(timestep_image_times[i] - prev_time) >= config['data_timegap']:
                    is_same_dataset += 1
            if is_same_dataset is 0:
                image_names.append(timestep_image_names)
                measurements.append(timestep_measurements)
                velocities.append(timestep_velocities)
                
            image_names.append(timestep_image_names)
            measurements.append(timestep_measurements)
            velocities.append(timestep_velocities)
            
        if config['data_split'] is True:
            samples = list(zip(image_names, velocities, measurements))
            train_data, valid_data = train_test_split(samples, 
                                        test_size=config['validation_rate'])
        else:
            # put velocities regardless we use them or not for simplicity.
            train_data = list(zip(image_names, velocities, measurements))
            train_data = sklearn.utils.shuffle(train_data)
            # print(train_data)
            valid_data = None
            
        return train_data, valid_data

    ###########################################################################
    #
    def _build_model(self, show_summary=True):

        def _data_augmentation(image, steering_angle):
            if config['data_aug_flip'] is True:    
                # Flipping the image
                return True, self.data_aug.flipping(image, steering_angle)

            if config['data_aug_bright'] is True:    
                # Changing the brightness of image
                if steering_angle > config['steering_angle_jitter_tolerance'] or \
                    steering_angle < -config['steering_angle_jitter_tolerance']:
                    image = self.data_aug.brightness(image)
                return True, image, steering_angle

            if config['data_aug_shift'] is True:    
                # Shifting the image
                return True, self.data_aug.shift(image, steering_angle)

            return False, image, steering_angle

        def _prepare_batch_samples(batch_samples, data=None):
            images = []
            velocities = []
            measurements = []
            goal_velocities = []
            # steers = []
            # throttles = []
            # brakes = []
            # deltas = []
            styles = []

            if data is None:
                data_path = self.data_path
            elif data == 'train':
                data_path = self.t_data_path
            elif data == 'valid':
                data_path = self.v_data_path
                
            if config['num_inputs'] == 2 or config['only_thr_brk'] is True:
                for image_name, velocity, measurement, goal_velocity in batch_samples:
                    # for image_name, velocity, measurement, delta in batch_samples:
                    image_path = data_path + '/' + image_name
                    # print(image_path)
                    image = cv2.imread(image_path)
                    # if collected data is not cropped then crop here
                    # otherwise do not crop.
                    if Config.data_collection['crop'] is not True:
                        image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                    Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
                    image = cv2.resize(image, 
                                        (config['input_image_width'],
                                        config['input_image_height']))
                    image = self.image_process.process(image)
                    # cv2.imwrite('/home/kdh/oscar/oscar/e2e_fusion_data/test/aug/'+image_name, image)
                    # if data == 'train':
                    #     cv2.imwrite('/mnt/Data/oscar/train_data/'+image_name, image)
                    # print(image.shape)
                    images.append(image)
                    
                    velocities.append(velocity)
                    goal_velocities.append(goal_velocity-velocity)
                    # if no brake data in collected data, brake values are dummy
                    steering_angle, throttle, brake = measurement
                    
                    if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                        steering_angle = 0

                    if config['num_outputs'] == 2:                
                        measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale']))
                    elif config['num_outputs'] == 3 and config['only_thr_brk'] is False:                
                        measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale'], brake*config['brake_scale']))
                        # steers.append(steering_angle*config['steering_angle_scale'])
                        # throttles.append(throttle*config['throttle_scale'])
                        # brakes.append(brake*config['brake_scale'])
                    elif config['num_outputs'] == 3 and config['only_thr_brk'] is True:      
                        measurements.append((throttle*config['throttle_scale'], brake*config['brake_scale']))
                    else:
                        measurements.append(steering_angle*config['steering_angle_scale'])
                        # print("1 : ", steering_angle)
                    
                    # cv2.imwrite('/home/kdh/oscar/oscar/e2e_fusion_data/test/aug/'+image_name, image)
                    # data augmentation
                    append, image, steering_angle = _data_augmentation(image, steering_angle)
                    if append is True:
                        # cv2.imwrite('/home/kdh/oscar/oscar/e2e_fusion_data/test/aug/'+image_name, image)
                        images.append(image)
                        velocities.append(velocity)
                        goal_velocities.append(goal_velocity-velocity)
                        if config['num_outputs'] == 2:                
                            measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale']))
                        elif config['num_outputs'] == 3 and config['only_thr_brk'] is False:                  
                            measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale'], brake*config['brake_scale']))
                            # steers.append(steering_angle*config['steering_angle_scale'])
                            # throttles.append(throttle*config['throttle_scale'])
                            # brakes.append(brake*config['brake_scale'])
                        elif config['num_outputs'] == 3 and config['only_thr_brk'] is True:      
                            measurements.append((throttle*config['throttle_scale'], brake*config['brake_scale']))
                        else:
                            measurements.append(steering_angle*config['steering_angle_scale'])
                return images, velocities, measurements, goal_velocities #, steers, throttles, brakes
            
            elif config['num_inputs'] == 3:
                for image_name, velocity, measurement, goal_velocity in batch_samples:
                    # for image_name, velocity, measurement, delta in batch_samples:
                    image_path = data_path + '/' + image_name
                    # print(image_path)
                    image = cv2.imread(image_path)
                    # if collected data is not cropped then crop here
                    # otherwise do not crop.
                    if Config.data_collection['crop'] is not True:
                        image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                    Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
                    image = cv2.resize(image, 
                                        (config['input_image_width'],
                                        config['input_image_height']))
                    image = self.image_process.process(image)
                    # cv2.imwrite('/home/kdh/oscar/oscar/e2e_fusion_data/test/aug/'+image_name, image)
                    # if data == 'train':
                    #     cv2.imwrite('/mnt/Data/oscar/train_data/'+image_name, image)
                    # print(image.shape)
                    images.append(image)
                    
                    velocities.append(velocity)
                    goal_velocities.append(goal_velocity-velocity)
                    # if no brake data in collected data, brake values are dummy
                    steering_angle, throttle, brake = measurement
                    
                    if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                        steering_angle = 0

                    if config['num_outputs'] == 2:                
                        measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale']))
                    elif config['num_outputs'] == 3:                
                        measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale'], brake*config['brake_scale']))
                    else:
                        measurements.append(steering_angle*config['steering_angle_scale'])
                        # print("1 : ", steering_angle)
                    
                    # cv2.imwrite('/home/kdh/oscar/oscar/e2e_fusion_data/test/aug/'+image_name, image)
                    # data augmentation
                    append, image, steering_angle = _data_augmentation(image, steering_angle)
                    if append is True:
                        # cv2.imwrite('/home/kdh/oscar/oscar/e2e_fusion_data/test/aug/'+image_name, image)
                        images.append(image)
                        velocities.append(velocity)
                        goal_velocities.append(goal_velocity-velocity)
                        if config['num_outputs'] == 2:                
                            measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale']))
                        elif config['num_outputs'] == 3:                
                            measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale'], brake*config['brake_scale']))
                        else:
                            measurements.append(steering_angle*config['steering_angle_scale'])
                return images, velocities, measurements, goal_velocities

            elif config['num_inputs'] == 4:
                for image_name, velocity, measurement, goal_velocity, style in batch_samples:
                    # for image_name, velocity, measurement, delta in batch_samples:
                    image_path = data_path + '/' + image_name
                    # print(image_path)
                    image = cv2.imread(image_path)
                    # if collected data is not cropped then crop here
                    # otherwise do not crop.
                    if Config.data_collection['crop'] is not True:
                        image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                    Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
                    image = cv2.resize(image, 
                                        (config['input_image_width'],
                                        config['input_image_height']))
                    image = self.image_process.process(image)
                    # cv2.imwrite('/home/kdh/oscar/oscar/e2e_fusion_data/test/aug/'+image_name, image)
                    # if data == 'train':
                    #     cv2.imwrite('/mnt/Data/oscar/train_data/'+image_name, image)
                    # print(image.shape)
                    images.append(image)
                    
                    velocities.append(velocity)
                    goal_velocities.append(goal_velocity-velocity)
                    styles.append(style)
                    # if no brake data in collected data, brake values are dummy
                    steering_angle, throttle, brake = measurement
                    
                    if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                        steering_angle = 0

                    if config['num_outputs'] == 2:                
                        measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale']))
                    elif config['num_outputs'] == 3:                
                        measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale'], brake*config['brake_scale']))
                    else:
                        measurements.append(steering_angle*config['steering_angle_scale'])
                        # print("1 : ", steering_angle)
                    
                    # cv2.imwrite('/home/kdh/oscar/oscar/e2e_fusion_data/test/aug/'+image_name, image)
                    # data augmentation
                    append, image, steering_angle = _data_augmentation(image, steering_angle)
                    if append is True:
                        # cv2.imwrite('/home/kdh/oscar/oscar/e2e_fusion_data/test/aug/'+image_name, image)
                        images.append(image)
                        velocities.append(velocity)
                        goal_velocities.append(goal_velocity-velocity)
                        styles.append(style)
                        if config['num_outputs'] == 2:                
                            measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale']))
                        elif config['num_outputs'] == 3:                
                            measurements.append((steering_angle*config['steering_angle_scale'], throttle*config['throttle_scale'], brake*config['brake_scale']))
                        else:
                            measurements.append(steering_angle*config['steering_angle_scale'])
                
                return images, velocities, measurements, goal_velocities, styles

            

        def _prepare_lstm_batch_samples(batch_samples, data=None):
            images = []
            velocities = []
            measurements = []
            steer = []
            thr = []
            if data is None:
                data_path = self.data_path
            elif data == 'train':
                data_path = self.t_data_path
            elif data == 'valid':
                data_path = self.v_data_path
            for i in range(0, config['batch_size']):
                images_timestep = []
                image_names_timestep = []
                velocities_timestep = []
                measurements_timestep = []
                images_aug_timestep = []
                velocities_aug_timestep = []
                measurements_aug_timestep = []
                steer_timestep = []
                thr_timestep = []
                for j in range(0, config['lstm_timestep']):
                    image_name = batch_samples[i][0][j]
                    image_path = data_path + '/' + image_name
                    image = cv2.imread(image_path)
                    # if collected data is not cropped then crop here
                    # otherwise do not crop.
                    if Config.data_collection['crop'] is not True:
                        image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                    Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
                    image = cv2.resize(image, 
                                    (config['input_image_width'],
                                    config['input_image_height']))
                    image = self.image_process.process(image)
                    images_timestep.append(image)
                    image_names_timestep.append(image_name)
                    velocity = batch_samples[i][1][j]
                    velocities_timestep.append(velocity)
                    
                    if j is config['lstm_timestep']-1:
                        measurement = batch_samples[i][2][j]
                        # if no brake data in collected data, brake values are dummy
                        steering_angle, throttle, brake = measurement
                        
                        if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                            steering_angle = 0
                            
                        if config['num_outputs'] == 2:                
                            measurements_timestep.append((steering_angle*config['steering_angle_scale'], throttle))
                            steer_timestep.append(steering_angle*config['steering_angle_scale'])
                            thr_timestep.append(throttle*config['throttle_scale'])
                        else:
                            measurements_timestep.append(steering_angle*config['steering_angle_scale'])
                            steer_timestep.append(steering_angle*config['steering_angle_scale'])
                

                images.append(images_timestep)
                velocities.append(velocities_timestep)
                measurements.append(measurements_timestep)
                steer.append(steer_timestep)
                thr.append(thr_timestep)
                
            return images, velocities, measurements, steer, thr
        
        def _generator(samples, batch_size=config['batch_size'], data=None):
            num_samples = len(samples)
            while True: # Loop forever so the generator never terminates
                if config['lstm'] is True:
                    for offset in range(0, (num_samples//batch_size)*batch_size, batch_size):
                        batch_samples = samples[offset:offset+batch_size]

                        images, velocities, measurements, steer, thr, brk = _prepare_lstm_batch_samples(batch_samples, data)
                        
                        if config['num_inputs'] == 1:
                            X_train = np.array(images)
                        elif config['num_inputs'] == 2:
                            X_train = np.array(images)
                            X_train_vel = np.array(velocities).reshape(-1,config['lstm_timestep'],1)
                            X_train = [X_train, X_train_vel]
                            
                        if config['num_outputs'] == 1:
                            y_train = np.array(measurements)
                        elif config['num_outputs'] == 2:
                            # print(y_train_str.shape)
                            if config['only_thr_brk'] is False:
                                y_train_str = np.array(steer).reshape(-1,1)
                                y_train_thr = np.array(thr).reshape(-1,1)
                                y_train = [y_train_str, y_train_thr]
                            elif config['only_thr_brk'] is True:
                                y_train_thr = np.array(thr).reshape(-1,1)
                                y_train_brk = np.array(brk).reshape(-1,1)
                                y_train = [y_train_thr, y_train_brk]
                        elif config['num_outputs'] == 3:
                            # print(y_train_str.shape)
                            y_train_str = np.array(steer).reshape(-1,1)
                            y_train_thr = np.array(thr).reshape(-1,1)
                            y_train_brk = np.array(brk).reshape(-1,1)
                            y_train = [y_train_str, y_train_thr, y_train_brk]
                        
                        # print(X_train_vel.shape)
                        yield X_train, y_train
                        
                else: 
                    samples = sklearn.utils.shuffle(samples)

                    for offset in range(0, num_samples, batch_size):
                        batch_samples = samples[offset:offset+batch_size]
                        # print(len(batch_samples))
                        if config['num_inputs'] == 2:
                            images, velocities, measurements, _= _prepare_batch_samples(batch_samples, data)
                            X_train_str = np.array(images)
                            X_train_vel = np.array(velocities).reshape(-1, 1)
                            X_train = [X_train_str, X_train_vel]
                        elif config['num_inputs'] == 3 and config['only_thr_brk'] is False:
                            images, velocities, measurements, goal_vel = _prepare_batch_samples(batch_samples, data)
                            X_train_str = np.array(images)
                            X_train_vel = np.array(velocities).reshape(-1, 1)
                            X_train_gvel = np.array(goal_vel).reshape(-1, 1)
                            X_train = [X_train_str, X_train_vel, X_train_gvel]
                        elif config['only_thr_brk'] is True:
                            images, velocities, measurements, goal_vel = _prepare_batch_samples(batch_samples, data)
                            X_train_str = np.array(images)
                            X_train_vel = np.array(velocities).reshape(-1, 1)
                            X_train_gvel = np.array(goal_vel).reshape(-1, 1)
                            X_train = [X_train_str, X_train_vel, X_train_gvel]
                        elif config['num_inputs'] == 4 and config['style_train'] is True:
                            #print(batch_samples)
                            images, velocities, measurements, goal_vel, styles = _prepare_batch_samples(batch_samples, data)
                            X_train_str = np.array(images)
                            X_train_vel = np.array(velocities).reshape(-1, 1)
                            X_train_gvel = np.array(goal_vel).reshape(-1, 1)
                            X_train_sty = np.array(styles).reshape(-1, 1)
                            X_train_mea = np.array(measurements)
                            X_train = [X_train_str, X_train_vel, X_train_gvel, X_train_sty, X_train_mea]
                        elif config['num_inputs'] == 4 and config['style_train'] is False:
                            images, velocities, measurements, goal_vel, styles = _prepare_batch_samples(batch_samples, data)
                            X_train_str = np.array(images)
                            X_train_vel = np.array(velocities).reshape(-1, 1)
                            X_train_gvel = np.array(goal_vel).reshape(-1, 1)
                            X_train_sty = np.array(styles).reshape(-1, 1)
                            X_train = [X_train_str, X_train_vel, X_train_gvel, X_train_sty]

                            
                        else:
                            images, _, measurements = _prepare_batch_samples(batch_samples, data)
                            X_train = np.array(images)
                            
                        
                        # if config['num_outputs'] == 2:
                        #     y_train = np.array(measurements)
                        # elif config['num_outputs'] == 3:
                        y_train = np.array(measurements)
                            
                        yield X_train, y_train
                        
        if config['data_split'] is True:
            self.train_generator = _generator(self.train_data)
            self.valid_generator = _generator(self.valid_data)       
        else:
            self.train_generator = _generator(self.train_data, data='train')
            self.valid_generator = _generator(self.valid_data, data='valid')
        
        if (show_summary):
            self.net_model.model.summary()
    
    ###########################################################################
    #
    def _start_training(self):
        
        if (self.train_generator == None):
            raise NameError('Generators are not ready.')
        
        ######################################################################
        # callbacks
        from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
        
        # checkpoint
        callbacks = []
        #weight_filename = self.data_path + '_' + Config.config_yaml_name \
        #    + '_N' + str(config['network_type']) + '_ckpt'
        checkpoint = ModelCheckpoint(self.model_ckpt_name +'.{epoch:02d}-{val_loss:.3f}.h5',
                                     monitor='val_loss', 
                                     verbose=1, save_best_only=True, mode='min')
        callbacks.append(checkpoint)
        
        # early stopping
        patience = config['early_stopping_patience']
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, 
                                  verbose=1, mode='min')
        callbacks.append(earlystop)

        # tensor board
        logdir = config['tensorboard_log_dir'] + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard)
        
        # KL warm-up
        if config['network_type'] == const.NET_TYPE_CVAE:
            KL_warmup = KLAnneal(kl_var=self.net_model.model.kl_w, warmup_epochs=15, start=0.0, end=1.0)
            callbacks.append(KL_warmup)

        self.train_hist = self.net_model.model.fit_generator(
                self.train_generator, 
                steps_per_epoch=self.num_train_samples//config['batch_size'], 
                epochs=config['num_epochs'], 
                validation_data=self.valid_generator,
                validation_steps=self.num_valid_samples//config['batch_size'],
                verbose=1, callbacks=callbacks, 
                use_multiprocessing=True,
                workers=24)
        
    ###########################################################################
    #
    def _plot_training_history(self):
    
        print(self.train_hist.history.keys())
        
        plt.figure() # new figure window
        ### plot the training and validation loss for each epoch
        plt.plot(self.train_hist.history['loss'][1:])
        plt.plot(self.train_hist.history['val_loss'][1:])
        #plt.title('Mean Squared Error Loss')
        plt.ylabel('mse loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validatation set'], loc='upper right')
        plt.tight_layout()
        #plt.show()
        plt.savefig(self.model_name + '_model.png', dpi=150)
        plt.savefig(self.model_name + '_model.pdf', dpi=150)
        
        new_txt = []
        bar = ProgressBar()
        for i in bar(range(len(self.train_hist.history['loss']))):
            new_txt.append(
                str(i)
                + ', '
                + str(self.train_hist.history['loss'][i])
                + ', '
                + str(self.train_hist.history['val_loss'][i])+ '\n')
            
        new_txt_fh = open(self.model_name + '_loss.csv', 'w')
        for i in range(len(new_txt)):
            new_txt_fh.write(new_txt[i])
        new_txt_fh.close()
        
    ###########################################################################
    #
    def train(self, show_summary=True, load_model_name=None):
        
        self._prepare_data()
        if config['weight_load'] is True:
            self.net_model.weight_load(load_model_name)
            # print("done")
        self._build_model(show_summary)
        self._start_training()
        self.net_model.save(self.model_name)
            
        self._plot_training_history()
        Config.summary()
