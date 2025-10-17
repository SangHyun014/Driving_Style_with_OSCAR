#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:07:31 2019
History:
11/28/2020: modified for OSCAR 

@author: Donghyun Kim
"""

import numpy as np
#import keras
#import sklearn
#import resnet
from progressbar import ProgressBar
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd

import const
from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess

###############################################################################
#
class DrivieStyleLog:
    csv_header = ['image_fname', 
                  'steering_angle', 'throttle', 'brake', 
                  'linux_time', 
                  'vel', 'vel_x', 'vel_y', 'vel_z',
                  'accel_x', 'accel_y', 
                  'pos_x', 'pos_y', 'pos_z',
                  'goal_vel', 'style']
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
       
    def __init__(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            model_name = data_path[loc_slash+1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            model_name = data_path

        self.csv_path = data_path + '/' + model_name + const.DATA_EXT   
        
        self.data_path = data_path
        # self.data = DriveData(self.csv_path)
        self.df = None
    
        self.pos_x = []
        self.pos_y = []
        self.accel = []
        self.min_acc = 0
        self.max_acc = 0

    ###########################################################################
    #
    # def _prepare_data(self):
        
    #     self.data.read(normalize = False)
    
        # self.test_data = list(zip(self.data.positions_xyz[0], self.data.positions_xyz[1], self.data.velocities_xyz[3]))
        # self.num_test_samples = len(self.test_data)
        
        # print('Test samples: {0}'.format(self.num_test_samples))

    
   ###########################################################################
    #
    def _savefigs(self, plt, filename):
        plt.savefig(filename + '.png', dpi=150)
        plt.savefig(filename + '.pdf', dpi=150)
        print('Saved ' + filename + '.png & .pdf.')


    ###########################################################################
    #
    def _plot_results(self):
        plt.figure()
        colormap = plt.cm.seismic
        # normalize = None
        normalize = colors.Normalize(vmin=self.min_acc, vmax=self.max_acc)
        # print(self.min_acc, self.max_acc)
        bar = ProgressBar()
        print('\nPlotting Driving Style:')
        # for i in bar(range(len(self.df))):
        #     plt.scatter(self.pos_x[i], self.pos_y[i], c=self.accel[i], s=4, cmap=colormap, norm=normalize)
        # plt.scatter(self.pos_x, self.pos_y, c=colormap(normalize(self.accel)))
        plt.plot(self.pos_x, self.pos_y, c=colormap(normalize(self.accel)))
        
        # plt.tight_layout()
        self._savefigs(plt, self.data_path + '_N' + str(Config.neural_net['network_type']) + '_style')

    def run(self):
        self.df = pd.read_csv(self.csv_path, names=self.csv_header, index_col=False)
        #self.fname = fname

        num_data = len(self.df)
        bar = ProgressBar()
        
        for i in bar(range(num_data)): # we don't have a title
            self.pos_x.append(float(self.df.loc[i]['pos_x']))
            self.pos_y.append(float(self.df.loc[i]['pos_y']))
            self.accel.append(float(self.df.loc[i]['accel_x']))
            if self.min_acc > float(self.df.loc[i]['accel_x']):
                self.min_acc = float(self.df.loc[i]['accel_x'])
            if self.max_acc < float(self.df.loc[i]['accel_x']):
                self.max_acc = float(self.df.loc[i]['accel_x'])
        
    
        ############################################ 
        # read out
        
        self._plot_results()



from drive_style_log import DrivieStyleLog


###############################################################################
#       
def main(data_folder_name):
    drive_style_log = DrivieStyleLog(data_folder_name) 
    drive_style_log.run() # data folder path to test
       

###############################################################################
#       
if __name__ == '__main__':
    import sys

    try:
        if (len(sys.argv) != 2):
            exit('Usage:\n$ python {} data_folder_name'.format(sys.argv[0]))
        
        main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
