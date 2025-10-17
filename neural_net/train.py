#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""


import sys
from drive_train import DriveTrain
from config import Config

###############################################################################
#
def train(data_folder_name, load_model_name=None, base_model_path=None):
        drive_train = DriveTrain(data_folder_name, base_model_path=base_model_path)
        drive_train.train(show_summary=False, load_model_name=load_model_name)
        


###############################################################################
#
if __name__ == '__main__':
    try:
        if Config.neural_net['weight_load'] is True:
            if Config.neural_net['style_train'] is True:
                if (len(sys.argv) != 4):
                    exit('Usage:\n$ python {} data_path load_model_name base_model_name'.format(sys.argv[0]))
                train(data_folder_name=sys.argv[1], load_model_name=sys.argv[2], base_model_path=sys.argv[3])
            else:
                if (len(sys.argv) != 3):
                    exit('Usage:\n$ python {} data_path load_model_name'.format(sys.argv[0]))
                train(data_folder_name=sys.argv[1], load_model_name=sys.argv[2])
        
        else:
            if Config.neural_net['style_train'] is True:
                if (len(sys.argv) != 3):
                    exit('Usage:\n$ python {} data_path base_model_name'.format(sys.argv[0]))
                train(data_folder_name=sys.argv[1], base_model_path=sys.argv[2])
            else:
                if (len(sys.argv) != 2):
                    exit('Usage:\n$ python {} data_path'.format(sys.argv[0]))
                train(data_folder_name=sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
22222   