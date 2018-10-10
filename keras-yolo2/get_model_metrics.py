#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:34:11 2018

@author: vineeth
"""

### Find a saved model accuracy

## validation dataset usedduring training

import argparse
import os
from preprocessing import parse_annotation, BatchGenerator
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='get YOLO_v2 model metrics from a saved mdoel')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

#argparser.add_argument(
#    '-i',
#    '--input',
#    help='path to validation dataset')

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)
        
    # parse annotations of the validation set, if any, otherwise raise an error
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'], 
                                                    config['valid']['valid_image_folder'], 
                                                    config['model']['labels'])
    else:
         print('Folder ' + config['valid']['valid_annot_folder'] + 'does not exist')
    

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ############################################
    # Make validation generators
    ############################################
    generator_config = {
            'IMAGE_H'         : yolo.input_size, 
            'IMAGE_W'         : yolo.input_size,
            'GRID_H'          : yolo.grid_h,  
            'GRID_W'          : yolo.grid_w,
            'BOX'             : yolo.nb_box,
            'LABELS'          : yolo.labels,
            'CLASS'           : len(yolo.labels),
            'ANCHORS'         : yolo.anchors,
            'BATCH_SIZE'      : config['train']['batch_size'],
            'TRUE_BOX_BUFFER' : yolo.max_box_per_image,
        }
    
    valid_generator = BatchGenerator(valid_imgs, 
                                 generator_config, 
                                 norm=yolo.feature_extractor.normalize,
                                 jitter=False) 

    ############################################
    # Compute mAP on the validation set
    ############################################
    average_precisions = yolo.evaluate(valid_generator)     

    # print evaluation
    for label, average_precision in average_precisions.items():
        print(yolo.labels[label], '{:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions))) 

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)