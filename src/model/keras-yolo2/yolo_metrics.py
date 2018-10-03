#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 11:00:27 2018

@author: vineeth
"""

import os
import numpy as np
from preprocessing import parse_annotation, BatchGenerator
from frontend import YOLO
import json
import argparse
from sklearn.metrics import confusion_matrix, classification_report


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')


# the script calculates metrics over the validation dataset
# further, it is assumed that only object (and hence one class label) is present per image


def _main_(args):
    config_path  = args.conf
    weights_path = args.weights

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

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
    

    ###############################
    #   Parse the annotations 
    ###############################
    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'], 
                                                    config['valid']['valid_image_folder'], 
                                                    config['model']['labels'])
    else:
        raise ValueError('Validation folder does not exist or is not specified')
    
    
    
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
    
    generator = BatchGenerator(valid_imgs, 
                                 generator_config, 
                                 norm=yolo.feature_extractor.normalize,
                                 jitter=False) 
    
    y_true = []
    y_predicted = []
    for i in range(generator.size()):
        raw_image = generator.load_image(i)
        raw_height, raw_width, raw_channels = raw_image.shape

        # make the boxes and the labels
        pred_boxes  = yolo.predict(raw_image)
        
        score = np.array([box.score for box in pred_boxes])
        pred_labels = np.array([box.label for box in pred_boxes])     
        
        if len(pred_boxes) > 0:
            pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score] for box in pred_boxes])
        else:
            pred_boxes = np.array([[]])  
        
        # sort the boxes and the labels according to scores
        score_sort = np.argsort(-score)
        pred_labels = pred_labels[score_sort]
        pred_boxes  = pred_boxes[score_sort]
        
        # store predicted label for the image i. 
        # since multiple boxes may be predicted, choose the one with the highest score
        # TODO: find out why there are no predictions at all for certain images
        if pred_labels.any():
            y_predicted.append(pred_labels[0])
        else:
            y_predicted.append(4)
        
        # load true image annotations
        annotations = generator.load_annotation(i)
        
        if annotations.shape[0] > 1:
            raise ValueError('Multiple objects exist per image not supported')
        
        ### store the true label for the image i
        y_true.append(annotations[0,4])
    
    print('Processed ' + str(len(y_true)) + 'imgaes' )
    
    print('Confusion Matrix')
    print(confusion_matrix(y_true, y_predicted))
    print('Classification Report')
    
    # added NoPrediction label to number of classes as yolo model returned null prediction for some images
    target_names = config['model']['labels'] +['NoPrediction']
    print(classification_report(y_true, y_predicted, target_names=target_names))


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
    
