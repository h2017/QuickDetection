#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from utils import draw_boxes
from frontend import YOLO
import json
import time

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

argparser.add_argument(
    '-i',
    '--input',
    help='path to directory containing test images')

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input
    
    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################
    model_load_time = time.time()
    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    print('Model load time is '+ str(time.time() - model_load_time))
    
    ###############################
    #   Predict bounding boxes 
    ###############################
    
    inference_time = []
    for filename in os.listdir(image_path):
        img_path = os.path.join(image_path, filename)
        inference_start_time = time.time()
        image = cv2.imread(img_path)
        boxes = yolo.predict(image)
        inference_time.append(time.time() - inference_start_time)
        print(len(boxes), 'boxes are found')
        image = draw_boxes(image, boxes, config['model']['labels'])
        cv2.imwrite(img_path[:-4] + '_detected' + img_path[-4:], image)
    print('Avg inference time is '+ str(np.mean(inference_time)) + '+/-' + str(np.std(inference_time)))

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
