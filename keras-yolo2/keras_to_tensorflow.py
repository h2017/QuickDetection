#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 19:33:28 2018
@author: vineeth
Inspired and based off of:
    1) https://github.com/amir-abdi/keras_to_tensorflow
    2) https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#optimizing-for-deployment
"""
#! /usr/bin/env python

import argparse
import os
import numpy as np
from frontend import YOLO
import json
import time


import tensorflow as tf
from tensorflow.python.summary import summary
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.tools.graph_transforms import TransformGraph
from keras import backend as K

from pathlib import Path


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Convert keras model into a frozen tensorflow graph')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-t',
    '--transforms',
    default = False,
    help='Activate this option for graph transforms')

argparser.add_argument(
    '-v',
    '--visualize',
    default = False,
    help='Actiate this option to visulaize graph in tensorboard')

# ToDo: checks for argument types that are passed.  
#       Output_node_name can be made as an external input.
    


# =============================================================================
#  Function to visulaize the graph in tensorboard and then find the output node names.
# =============================================================================

def graph_visualize(sess, log_dir = '/tmp/'):
     pb_visual_writer = summary.FileWriter(log_dir )
     pb_visual_writer.add_graph(sess.graph)
     print("Model Imported. Visualize by running the bash command: "
               "tensorboard --logdir={}".format(log_dir))
    


def _main_(args): 
    config_path  = args.conf
    weights_path = args.weights
    
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
    
    print('Keras model load time is %.2f seconds'%(time.time() - model_load_time))
    
    #  Get tensorflow session handle    
    sess = K.get_session()
    
    # Write the tf.graph summary so as to visualize graph in tensorboard
    if args.visualize:
        return graph_visualize(sess, log_dir = '/tmp/')
    
    
    # Names of the output nodes. My model has just one output node.
    # Note: It is not the final layer name, but an output node inside that layer and
    #       and is typically a identity node or activation function node 
    pred_node_names = ['lambda_2/Identity']
    

    # =============================================================================
    # Freeze graph : convert variables to constants and save
    # =============================================================================
    if args.transforms:
        # ToDo: Give the user to specify a subset of the below transforms
        transforms = ["add_default_attributes",
                      #"strip_unused_nodes(type=float, shape = '1,416,416,3')",
                      "remove_nodes(op=Identity, op=CheckNumerics)",
                      "fold_constants(ignore_errors=true)",
                      "fold_batch_norms",
                      "fold_old_batch_norms",
                      "quantize_weights",
                      "quantize_nodes",
                      "strip_unused_nodes",
                      "sort_by_execution_order"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names , transforms)
        constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names )
    else:
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)    
    
    # Write the frozen graph to a file in the present working directory
    output_filename = str(Path(weights_path).name)[:-3] + '.pb'
    graph_io.write_graph(constant_graph, '', output_filename, as_text=False)
    print('Saved the frozen graph (ready for inference) as: ', str(output_filename))
    
    K.clear_session()
        
if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

