#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 11:38:20 2018

@author: vineeth
"""


## Source: https://github.com/vineelyettella/not-so-random-forest/blob/master/utils/xml_to_csv.py

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse

argparser = argparse.ArgumentParser(description='Convert xml annotations to csv')

argparser.add_argument('-d',
                       '--dir',
                       help='path to directory containing annotation files')


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text),
                     member[0].text
                     )
            xml_list.append(value)
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main(args):
    dir_path = args.dir
    #dir_path = os.path.join(os.getcwd(), 'annotations')
    xml_df = xml_to_csv(dir_path)
    xml_df.to_csv('trash_labels.csv', index=None)
    print('Successfully converted xml to csv.')

if __name__ == '__main__':    
    args = argparser.parse_args()
    main(args)