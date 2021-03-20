# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:45:35 2021

@author: ibrahim
"""
import cv2
import os

import xml.etree.ElementTree as ET
import argparse


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = ''
    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(float(boxes.find("bndbox/ymin").text))
        xmin = int(float(boxes.find("bndbox/xmin").text))
        ymax = int(float(boxes.find("bndbox/ymax").text))
        xmax = int(float(boxes.find("bndbox/xmax").text))

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes

   
parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

print('PESMOD sequence path,', args.path)


imgPath = args.path + "/images/"
file_list = sorted(os.listdir(imgPath))

for filename in file_list:
    
    frame = cv2.imread( imgPath + filename)

    annoPath = imgPath.replace("images", "annotations")
    name, boxes = read_content(annoPath + filename.replace(".jpg", ".xml"))
    for box in boxes:
        p1 = (box[0], box[1])
        p2 = (box[2], box[3])
        cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
                

    frame = cv2.resize(frame, (960, 540))
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()