#!/usr/bin/env python
# coding: utf-8

# In[4]:
from mrcnn import model
import base64
import os
import json
from sys import argv
import datetime
import numpy as np
import skimage.draw
import cv2
import matplotlib.pyplot as plt
import sys
from droplets import Poly_DropletInferenceConfig

if len(argv) > 2:
    GPU_INDEX = str(argv[2])
else:
    GPU_INDEX = '1'

if len(argv) > 3:
    EVERY = int(argv[3])
else:
    EVERY = 1

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_INDEX

CLASS_NAMES = ['BG', 'low_positive', 'negative', 'high_positive', 'medium_positive']

# Root directory of the project
# ROOT_DIR = os.path.abspath("/home/zidalad-szu/repos/Mask_RCNN")
ROOT_DIR = os.path.abspath("/home/zy/Covid19_MRCNN/Mask_RCNN")
# ROOT_DIR = os.path.abspath("H:/Mask R-CNN/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import model as modellib, utils
#from mrcnn import visualize

DROPLET_DIR = 'F:/dLamp/20210925/droplets_0809_new'

sys.path.insert(0, DROPLET_DIR)

from droplets import DropletConfig, DropletDataset

'''
Configurations

For inferencing, modify the configurations a bit to fit the task
'''


class DropletInferenceConfig(DropletConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    # IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.1
    PRE_NMS_LIMIT = 9000


    POST_NMS_ROIS_INFERENCE = 4000

    DETECTION_NMS_THRESHOLD = 0.7

    DETECTION_MIN_CONFIDENCE = 0.
    DETECTION_MAX_INSTANCES = 2000
    USE_MINI_MASK = False
    # IMAGE_MIN_DIM = 1024
    # IMAGE_MAX_DIM = 1024
    # IMAGE_MIN_SCALE = 1.0

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    # BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Length of square anchor side in pixels
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512) # (16, 32, 64, 128, 256, 512)


config = DropletInferenceConfig()

config.display()

'''
Create Model and Load Trained Weights
'''
# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

model = model.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)

# ATTENTION 1: MODEL

#weights_path = model.find_last()
weights_path = './model/0503_mask_rcnn_droplet_0120.h5'


model.load_weights(weights_path, by_name=True)


def apply_mask(image, mask, color, alpha=0.4):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(mask,
                                  image[:, :, n] * (1 - alpha) + alpha * c,
                                  image[:, :, n])
    return image


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 0, 128)]  # BGR，blue. green, red, purple— low、empty、medium、large

def dict_json(imageData, shapes, imagePath):
    '''
    :param imageData: str
    :param shapes: list
    :param imagePath: str
    :param fillColor: list
    :param lineColor: list
    :return: dict
    '''
    return {"version": "4.5.7", "flags": {}, "shapes": shapes, 'imagePath': imagePath,
            "imageData": imageData, "imageHeight": 1608, "imageWidth": 1608}


def dict_shapes(label, points):
    return {"label": label, "points": points, "shape_type": "polygon", "flags": {}}


def detect_droplet(image):

    image = np.array(image)

    r = model.detect([image])[0]
    boxes = r['rois']
    masks = r.pop('masks')
    class_ids = r['class_ids']
    class_names = [CLASS_NAMES[i] for i in class_ids]
    scores = r['scores']
    r['area'] = np.sum(np.sum(masks, axis=0), axis=0)
    n_instances = boxes.shape[0]

    image_json_shapes = []

    for i in range(n_instances):
        points = []
        class_id = class_ids[i]
        color = colors[class_id - 1]

        masks_detect_contour = masks[:, :, i]
        masks_detect_contour = masks_detect_contour.reshape(1608, 1608)
        masks_detect_contour = masks_detect_contour.astype(np.uint8)
        mask_contours, hierarchy = cv2.findContours(masks_detect_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for mask_contour in mask_contours:
            mask_contour = mask_contour.flatten()
            mask_contour = mask_contour.tolist()
            for index in range(len(mask_contour)):
                if index % 2 == 0:
                    points.append([mask_contour[index], mask_contour[index + 1]])
        if class_id == 1:
            image_json_shapes.append(dict_shapes('low_positive', points))
        elif class_id == 3:
            image_json_shapes.append(dict_shapes('medium_positive', points))
        elif class_id == 4:
            image_json_shapes.append(dict_shapes('high_positive', points))
        else:
            image_json_shapes.append(dict_shapes('negative', points))

        image = cv2.rectangle(image, (boxes[i, 3], boxes[i, 2]), (boxes[i, 1], boxes[i, 0]), color, 4)
        image = apply_mask(image, masks[:, :, i], color)
    return image, r, image_json_shapes


from collections import defaultdict
import pickle
import numpy as np


def parse_detect_results(result):
    rois = result['rois']  # [n, 4]
    n = rois.shape[0]
    if n == 0:
        return 0, 0, {1: [], 2: []}

    class_ids = result['class_ids']

    area = result['area']  #

    count = defaultdict(int)
    area_dict = defaultdict(list)

    for i, a in zip(class_ids, area):
        count[i] += 1
        area_dict[i].append(a)
    return count[1], count[2], area_dict

def parse_one(detection_results):
    result = {}
    result['positive_count_in_frame'] = []
    result['negative_count_in_frame'] = []
    result['positive_area'] = []
    result['negative_area'] = []
    area_dict = defaultdict(list)

    for d in detection_results[:]:
        c1, c2, dic = parse_detect_results(d)
        result['positive_count_in_frame'].append(c1)
        result['negative_count_in_frame'].append(c2)
        result['positive_area'].extend(dic[1])
        result['negative_area'].extend(dic[2])

    for k in result.keys():
        result[k] = np.array(result[k], dtype='int32').tolist()
    return result


def main():
    '''--------------two paths required to be modified----------'''
    image_path = './Sample images/Large/' # TO BE DETECTED
    save_path = './results/Sample images/Large/'
    '''---------------------------------------------------------'''

    image_save_path = save_path + 'image/'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    print("image path: " + image_path)
    print("image_save_path: " + image_save_path)

    for dir in os.listdir(image_path):
        image_dir = os.path.join(image_path, dir)
        image_save_name = image_save_path + dir.replace('tif', 'jpg')

        image = cv2.imread(image_dir)
        cv2.imencode('.jpg', image)[1].tofile(image_save_name)
        with open(image_save_name, "rb") as f:
            img_b64 = base64.b64encode(f.read())
            img_b64 = bytes.decode(img_b64)

        # OpenCV returns images as BGR, convert to RGB
        image = np.ascontiguousarray(image[..., ::-1])
        image, r, shapes = detect_droplet(image)
        cv2.imwrite(image_save_name, image)

        image_detect_data_json = dict_json(img_b64, shapes, image_dir)
        json_save_path = save_path + 'json/'
        if not os.path.exists(json_save_path):
            os.makedirs(json_save_path)
            print("json_save_path: " + json_save_path)
        json_save_name = json_save_path + dir.replace('tif', 'json')
        print("detecting: " + json_save_name)
        json.dump(image_detect_data_json, open(json_save_name, 'w'))
    print("image path: " + image_path)
    print("image_save_path: " + image_save_path)
    print("json_save_path: " + json_save_path)
    print("Finished!")

'''
Main function
'''
if __name__ == "__main__":
    main()
