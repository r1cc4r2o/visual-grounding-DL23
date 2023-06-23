import pickle
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2

base_path_images = '/home/rickbook/document/dl/extractCOCO/refcocog/images/'
base_path_annotations = '/home/rickbook/document/dl/extractCOCO/refcocog/annotations/'

###############################################
######### 1. plot the bounding box of the image
###############################################


def class_to_color(class_id: int = 0):
    # we can chose the color of the bounding box
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,100,100),
              (100,255,100),(100,100,255),(255,100,0),(255,0,100),(100,0,255),(100,100,255),(100,255,0),
              (100,255,100)]
    return colors[class_id]

# draw a single bounding box onto a numpy array image
def draw_bounding_box(img, annotation, sent):

    # annotation = 'bbox': [x_min, y_min, width, height]
    # source: https://github.com/cocodataset/cocoapi/issues/34
    # https://github.com/shoumikchow/bbox-visualizer/blob/db8f331efc28536169b58985083e358c89bd43d3/bbox_visualizer/bbox_visualizer.py
    # line 41
    x_min, y_min = int(annotation[0]), int(annotation[1]) # x_min, y_min
    x_max, y_max = int(annotation[2]), int(annotation[3]) # x_max, y_max 
    
    # color = (255,0,0)
    # we can chose the color of the bounding box
    color = class_to_color()
    
    # draw the bounding box
    plt.imshow(cv2.rectangle(img,(x_min,y_min),(x_min+x_max,y_min+y_max), color, 2))
    plt.title(sent)
    # plt.axes('off')
    plt.show()

# draw all annotation bounding boxes on an image
def annotate_image(img, bbox, sent):
    draw_bounding_box(img, bbox, sent)


###############################################
######## 2. extract the data from the json file
###############################################

path = '/home/rickbook/document/dl/extractCOCO/refcocog/annotations/instances.json'
json_data = json.load(open(path, "r"))

# print the first image
# print(json_data['images'][0])

# print all keys of the json file
for key in json_data.keys():
    print(key)

# extract an image and its annotation
idx = 0

# annotation of the data to use to train the model
path = '/home/rickbook/document/dl/extractCOCO/refcocog/annotations/refs(google).p'
print([i for i in pickle.load(open(path, "rb")) if i['image_id'] == json_data['images'][idx]['id']])
# path = '/home/rickbook/document/dl/extractCOCO/refcocog/annotations/refs(umd).p'
# print(pickle.load(open(path, "rb"))[0])

sent = [i for i in pickle.load(open(path, "rb")) if i['image_id'] == json_data['images'][idx]['id']][0]['sentences'][0]['raw']


print()
print(json_data['images'][idx])
print()
print([i for i in json_data['annotations'] if i['image_id'] == json_data['images'][idx]['id']])
print()
print(json_data['info'])
print()

annotation = [i for i in json_data['annotations'] if i['image_id'] == json_data['images'][idx]['id']][0]


# load the image
img = mpimg.imread(base_path_images + json_data['images'][idx]['file_name'])
# plt.imshow(mpimg.imread(base_path_images + json_data['images'][0]['file_name']))

# annotate the image with the bounding box
annotate_image(img, annotation['bbox'], sent)

# show the image
plt.show()
