import os
import numpy as np
import pandas as pd

import torchvision

from PIL import Image
import matplotlib.pyplot as plt



#########################################################


def load_image_plt(path):
    """ Load image with matplotlib"""
    return plt.imread(path)

#########################################################


def load_image_pil(path):
    """ Load image with PIL"""
    return Image.open(path)


#########################################################


def get_distance_box_iou_accuracy(box_pred, box_true, iou_threshold=0.5):
    """
    Given the target boxes and the prediction return the
    accuracy of the prediction. The accuracy is computed as
    the percentage of boxes that have an IoU > iou_threshold
    with the target box.

    Args:
    @params box_pred: tensor of shape (batch_size, n_boxes, 4)
    @params box_true: tensor of shape (batch_size, n_boxes, 4)
    @params iou_threshold: float

    Returns:
    @params accuracy: float

    """

    iou = torchvision.ops.box_iou(box_pred, box_true).diagonal()
    giou = torchvision.ops.generalized_box_iou(box_pred, box_true).diagonal()

    return (iou > iou_threshold).float().mean(), iou.mean(), giou.mean()