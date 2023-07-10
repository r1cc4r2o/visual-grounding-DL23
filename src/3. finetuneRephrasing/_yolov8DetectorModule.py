import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from matplotlib import pyplot as plt
from _utils import *
from matplotlib.patches import Rectangle

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class YOLOv8DetectorModule(nn.Module):
    def __init__(self, model, preprocess, device):
        super(YOLOv8DetectorModule, self).__init__()
        self.model_yolo = model
        self.preprocess = preprocess
        self.device = device

        # image normalization
        # mean-std input image normalization
        self.transform = transforms.Compose([
                                transforms.Resize(800),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

    def get_crops(self, yolov8_df, image):
        """ Get crops of the image from yolov8_df

        Args:
            yolov8_df (pd.DataFrame): DataFrame with yolov5 predictions
            image (np.array): image as np.array

        Returns:
            list: crops

        """
        crops = []
        image = image

        for box in yolov8_df.values:
            x_min, y_min, x_max, y_max, confidence = box[:5]

            if confidence > 0.5:
                crop = image[int(y_min):int(y_max), int(x_min):int(x_max)]
                crop = Image.fromarray((crop * 255).astype(np.uint8))
                crops.append(crop)

        # add the entire image
        crops.append(Image.fromarray(image))

        return crops

    def get_crops_preprocessed_for_clip(self, yolov8_df, image):
        """ Get crops from yolov8_df and preprocess them for CLIP

        Args:
            yolov8_df (pd.DataFrame): DataFrame with yolov5 predictions
            image (np.array): image as np.array

        Returns:
            torch.tensor: preprocessed crops

        """
        crops = []
        boxes = []
        image = image

        for box in yolov8_df.values:
            x_min, y_min, x_max, y_max, confidence = box[:5]
            if confidence > 0.3:
                crop = image[int(y_min):int(y_max), int(x_min):int(x_max)]
                crop = Image.fromarray((crop * 255).astype(np.uint8))#.convert('RGB')
                crop = self.preprocess(crop)
                crops.append(crop)
                boxes.append(torch.tensor([x_min, y_min, x_max, y_max]).type(torch.float32).unsqueeze(0))

        return torch.stack(crops).to(self.device), torch.cat(boxes).to(self.device)

    def plot_image_yolov8(self, results, image):
        """Plot the images with the boxes"""
        for i in results.xyxy[0]:
            if i[4] > 0.5: # if confidence is greater than 0.5
                # Create figure and axes
                _, ax = plt.subplots()

                boxes = i[:4]

                # Display the image
                ax.imshow(image)

                # Create a Rectangle patch
                x_min, y_min, width, height = boxes.tolist()
                ax.add_patch(Rectangle((x_min, y_min), width-x_min, height-y_min, linewidth=1, edgecolor='r', facecolor='none'))

                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)

                plt.show()

    def plot_image_bbox(self, boxes, images, captions):
        """Plot the images with the boxes"""
        for bbox, image, c in zip(boxes.tolist(), images, captions):
              # Create figure and axes
              _, ax = plt.subplots()

              # Display the image
              ax.imshow(image)

              # Create a Rectangle patch
              x_min, y_min, width, height = bbox
              ax.add_patch(Rectangle((x_min, y_min), width-x_min, height-y_min, linewidth=1, edgecolor='r', facecolor='none'))

              ax.axes.xaxis.set_visible(False)
              ax.axes.yaxis.set_visible(False)

              plt.title(c[0])

              plt.show()

    def forward(self, images_path):

        print('Detecting objects in the images...')
        # get yolov8 predictions
        results = self.model_yolo([load_image_pil(i) for i in images_path])

        print('Boxes detected!')

        # store the predictions into a dataframe
        results = [pd.DataFrame(result.boxes.boxes.tolist(), columns=['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class']) for result in results]

        print('Preprocessing crops with CLIP...')

        # preprocess the images for CLIP
        cropPreprocessed_boxes = [
            self.get_crops_preprocessed_for_clip(yolov8_df, image)
                    for yolov8_df, image in zip(results, [load_image_plt(i) for i in images_path])
                ]

        imgs_preproc = [i[0] for i in cropPreprocessed_boxes]
        yolo_boxes = [i[1] for i in cropPreprocessed_boxes]

        if len(yolo_boxes) == 1:
            yolo_boxes = yolo_boxes[0].unsqueeze(0).to(torch.float32)
            imgs_preproc = imgs_preproc[0].unsqueeze(0).to(torch.float32)
        else:
            yolo_boxes = torch.cat(yolo_boxes).to(torch.float32)
            imgs_preproc = torch.cat(imgs_preproc).to(torch.float32)

        return imgs_preproc, yolo_boxes