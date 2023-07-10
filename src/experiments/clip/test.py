""" Here I test the model CLIP given different images.

The objective is to find the bounding box of the relative 
label given the image.

With CLIP we can compute the score 



"""


import numpy as np
import torch
import clip



print('available models: ',clip.available_models())


model, preprocess = clip.load("ViT-B/32")

# set the model in evel mode
model.cuda().eval()


input_resolution = model.visual.input_resolution

context_length = model.context_length

vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

print('Tokenization of the word person')
print(clip.tokenize('person'))


# Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print('Loaded YOLOv5 model')

# Load image
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

# Preprocess image
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)