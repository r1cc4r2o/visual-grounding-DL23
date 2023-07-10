import clip
import time
import os
import json
import torch
import pickle
import numpy as np
from PIL import Image, ImageFilter 
import matplotlib.pyplot as plt
from torch import nn


import clip
import torch
import time
# source: https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=Y6Jrz6xz71C0
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')

# load the model from the torch hub
clip_model, preprocess = clip.load("RN50", device=device)
detr = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True).cuda()


#####################################################
############# functions for pre-processing boxes ####
#####################################################

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#####################################################

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

#####################################################

def rescale_bboxes(out_bbox, size, device):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b

#####################################################

def detect(im, model, transform, device):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).to(device)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    # assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.5
    # keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size, device)
    return probas[keep], bboxes_scaled

#####################################################

def blur_out_detr(image, box):
    """ Blur the image out of the bounding box
    
    """
    x_min, y_min, x_max, y_max = box

    # blur the image
    im = np.array((image).filter(ImageFilter.GaussianBlur(radius = 30)))

    # create the mask
    mask = np.zeros_like(np.array(image))

    # get blur image out of the bounding box
    y_1, y_2 = int(y_min), int(y_max)
    x_1, x_2 = int(x_min), int(x_max)

    # blur the image
    mask[y_1:y_2,x_1:x_2] = np.array(image)[y_1:y_2,x_1:x_2]
    im[y_1:y_2,x_1:x_2] = im[y_1:y_2,x_1:x_2]-im[y_1:y_2,x_1:x_2]
    im[y_1:y_2,x_1:x_2] = im[y_1:y_2,x_1:x_2] + mask[y_1:y_2,x_1:x_2]

    # return the image blurred
    return Image.fromarray(im)

#####################################################

def blur_in_detr(image, box):
    """ Blur the image in the bounding box
    
    """
    x_min, y_min, x_max, y_max = box

    # blur the image
    im = np.array((image).filter(ImageFilter.GaussianBlur(radius = 30)))

    # create the mask
    mask = np.zeros_like(np.array(image))

    # get blur image out of the bounding box
    y_1, y_2 = int(y_min), int(y_max)
    x_1, x_2 = int(x_min), int(x_max)

    # blur the image
    mask[y_1:y_2,x_1:x_2] = np.array(image)[y_1:y_2,x_1:x_2]

    # blur the image in the box
    remove_box = np.array(image)-mask
    im = np.array((image).filter(ImageFilter.GaussianBlur(radius = 30)))
    remove_box[y_1:y_2,x_1:x_2] = remove_box[y_1:y_2,x_1:x_2] + np.array(im)[y_1:y_2,x_1:x_2]

    return Image.fromarray(remove_box)

#####################################################

def get_img_blur_outin(Images, detected):
    """ Return a list of tuple of pillow images
    [[(blur_out, blur_in),...],[...]] 
    Dimensions of the list:
        dim_1: number of images
        dim_2: number of boxes found by detr
    Args:
        


    """
    return [[(blur_out_detr(I, d[1][idx].tolist()),blur_in_detr(I, d[1][idx].tolist()))
                for idx in range(len(d[1]))]
                    for I, d in zip(Images, detected)]

#####################################################

def get_img_blur_outin_preprocessed(Images, detected):
    """Return a list of tuple of pillow images preprocessed
    with CLIP preprocessor

    Args:
        Images (list): list of pillow images
        detected (list): list of detected boxes with detr

    Returns:
        _list_: list of preprocessed images blur_out, blur_in
    """
    return [[torch.stack([preprocess(i).unsqueeze(0) 
                for i in [blur_out_detr(I, d[1][idx].tolist()),blur_in_detr(I, d[1][idx].tolist())]]).squeeze(1)
                for idx in range(len(d[1]))]
                for I, d in zip(Images, detected)]

#####################################################

def get_listed_img_blur_outin(Images, detected):
    """ Return the list of list with the boxex
    blurd in, blurd out.

    example:
        [   
           [
            [list of img with box blurd out],
            [list of img with box blurd in]
           ],
           [
            [list of img with box blurd out],
            [list of img with box blurd in]
           ],
           ...
        ]
    Args:
        Images (list): list of pillow images
        detected (list): list of detected boxes with detr

    Returns:
        _list_: list of list of images blur_out, blur_in
    
    """
    
    return [[torch.stack([i[0] for i in img]), # blur out 
            torch.stack([i[1] for i in img])] # blurd in
                for img in get_img_blur_outin_preprocessed(Images, detected)]

#####################################################

def get_prob_clip(clip, clip_model, samples_json, blur_outin, offset, device):
    
    probabilities = []
    for idx, sample in enumerate(blur_outin):
        text = clip.tokenize([samples_json[idx+offset]["caption"]]).to(device)

        # to do not keep track of the computational
        # graph to compute the gradients
        with torch.no_grad():
            _, logits_per_text_blur_out = clip_model(sample[0].to(device), text)
            _, logits_per_text_blur_in = clip_model(sample[1].to(device), text)
            probs_blur_out = logits_per_text_blur_out.softmax(dim=-1).cpu().numpy().round(4).type(torch.float16)
            probs_blur_in = logits_per_text_blur_in.softmax(dim=-1).cpu().numpy().round(4).type(torch.float16)
            probs = torch.stack([probs_blur_out, probs_blur_in])
        probabilities.append(probs.squeeze(1))

    return probabilities

#####################################################

def get_prob_boxes_clip(clip, clip_model, samples_json, blur_outin, device, boxes):
    
    # probabilities_boxes = []
    dict_prob_boxes = {}
    for idx, (sample, b) in enumerate(zip(blur_outin, boxes)):
        text = clip.tokenize([samples_json[idx]["caption"]]).to(device)

        # to do not keep track of the computational
        # graph to compute the gradients
        with torch.no_grad():
            emb_blur_out = clip_model.encode_image(sample[0].to(device)) # get embeddings blur_out
            emb_text = clip_model.encode_text(text) # get embeddings text
            _, logits_per_text_blur_out = clip_model(sample[0].to(device), text)
            _, logits_per_text_blur_in = clip_model(sample[1].to(device), text)
            probs_blur_out = logits_per_text_blur_out.softmax(dim=-1).cpu().numpy().round(4).astype(np.float16)
            probs_blur_in = logits_per_text_blur_in.softmax(dim=-1).cpu().numpy().round(4).astype(np.float16)
            probs = np.stack([probs_blur_out, probs_blur_in])
        # to get the boxes in the refCOCO format
        # x_min, y_min, x_max-x_min, y_max-y_min
        b[:,2] = b[:,2] - b[:,0]
        b[:,3] = b[:,3] - b[:,1]
        # probabilities_boxes.append(torch.concat([torch.from_numpy(probs.squeeze(1)),b.cpu().permute(1,0)],0).permute(1,0))
        temp = dict()
        temp['prob-box-map'] = torch.concat([torch.from_numpy(probs.squeeze(1)),b.cpu().permute(1,0)],0).permute(1,0)
        temp['embeds-boxes'] = emb_blur_out
        temp['embeds-caption'] = emb_text

        dict_prob_boxes[idx] = temp

    return dict_prob_boxes

#################################################
################################ load the dataset
#################################################

import os
import json

from torch.utils.data import Dataset

class RefCOCOg(Dataset):
    def __init__(self, refs, annotations, split="train"):

        self.dataset = [{"file_name": os.path.join("./refcocog/images/", f'{"_".join(elem["file_name"].split("_")[:3])}.jpg'),
                            "caption": elem["sentences"][0]["raw"],
                            "ann_id": int(elem["file_name"].split("_")[3][:-4]),
                            "bbox": annotations[int(elem["file_name"].split("_")[3][:-4])]}
                        for elem in [d for d in refs if d["split"]==split]]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __call__(self, idx):
        print(json.dumps(self.dataset[idx], indent=4))


# Load refs and annotations
import pickle

with open("../extractCOCO/refcocog/annotations/refs(umd).p", "rb") as fp:
  refs = pickle.load(fp)

with open("../extractCOCO/refcocog/annotations/instances.json", "rb") as fp:
  data = json.load(fp)
  annotations = dict(sorted({ann["id"]: ann["bbox"] for ann in data["annotations"]}.items()))


# load the train dataset
train_dataset = RefCOCOg(refs=refs, annotations=annotations, split="train")

print('len training datasets:',len(train_dataset))





# initialize steps for the loop
min_image = 0
max_image = 10000
# batch of 10 images
N_images_batches = 10
steps = int(10000/10)
steps = np.linspace(min_image,max_image,steps).astype(int)

from tqdm import tqdm

# loop over the images in refCOCOg
for m, M in tqdm(zip(steps[:max_image-1], steps[1:])):
    
  try:
    # load the images
    Images = [Image.open('../extractCOCO/'+sample["file_name"][1:]) for sample in train_dataset[m:M]]

    # detect the boxes
    detected = [detect(im, detr, transform, device) for im in Images]

    # get the blur out and blur in images
    # preprocessed with clip prepeocessor
    blur_outin = get_listed_img_blur_outin(Images, detected)

    # extract the boxes found
    boxes = [d[1].round().type(torch.float16) for d in detected]

    # get a copy of the current json samples
    json_samples = train_dataset[m:M]

    # get a dict with:
    # ['prob-box-map'] get the probabilities concateneted with the boxes
    # ['embeds-boxes'] get the CLIP embeddings of the boxes blur out
    # ['embeds-caption'] get the CLIP embeddings of the caption
    dict_prob_boxes = get_prob_boxes_clip(clip, clip_model, json_samples, blur_outin, device, boxes)

    # save the dict
    with open(f'./data_refcoco/dict_preprocessedimages_{m}_{M}.p', 'wb') as fp:
      pickle.dump(dict_prob_boxes, fp, protocol=pickle.HIGHEST_PROTOCOL)

  except:
    print('error with images:',m,M)