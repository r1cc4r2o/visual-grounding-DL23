


import torch
import matplotlib.pyplot as plt
from torchvision import transforms as T
import clip
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import time
import pandas as pd
from ultralytics import YOLO


# Number of images to test
N_img = 10

# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True) # yolo5l6
model = YOLO("yolov8x.pt")

print(clip.available_models())

clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

def get_crops(yolov5_df, image):
    """ Get crops from yolov8_df 

    Args:
        yolov5_df (pd.DataFrame): DataFrame with yolov5 predictions
        image (np.array): image as np.array

    Returns:
        list: crops
    
    """
    crops = []
    image = image

    for box in yolov5_df.values:
        x_min, y_min, x_max, y_max, confidence = box[:5]

        if confidence > 0.5:
            crop = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            crop = Image.fromarray(crop)
            crops.append(crop)
        
    # add the entire image
    crops.append(Image.fromarray(image))

    return crops

def get_crops_preprocessed_for_clip(yolov8_df, image):
    """ Get crops from yolov5_df and preprocess them for CLIP

    Args:
        yolov8_df (pd.DataFrame): DataFrame with yolov5 predictions
        image (np.array): image as np.array

    Returns:
        torch.tensor: preprocessed crops
    
    """
    crops = []
    image = image

    for box in yolov8_df.values:
        x_min, y_min, x_max, y_max, confidence = box[:5]

        if confidence > 0.5:
            crop = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            crop = Image.fromarray(crop)
            crop = preprocess(crop).cuda()
            crops.append(crop)
        
    # add the entire image
    crops.append(preprocess(Image.fromarray(image)).cuda())

    return torch.stack(crops)

def plot_image_yolov8(results, image):
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

def get_dict_clip_emb_texscores(clip_model, img_preproc, text, idxs):
    """ Get dictionary with CLIP embeddings and text scores

    Args:
        clip_model (CLIP): CLIP model
        img_preproc (torch.tensor): preprocessed crops
        text (list): tokenized text
    
    Returns:
        dict: dictionary with CLIP embeddings and text scores
    
    """
    d_emb_texscores = {}

    for idx, images, captions in zip(idxs, img_preproc, text):
        temp = {}
        with torch.no_grad():
            # get CLIP embeddings
            image_features = clip_model.encode_image(images)
            text_features = clip_model.encode_text(captions)

        temp['image_emb'] = image_features.cpu()
        temp['text_emb'] = text_features.cpu()

        # normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        temp['text_similarity'] = (100.0 * text_features @ image_features.T).softmax(dim=-1).cpu()

        # print("Label probs:", similarity.cpu().numpy())

        d_emb_texscores[idx] = temp

    return d_emb_texscores


# Inference
def load_image_plt(ref):
    return plt.imread('../extractCOCO/refcocog/images/'+ref['file_name'].split('/')[-1])   
def load_image_pil(ref):
    return Image.open('../extractCOCO/refcocog/images/'+ref['file_name'].split('/')[-1]) 

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


import os
import json

from torch.utils.data import Dataset

class RefCOCOg(Dataset):
    def __init__(self, refs, annotations, split="train"):

        self.dataset = [{"file_name": os.path.join("./refcocog/images/", f'{"_".join(elem["file_name"].split("_")[:3])}.jpg'),
                            "caption": [i["raw"] for i in elem["sentences"]],
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
test_dataset_full = RefCOCOg(refs=refs, annotations=annotations, split="test")
# test_dataset_full = RefCOCOg(refs=refs, annotations=annotations, split="val")
# test_dataset_full = RefCOCOg(refs=refs, annotations=annotations, split="train")


print('len test datasets:',sum([len(i['caption']) for i in test_dataset_full]))

import pickle

# initialize steps for the loop
min_image = 0
max_image = len(test_dataset_full)
# batch of 50 images
N_images_batches = 40
steps = int(max_image/N_images_batches)
steps = np.linspace(min_image,max_image,steps).astype(int)

dictionary_full = {}

for m, M in zip(steps[:max_image-1], steps[1:]):
    
    print('step:',m,M)
    # initialize patch
    test_dataset = test_dataset_full[m:M]
    N_img = len(test_dataset)

    start = time.time()
    results = model([load_image_pil(test_dataset[i]) for i in range(N_img)])

    # Results
    # results.print()

    # store the df
    # results = [i for i in results.pandas().xyxy]
    results = [pd.DataFrame(result.boxes.boxes.tolist(), columns=['x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class']) for result in results]

    # plot_image_yolov8(results, np.array(load_image_plt(test_dataset[0])))
        
    img_preproc = [
        get_crops_preprocessed_for_clip(yolov8_df, image) 
            for yolov8_df, image in zip(results, [load_image_plt(test_dataset[i]) for i in range(N_img)])
            ]

    # img_preproc[0].shape

    # tokenize the text with clip
    text = [clip.tokenize([caption for caption in test_dataset[idx]['caption']]).cuda() for idx in range(N_img)]
    captions = [[caption for caption in test_dataset[idx]['caption']] for idx in range(N_img)]

    d_emb_texscores = get_dict_clip_emb_texscores(clip_model, img_preproc, text, np.arange(m,M))
    end = time.time()
    print('time:', end - start)


    for i, (idx, df_boxes, caption) in enumerate(zip(d_emb_texscores.keys(), results, captions)):
        d_emb_texscores[idx]['df_boxes'] = df_boxes
        d_emb_texscores[idx]['caption'] = caption
        d_emb_texscores[idx]['bbox_target'] = test_dataset[i]['bbox']

    dictionary_full.update(d_emb_texscores)

    # # save checkpoint
    # with open(f'./data/yolo_v8x/1_dictionary_full_train_checkpoints.p', 'wb') as fp:
    #     pickle.dump(dictionary_full, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # save checkpoint
    with open(f'./data/yolo_v8x/1_dictionary_full_test_checkpoints.p', 'wb') as fp:
        pickle.dump(dictionary_full, fp, protocol=pickle.HIGHEST_PROTOCOL)


with open('./data/yolo_v8x/1_dictionary_full_test.p', 'wb') as fp:
    pickle.dump(dictionary_full, fp, protocol=pickle.HIGHEST_PROTOCOL)

# with open('./data/yolo_v8x/1_dictionary_full_train.p', 'wb') as fp:
#     pickle.dump(dictionary_full, fp, protocol=pickle.HIGHEST_PROTOCOL)

