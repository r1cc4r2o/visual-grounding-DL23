import pickle
import json
import torch
import clip
from PIL import Image
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm


# path 

base_path_images = '/home/rickbook/document/dl/extractCOCO/refcocog/images/'
base_path_annotations = '/home/rickbook/document/dl/extractCOCO/refcocog/annotations/'

# annotation of the data to use to train the model
path_annotation =  base_path_annotations+'refs(google).p'

path_instances = '/home/rickbook/document/dl/extractCOCO/refcocog/annotations/instances.json'

# index of the image to use
idx = 1



def get_data(full_data, json_data, model, device):
    """ This function extract the data from the json file and the annotation file.
    It returns a dictionary with the following keys:
        dictionary = {idx: {
            'image_id': None,
            'sentences': None,
            'boxe': None,
            'category_id': None,
            'attn_map': None,
            'fname_image': None
        }
        ...
        }

    Args:
        full_data: list of dictionaries
        json_data: dictionary
        model: clip model
        device: device to use

    Returns:
        dictionary: dictionary with the data

    """
    dictionary = dict()

    for idx, data in tqdm(enumerate(full_data[8000:12000])):
        # initialize the dictionary
        d = dict()

        # get the file name of the image
        file_name = [i['file_name'] for i in json_data['images'] if i['id'] == data['image_id']][0]

        # get the box of the image
        box = [i['bbox'] for i in json_data['annotations'] if i['image_id'] == data['image_id']][0]


        # add the file name to the dictionary
        d['fname_image'] = file_name

        # add the image id to the dictionary
        d['image_id'] = data['image_id']

        # add the sentences to the dictionary
        d['sentences'] = data['sentences']

        # add the category id to the dictionary
        d['category_id'] = data['category_id']

        # add the box to the dictionary
        d['boxe'] = box

        # load the image
        img = mpimg.imread(base_path_images + file_name)
        image = preprocess(Image.fromarray(np.array(img))).unsqueeze(0).to(device)

        # initialize the list of attention maps
        attn_maps = []

        for sent in data['sentences']:
            # tokenize the text
            text = clip.tokenize([sent['raw']]).to(device)

            # encode the image and the text
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

            # compute the attention map
            # append the attention map to the list
            attn_maps.append(image_features.permute(1,0) @ text_features)

        # add the attention maps to the dictionary
        d['attn_map'] = attn_maps

        # add the dictionary to the return dictionary
        dictionary[idx] = d

    return dictionary



######################################
# main
######################################


# load the data

data = pickle.load(open(path_annotation, "rb"))
print('number of images: ',len(data))

# cuda or cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

# load the model clip
model, preprocess = clip.load("ViT-B/32", device=device)

# load the json file
json_data = json.load(open(path_instances, "r"))


# get the data
data = get_data(data, json_data, model, device)

# save the data
with open('data_8000-12000.pkl', 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)













# junk

# print(data[idx]['image_id'])

# # get the file name of the image
# file_name = [i['file_name'] for i in json_data['images'] if i['id'] == data[idx]['image_id']][0]

# # load the image
# img = mpimg.imread(base_path_images + file_name)
# image = preprocess(Image.fromarray(np.array(img))).unsqueeze(0).to(device)
# text = clip.tokenize([data[idx]['sentences'][0]['raw']]).to(device)


# # time
# start_time = time.time()

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)

# print("--- %s seconds ---" % (time.time() - start_time))

# # time
# start_time = time.time()

# att_n_map = image_features.permute(1,0) @ text_features

# print(att_n_map.shape)

# print("--- %s seconds ---" % (time.time() - start_time))

# print(att_n_map.cpu().numpy())

