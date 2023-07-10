


import torch
import matplotlib.pyplot as plt
from torchvision import transforms as T
import clip
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import time

# Number of images to test
N_img = 10

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True) # yolo5l6

print(clip.available_models())

clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

def get_crops(yolov5_df, image):
    """ Get crops from yolov5_df 

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

        if confidence > 0.7:
            crop = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            crop = Image.fromarray(crop)
            crops.append(crop)
        
    # add the entire image
    crops.append(Image.fromarray(image))

    return crops

def get_crops_preprocessed_for_clip(yolov5_df, image):
    """ Get crops from yolov5_df and preprocess them for CLIP

    Args:
        yolov5_df (pd.DataFrame): DataFrame with yolov5 predictions
        image (np.array): image as np.array

    Returns:
        torch.tensor: preprocessed crops
    
    """
    crops = []
    image = image

    for box in yolov5_df.values:
        x_min, y_min, x_max, y_max, confidence = box[:5]

        if confidence > 0.7:
            crop = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            crop = Image.fromarray(crop)
            crop = preprocess(crop).cuda()
            crops.append(crop)
        
    # add the entire image
    crops.append(preprocess(Image.fromarray(image)).cuda())

    return torch.stack(crops)

def plot_image_yolov5l6(results, image):
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
# test_dataset_full = RefCOCOg(refs=refs, annotations=annotations, split="test")
test_dataset_full = RefCOCOg(refs=refs, annotations=annotations, split="train")


print('len test datasets:',sum([len(i['caption']) for i in test_dataset_full]))

import pickle

# initialize steps for the loop
min_image = 0
max_image = len(test_dataset_full)
# batch of 50 images
N_images_batches = 50
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
    results = [i for i in results.pandas().xyxy]

    # plot_image_yolov5l6(results, np.array(load_image_plt(test_dataset[0])))
        
    img_preproc = [
        get_crops_preprocessed_for_clip(yolov5_df, image) 
            for yolov5_df, image in zip(results, [load_image_plt(test_dataset[i]) for i in range(N_img)])
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

    # save checkpoint
    with open(f'./data/1_dictionary_full_train_checkpoints.p', 'wb') as fp:
        pickle.dump(dictionary_full, fp, protocol=pickle.HIGHEST_PROTOCOL)


# with open('./data/1_dictionary_full_test.p', 'wb') as fp:
#     pickle.dump(dictionary_full, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/1_dictionary_full_train.p', 'wb') as fp:
    pickle.dump(dictionary_full, fp, protocol=pickle.HIGHEST_PROTOCOL)

# test set
# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
# len test datasets: 9602
# step: 0 50
# time: 3.7851085662841797
# step: 50 101
# time: 3.9339659214019775
# step: 101 152
# time: 4.259373188018799
# step: 152 202
# time: 3.663057804107666
# step: 202 253
# time: 3.6984312534332275
# step: 253 304
# time: 3.7264747619628906
# step: 304 355
# time: 3.9506747722625732
# step: 355 405
# time: 3.5764055252075195
# step: 405 456
# time: 3.4356629848480225
# step: 456 507
# time: 3.5541203022003174
# step: 507 558
# time: 3.7669928073883057
# step: 558 608
# time: 3.55378794670105
# step: 608 659
# time: 3.674967050552368
# step: 659 710
# time: 3.717799186706543
# step: 710 761
# time: 3.9064669609069824
# step: 761 811
# time: 3.7030770778656006
# step: 811 862
# time: 3.9790546894073486
# step: 862 913
# time: 3.732635736465454
# step: 913 964
# time: 3.725435972213745
# step: 964 1014
# time: 3.323852777481079
# step: 1014 1065
# time: 3.3412907123565674
# step: 1065 1116
# time: 3.529946804046631
# step: 1116 1166
# time: 3.1902787685394287
# step: 1166 1217
# time: 3.5667171478271484
# step: 1217 1268
# time: 3.6234869956970215
# step: 1268 1319
# time: 3.463444948196411
# step: 1319 1369
# time: 3.539233684539795
# step: 1369 1420
# time: 3.657008171081543
# step: 1420 1471
# time: 3.627720355987549
# step: 1471 1522
# time: 3.2799322605133057
# step: 1522 1572
# time: 3.4321377277374268
# step: 1572 1623
# time: 3.4701263904571533
# step: 1623 1674
# time: 3.6144468784332275
# step: 1674 1725
# time: 3.4885332584381104
# step: 1725 1775
# time: 3.335554838180542
# step: 1775 1826
# time: 3.9475512504577637
# step: 1826 1877
# time: 3.6461219787597656
# step: 1877 1928
# time: 3.86903977394104
# step: 1928 1978
# time: 3.700349807739258
# step: 1978 2029
# time: 3.7860898971557617
# step: 2029 2080
# time: 3.3565735816955566
# step: 2080 2130
# time: 4.047240972518921
# step: 2130 2181
# time: 3.3285415172576904
# step: 2181 2232
# time: 3.427074432373047
# step: 2232 2283
# time: 3.410266399383545
# step: 2283 2333
# time: 3.3181045055389404
# step: 2333 2384
# time: 4.375714302062988
# step: 2384 2435
# time: 3.44759464263916
# step: 2435 2486
# time: 3.4983670711517334
# step: 2486 2536
# time: 3.573767900466919
# step: 2536 2587
# time: 3.625868558883667
# step: 2587 2638
# time: 3.5842325687408447
# step: 2638 2689
# time: 3.5285186767578125
# step: 2689 2739
# time: 3.78011155128479
# step: 2739 2790
# time: 4.052789211273193
# step: 2790 2841
# time: 3.587026834487915
# step: 2841 2892
# time: 3.1533756256103516
# step: 2892 2942
# time: 3.301790475845337
# step: 2942 2993
# time: 3.267988681793213
# step: 2993 3044
# time: 3.2469754219055176
# step: 3044 3094
# time: 3.142981767654419
# step: 3094 3145
# time: 3.333580732345581
# step: 3145 3196
# time: 3.3435099124908447
# step: 3196 3247
# time: 3.214261531829834
# step: 3247 3297
# time: 3.5023903846740723
# step: 3297 3348
# time: 3.2649731636047363
# step: 3348 3399
# time: 3.2073469161987305
# step: 3399 3450
# time: 3.3141438961029053
# step: 3450 3500
# time: 3.232750415802002
# step: 3500 3551
# time: 3.4772465229034424
# step: 3551 3602
# time: 3.4969966411590576
# step: 3602 3653
# time: 3.6865077018737793
# step: 3653 3703
# time: 3.295884132385254
# step: 3703 3754
# time: 3.3736042976379395
# step: 3754 3805
# time: 3.753462791442871
# step: 3805 3856
# time: 3.8165509700775146
# step: 3856 3906
# time: 3.487485408782959
# step: 3906 3957
# time: 3.2246971130371094
# step: 3957 4008
# time: 3.1838343143463135
# step: 4008 4058
# time: 3.407219409942627
# step: 4058 4109
# time: 3.337991952896118
# step: 4109 4160
# time: 3.280672311782837
# step: 4160 4211
# time: 3.9004948139190674
# step: 4211 4261
# time: 3.195539712905884
# step: 4261 4312
# time: 3.2926371097564697
# step: 4312 4363
# time: 3.461406946182251
# step: 4363 4414
# time: 3.2154738903045654
# step: 4414 4464
# time: 3.4142873287200928
# step: 4464 4515
# time: 3.2455101013183594
# step: 4515 4566
# time: 3.3749961853027344
# step: 4566 4617
# time: 3.5733723640441895
# step: 4617 4667
# time: 3.235461950302124
# step: 4667 4718
# time: 3.257707118988037
# step: 4718 4769
# time: 3.2723894119262695
# step: 4769 4820
# time: 3.596256971359253
# step: 4820 4870
# time: 3.3832147121429443
# step: 4870 4921
# time: 3.459261178970337
# step: 4921 4972
# time: 3.340383768081665
# step: 4972 5023
# time: 3.519461154937744
