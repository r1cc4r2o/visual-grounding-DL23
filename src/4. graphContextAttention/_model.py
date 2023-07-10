import torch
import torch.nn as nn
import torch.nn.functional as F

from ._contextAttentionModule import ContextAttentionModule
from ._clipScoreBlock import CLIPscoreBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CLIP_GraphDetectorModule(nn.Module):
    def __init__(self, clip_model, clip_tokenizer, yolo_module, finetuned_model, device):
        super().__init__()

        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.yolo_module = yolo_module
        self.finetuned_model = finetuned_model
        self.device = device

    def get_image_features(self, images):
        """Get the clip image features"""

        print('Encoding images...')

        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_text_features(self, text):
        """Get the clip text features"""

        print('Encoding text...')

        with torch.no_grad():
            text_features = self.clip_model.encode_text(self.clip_tokenizer([t for t in text]).to(device))
        # Normalize features
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_score_similarities(self, image_features, text_features):
        """Get the clip similarities"""
        # Compute the score similarities
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarities

    def detector_inference(self, images):
        # get preprocessed crops of the objects
        # in the original images
        x = [self.yolo_module(image) for image in images]
        max_crop_shape = max([crop[1].shape[1] for crop in x])
        imgs_crops, yolo_boxes = [crop[0] for crop in x], torch.cat([torch.nn.functional.pad(crop[1], (0, 0, max_crop_shape-crop[1].shape[1], 0)) for crop in x])
        return imgs_crops, yolo_boxes

    def box_norm_rescale(self, box_target):
        """ Rescale the box_target
        Args:
            box_target: (number of samples, 1, 4)

        Returns:
            box_target: (number of samples, 1, 4)

        """
        # convert the box_pred to x1, y1, x2, y2
        box_target[:, 0, 2] = box_target[:, 0, 0] + box_target[:, 0, 2]
        box_target[:, 0, 3] = box_target[:, 0, 1] + box_target[:, 0, 3]

        return box_target

    def forward(self, x, captions):
        # get the preprocessed crops of the objects in
        # the original images
        imgs_crops, yolo_boxes = self.detector_inference(x)

        print('Extracting features with CLIP...')

        # extract the clip text features
        if len(captions) == 1:
            caption_features = self.get_text_features(captions).unsqueeze(-1).unsqueeze(1).to(torch.float32)
        else:
            caption_features = torch.stack(
                [self.get_text_features(caption) for caption in captions]
            ).unsqueeze(-1).to(torch.float32)

        # extract the clip image features
        images_features = [self.get_image_features(img_crops.squeeze(0)).unsqueeze(0) for img_crops in imgs_crops]
        max_shape = max([crops.shape[1] for crops in images_features])
        images_features = torch.cat([torch.nn.functional.pad(crops, (0, 0, max_shape-crops.shape[1], 0)) for crops in images_features]).permute(0, 2, 1).unsqueeze(1).to(torch.float32)


        print('Prepare the data...')

        # check that the number of boxes are > 4
        if yolo_boxes.shape[-2] < 4:
            # padd the image features
            images_features = F.pad(images_features, (0, 0, 0, 0, 0, 4 - yolo_boxes.shape[-2]))

        yolo_boxes = self.box_norm_rescale(yolo_boxes)

        pred = self.finetuned_model(caption_features, images_features, yolo_boxes)

        return pred