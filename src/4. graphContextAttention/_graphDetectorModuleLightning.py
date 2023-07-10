import torch
import torch.nn as nn

import pytorch_lightning as pl

from ._utils import get_distance_box_iou_accuracy
from ._graphDetectorModule import GraphDetectorModule

class GraphDetectorModuleLightning(pl.LightningModule):
    def __init__(self, dim_embedding = 512, max_num_boxes = 4) -> None:
        super().__init__()

        self.dim_embedding = dim_embedding
        self.max_num_boxes = max_num_boxes

        self.gam_l = GraphDetectorModule(dim_embedding, max_num_boxes)

        self.HUBER = nn.SmoothL1Loss()

        self.accuracy = get_distance_box_iou_accuracy

        # self.MSE = nn.MSELoss()
        # self.MAE = nn.L1Loss()
        # GENERALIZED_BOX_IOU_LOSS https://arxiv.org/abs/1902.09630
        # self.generalized_box_iou_loss = torchvision.ops.generalized_box_iou_loss
        # DISTANCE_BOX_IOU_LOSS https://arxiv.org/abs/1911.08287
        # self.distance_box_iou_loss = torchvision.ops.distance_box_iou_loss
        # COMPLETE_BOX_IOU_LOSS https://arxiv.org/abs/1911.08287
        # self.complete_box_iou_loss = torchvision.ops.complete_box_iou_loss


    def forward(self,text_feat, x, boxes):
        return self.gam_l(text_feat, x, boxes)

    def training_step(self, batch, batch_idx):

        text_feat, x, boxes, target = batch

        pred = self(text_feat, x, boxes)

        # keep track of the losses

        huber_loss = self.HUBER(pred, target)

        accuracy, mean_iou, mean_giou = self.accuracy(pred, target)

        self.log('train_huber_loss', huber_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log('train_accuracy', accuracy, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log('train_mean_iou', mean_iou, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log('train_mean_giou', mean_giou, on_step = True, on_epoch = True, prog_bar = True, logger = True)


        return huber_loss * (1.4 - mean_iou)

    def validation_step(self, batch, batch_idx):

        text_feat, x, boxes, target = batch

        pred = self(text_feat, x, boxes)

        # keep track of the losses

        huber_loss = self.HUBER(pred, target)

        accuracy, mean_iou, mean_giou = self.accuracy(pred, target)

        self.log('val_huber_loss', huber_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log('val_accuracy', accuracy, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log('val_mean_iou', mean_iou, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log('val_mean_giou', mean_giou, on_step = True, on_epoch = True, prog_bar = True, logger = True)

        return huber_loss * (1.4 - mean_iou)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = 0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20)
        return [optimizer], [scheduler]