import torch
import torch.nn as nn
import pytorch_lightning as pl

from ._utils import get_distance_box_iou_accuracy
from ._clipScoreBlock import CLIPscoreBlock
from ._linWeightingBlock import LinearBoxWeightingBlock

class FinetunedRephrasing(pl.LightningModule):
    def __init__(self, out_dim_box=512, latent_dim=240, hidden_dim_regressor=256):
        super().__init__()

        self.linWeighting = LinearBoxWeightingBlock()
        self.clipScoreBlock = CLIPscoreBlock()

        self.resizer = nn.Linear(1, 4)

        self.save_hyperparameters()

        # https://arxiv.org/pdf/2108.12627
        self.HUBER = nn.SmoothL1Loss()
        self.accuracy = get_distance_box_iou_accuracy

        # self.MSE = nn.MSELoss()
        # self.MAE = nn.L1Loss()
        # self.generalized_box_iou_loss = torchvision.ops.generalized_box_iou_loss
        # https://arxiv.org/abs/1911.08287
        # self.distance_box_iou_loss = torchvision.ops.distance_box_iou_loss
        # https://arxiv.org/abs/1902.09630
        # self.complete_box_iou_loss = torchvision.ops.complete_box_iou_loss

    def forward(self, x, box, text_emb):

        x_1 = self.clipScoreBlock(x[:,0,:,:], text_emb)

        # sort the boxes according to the similarity score
        _, idx = torch.sort(x_1, dim=1, descending=True)
        x = x.gather(3, idx.unsqueeze(-2).unsqueeze(-2).repeat(1, 2, 512, 1))[:,:,:,:3]
        x_1 = x_1.gather(1, idx)[:,:3].unsqueeze(-1)
        box = box.gather(1, idx.unsqueeze(-1).repeat(1, 1, 4))[:,:3]

        x = self.linWeighting(x)

        # shift the scores
        x = x_1 + x

        # box: (batch_size, 48, 1), (batch_size, 48, 4) -> (batch_size, 48, 5)
        x = torch.cat([x, box], dim=-1)

        # sort again the boxes according to the similarity score
        _, idx = torch.sort(x[:, :, 0], dim=1, descending=True)
        x = x.gather(1, idx.unsqueeze(-1).repeat(1, 1, 5))[:, 0, :]

        resize = self.resizer(x[:,0].unsqueeze(-1))

        return resize + x[:,1:]

    def training_step(self, batch, batch_idx):

        cat_emb_text, box, t_emb, target = batch

        pred = self(cat_emb_text, box, t_emb)

        huber_loss = self.HUBER(pred, target)

        accuracy, iou_mean, giou_mean = self.accuracy(pred, target)

        self.log('train_accuracy', accuracy, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log('train_huber_loss', huber_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log('train_iou_mean', iou_mean, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log('train_giou_mean', giou_mean, on_step = True, on_epoch = True, prog_bar = True, logger = True)

        return huber_loss * (1.4 - iou_mean)

    def validation_step(self, batch, batch_idx):
        cat_emb_text, box, t_emb, target = batch
        # print(cat_emb_text.shape, box.shape, y.shape)
        pred = self(cat_emb_text, box, t_emb)

        huber_loss = self.HUBER(pred, target)

        accuracy, iou_mean, giou_mean = self.accuracy(pred, target)

        self.log('train_accuracy', accuracy, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log('train_huber_loss', huber_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log('train_iou_mean', iou_mean, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log('train_giou_mean', giou_mean, on_step = True, on_epoch = True, prog_bar = True, logger = True)

        return huber_loss * (1.4 - iou_mean)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
        return [optimizer], [scheduler]