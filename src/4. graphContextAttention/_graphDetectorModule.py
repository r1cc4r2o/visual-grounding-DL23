import torch
import torch.nn as nn
import torch.nn.functional as F

from ._contextAttentionModule import ContextAttentionModule
from ._clipScoreBlock import CLIPscoreBlock

class GraphDetectorModule(nn.Module):
    def __init__(self, dim_embedding = 512, max_num_boxes = 3, hidden_resizing_operator=16) -> None:
        super().__init__()

        self.dim_embedding = dim_embedding
        self.max_num_boxes = max_num_boxes

        self.clipScoreBlock = CLIPscoreBlock()

        self.gam = ContextAttentionModule(dim_embedding, max_num_boxes)

        self.resizing_operator = nn.Sequential(
            nn.Linear(1, hidden_resizing_operator),
            nn.GELU(),
            nn.LayerNorm(hidden_resizing_operator),
            nn.Linear(hidden_resizing_operator, 4),
            nn.ReLU()
        )

    def forward(self, text_feat, x, boxes):


        # compute the matching score
        x_1 = self.clipScoreBlock(x[:,0,:,:], text_feat)

        # sort the boxes according to the matching score
        _, idx = torch.sort(x_1, dim=1, descending=True)
        x = x.gather(3, idx.unsqueeze(-2).unsqueeze(-2).repeat(1, 1, 512, 1))[:,:,:,:self.max_num_boxes]
        x_1 = x_1.gather(1, idx)[:,:self.max_num_boxes].unsqueeze(-1)
        boxes = boxes.gather(1, idx.unsqueeze(-1).repeat(1, 1, 4))[:,:self.max_num_boxes]


        # apply the graph attention modulation
        x = self.gam(x, x_1) # (B, D)

        # The score in output from the gam model represents
        # the matching score between the boxes considering
        # also the context. The context refers to the other
        # boxes.
        x = x + x_1

        # concatenate the boxes to the embedding
        x = torch.cat([x, boxes], dim = -1) # (batch_size, self.max_num_boxes, 1 + 4)

        # sort the boxes according with the score
        _, idx = torch.sort(x[:, :, 0], dim=1, descending=True)
        x = x.gather(1, idx.unsqueeze(-1).repeat(1, 1, 5))[:, 0, :]

        resize = self.resizing_operator(x[:,0].unsqueeze(-1))

        return resize + x[:,1:]