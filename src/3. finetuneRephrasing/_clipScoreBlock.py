import torch
import torch.nn as nn


class CLIPscoreBlock(nn.Module):
    '''
    This block computes the batched clip score for each box.
    The input is the text embedding and the box embedding.
    The output is a tensor of shape (batch_size, n_boxes, 1)
    where each element is the score of the box.

    Args:
        @params text_emb: tensor of shape (batch_size, 512)
        @params box_emb: tensor of shape (batch_size, n_boxes, 512)

    Returns:
        @params returns x: tensor of shape (batch_size, n_boxes, 1)

    x:  [tex_0 @ box_1.T, tex_0 @ box_2.T, ..., tex_0 @ box_j]
        [tex_1 @ box_1.T, tex_1 @ box_2.T, ..., tex_1 @ box_j]
        [tex_2 @ box_1.T, tex_2 @ box_2.T, ..., tex_2 @ box_j]
                                ...
                                ...
        [tex_i @ box_1.T, tex_i @ box_2.T, ..., tex_i @ box_j]

    Normalizing over the rows and summing over the columns
    to get the overall score for each box.

    x summarize the matching in between the sentence and the boxes
    x: [sum(tex_[:] @ box_1.T), sum(tex_[:] @ box_2.T), ..., sum(tex_[:] @ box_j)]
    '''
    def __init__(self, n_box=48):
        super().__init__()
        self.n_box = n_box


    def forward(self, box_encoding, text_emb):

        x = torch.bmm(text_emb.squeeze(-1), box_encoding) # (B, D) @ (B, D)
        x = x.sum(dim=-2)

        return x