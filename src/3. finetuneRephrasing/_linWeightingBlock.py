import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBoxWeightingBlock(nn.Module):
    def __init__(self, n_head=2, number_transformer_layers=2, dropout=0.1, emb_size=128, max_num_boxes=3, dim_emb_clip=512):
        super().__init__()

        self.linear_1 = nn.Linear(2, emb_size)

        self.layernorm_1 = nn.LayerNorm([max_num_boxes, dim_emb_clip, emb_size])
        self.gelu = nn.GELU()

        self.linear_2 = nn.Linear(emb_size*dim_emb_clip, emb_size)

        # In our architecture the transformer encoder uses layer norm first.
        # According to the results of the paper 'On Layer Normalization in
        # the Transformer Architecture' the convergence is faster if layer
        # norm is done first.
        # source: https://arxiv.org/pdf/2002.04745.pdf
        self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_head, norm_first=True, batch_first=True, activation=F.gelu), num_layers=number_transformer_layers
            )

        # This layer combines the output of the transformer encoder
        # with the initial concatenated embedding. The result of this layer is
        # a single vector for each box.
        self.linear_3 = nn.Linear(emb_size, emb_size)
        self.layernorm_2 = nn.LayerNorm([max_num_boxes, emb_size])

        self.linear_4 = nn.Linear(emb_size, 1)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = x.permute(0, 3, 2, 1)                   # permute
        x = self.linear_1(x)                        # linear transformation
        x = self.gelu(x)                            # non-linearity
        x = self.layernorm_1(x)                     # norm
        x = x.flatten(start_dim=2, end_dim=-1)      # flatten
        x = self.linear_2(x)                        # linear transformation
        x = self.gelu(x)                            # non-linearity
        x = self.transformer_encoder(x)             # transformer encoder

        if self.training:
          x = self.linear_3(self.dropout(x))        # linear transformation
        else:
          x = self.linear_3(x)                      # linear transformation
        x = self.gelu(x)                            # non-linearity
        x = self.layernorm_2(x)                     # norm
        x = self.linear_4(x)                        # linear transformation

        return x