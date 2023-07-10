import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextAttentionModule(nn.Module):
    def __init__(self, dim_embedding, max_num_boxes = 3, temperature = 1.0, n_head=2, number_transformer_layers=1, hidden = 384) -> None:
        super().__init__()

        self.dim_embedding = dim_embedding
        self.num_boxes = max_num_boxes
        self.temperature = temperature
        self.hidden = hidden

        self.W = nn.Bilinear(max_num_boxes, 1, self.hidden)

        self.W1 = nn.Linear(self.hidden, 1)

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

        self.layernorm = nn.LayerNorm(self.hidden)

        # In our architecture the transformer encoder has the layer norm first.
        # According with the results of the paper 'On Layer Normalization in
        # the Transformer Architecture' the convergence is faster with the layer
        # norm first.
        # source: https://arxiv.org/pdf/2002.04745.pdf
        self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.num_boxes*self.num_boxes, nhead=n_head, norm_first=True, batch_first=True), num_layers=number_transformer_layers
            )


    def forward(self, x, matching):

        x = x.squeeze(-3)

        # the score over the last dim should sum to one
        c_s = F.softmax(torch.bmm(x.permute(0, 2, 1), x), dim=-1)

        c_s = c_s.flatten(start_dim=-2, end_dim=-1)

        c_s = self.transformer_encoder(c_s) # (B, D)

        c_s = c_s.reshape(c_s.shape[0], self.num_boxes, self.num_boxes)

        # combine the attention score
        # with the matching score
        x = self.W(c_s, matching) # (B, D)
        del c_s

        x = self.gelu(x)
        x = self.layernorm(x)

        x = self.W1(x) # (B, D)
        x = self.relu(x)

        return x