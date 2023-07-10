import torch
import torch.nn as nn



class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_emb, nhead: int = 2, dim_feedforward: int = 10, dropout: float =0.1, activation = nn.GELU):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attn = nn.MultiheadAttention(dim_emb, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim_emb, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_emb)

        self.norm1 = nn.LayerNorm(dim_emb)
        self.norm2 = nn.LayerNorm(dim_emb)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()

    def forward(self, data, src_mask=None, src_key_padding_mask=None):
        
        # MultiHeadAttention
        x, attn = self.attn(data, data, data, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        
        # add & norm
        x = data + self.dropout1(x)
        x = self.norm1(x)
        
        # Implementation of Feedforward model
        x1 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        # add & norm
        x = x + self.dropout2(x1)
        x = self.norm2(x)

        return x
    


class MLP(nn.Module):
    def __init__(self, dim_emb, dim_feedforward = 10, dropout=0.1, activation = nn.GELU):
        super(MLP, self).__init__()
        
        self.linear1 = nn.Linear(dim_emb, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_emb)

        self.dropout = nn.Dropout(dropout)

        self.activation = activation()

    def forward(self, data):
        
        x = self.linear2(self.dropout(self.activation(self.linear1(data))))
        
        return x

    

class Net(nn.Module):
    def __init__(self, dim_emb, dim_box: int = 4, nhead: int = 2, dim_feedforward: int = 10, dropout: float =0.1, activation = nn.GELU):
        super(Net, self).__init__()
        
        self.encoder_layer = TransformerEncoderLayer(dim_emb, nhead, dim_feedforward, dropout, activation)
        self.mlp = MLP(dim_emb, dim_feedforward, dropout, activation)
        self.linear = nn.Linear(dim_emb, dim_box)
        
    def forward(self, data):
        
        x = self.encoder_layer(data)
        x = self.mlp(x)
        x = self.linear(x)
        
        return x
    


def training(model, train_loader, optimizer, criterion = nn.CrossEntropyLoss, device = 'cuda', epochs = 10):

    sample = 0.0
    cum_loss = 0.0

    model.train()

    for e in range(epochs):

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            optimizer.zero_grad()

            sample += len(data)
            cum_loss += loss.item()

        print(f'Train Epoch: {e} Loss: {cum_loss/sample}')    


def test_fn(model, test_loader, criterion = nn.CrossEntropyLoss, device = 'cuda'):

    sample = 0.0
    cum_loss = 0.0

    model.eval()

    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):

            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            sample += len(data)
            cum_loss += loss.item()

        print(f'Test Loss: {cum_loss/sample}')       