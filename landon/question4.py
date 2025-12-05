from data_trf import load_imdb, load_imdb_synth, load_xor

from question1 import pad_and_convert, prepare_data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings("ignore")

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

x_train_t, y_train_t, x_val_t, y_val_t = prepare_data(x_train, y_train, x_val, y_val, w2i)

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x = (batch, time, emb)
        # 1. Compute attention scores: (batch, time, time)
        scores = torch.matmul(x, x.transpose(1, 2))

        # 2. Softmax over the time dimension
        attn = F.softmax(scores, dim=-1)

        # 3. Weighted sum of values
        out = torch.matmul(attn, x)  # (batch, time, emb)
        return out

class BaselineWithAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.attn = SelfAttention()
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: (batch, time)
        x = self.emb(x)              # (batch, time, emb)
        x = self.attn(x)             # (batch, time, emb)
        x = x.max(dim=1).values      # global max pooling over time
        logits = self.classifier(x)  # (batch, num_classes)
        return logits