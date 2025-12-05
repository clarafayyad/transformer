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

class BaselineClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        """
        Initializes the BaselineClassifier model.
        :param vocab_size: Size of the vocabulary
        :param emb_dim: Dimension of the embedding
        :param num_classes: Number of classes
        """
        super().__init__()
        
        self.emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim
        )
        
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the BaselineClassifier model.
        x: (batch, time), dtype long
        :return: Logits for the classes
        """
        # 1. Embedding
        emb = self.emb(x)  # → (batch, time, emb)

        # 2. Global average pool over time dimension
        pooled = emb.mean(dim=1)  # → (batch, emb)

        # 3. Linear projection to classes
        logits = self.classifier(pooled)  # → (batch, num_classes)

        return logits

# model = BaselineClassifier(
#     vocab_size=len(w2i),
#     emb_dim=300,
#     num_classes=numcls
# )

# logits = model(x_train_t[:32])
# print(logits.shape)