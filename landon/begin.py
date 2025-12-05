from data_trf import load_imdb, load_imdb_synth, load_xor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings("ignore")

# load the imdb dataset
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

# print the types of the data
print(type(x_train), type(y_train), type(i2w), type(w2i))

# print the 141st element of the x_train dataset
x_train[141]

# print the 141st element of the i2w and w2i dictionaries
print(i2w[141], w2i['film'])

# print a sentence from the x_train dataset
print([i2w[w] for w in x_train[141]])

# print the padding index
print(w2i['.pad'])