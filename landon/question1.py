from data_trf import load_imdb, load_imdb_synth, load_xor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings("ignore")

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

def pad_and_convert(seqs, pad_idx):
    """
    Pads and converts a list of sequences to a tensor.
    :param seqs: List of sequences
    :param pad_idx: Index of the padding token
    :return: Tensor of padded sequences
    """
    # 1. Find max sequence length
    max_len = max(len(s) for s in seqs)

    padded = []
    for s in seqs:
        # 2. Pad with pad_idx to max_len
        padded_seq = s + [pad_idx] * (max_len - len(s))
        padded.append(padded_seq)

    # 3. Convert to tensor (long dtype)
    return torch.tensor(padded, dtype=torch.long)


def prepare_data(x_train, y_train, x_val, y_val, w2i):
    """
    Prepares the data for training and validation.
    :param x_train: List of training sequences
    :param y_train: List of training labels
    :param x_val: List of validation sequences
    :param y_val: List of validation labels
    :param w2i: Dictionary of word to index mappings
    :return: Tuple of tensors for training and validation data
    """
    pad_idx = w2i[".pad"]

    x_train_t = pad_and_convert(x_train, pad_idx)
    x_val_t   = pad_and_convert(x_val, pad_idx)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)

    return x_train_t, y_train_t, x_val_t, y_val_t

# x_train_t, y_train_t, x_val_t, y_val_t = prepare_data(x_train, y_train, x_val, y_val, w2i)

# print(x_train_t.shape)
# print(y_train_t.shape)

# # print padded sequence at index 2
# print(x_train_t[2])

# # print the label at index 2
# print(y_train_t[2])