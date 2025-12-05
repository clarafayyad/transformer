# from ssl import _PasswordType
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

# Model with 3 global pooling options
class BaselinePool(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, pool_type="mean"):
        """
        Initializes the BaselinePool model.
        :param vocab_size: Size of the vocabulary
        :param emb_dim: Dimension of the embedding
        :param num_classes: Number of classes
        :param pool_type: Type of pooling to use
        """
        super().__init__()
        self.pool_type = pool_type 
        self.emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        emb = self.emb(x) 

        if self.pool_type == "mean":
            pooled = emb.mean(dim=1)
        elif self.pool_type == "max":
            pooled = emb.max(dim=1).values
        elif self.pool_type == "select":
            pooled = emb[:, 0, :]

        logits = self.classifier(pooled)
        return logits

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for xb, yb in loader: # iterate through batches
        xb, yb = xb.to(device), yb.to(device) # move batch to either CPU or GPU 

        optimizer.zero_grad() # clear gradients

        logits = model(xb) # feed the input batch through the model
        loss = F.cross_entropy(logits, yb) # compute loss
        loss.backward() # compute gradients
        optimizer.step() # update model parameters using computed gradients 
        optimizer.zero_grad() # clear gradients

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    total_acc = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_acc += accuracy(logits, yb) * xb.size(0)

    return total_acc / len(loader.dataset)

def make_loaders(x_train, y_train, x_val, y_val, batch_size):
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def run_experiment(dataset_name, x_train, y_train, x_val, y_val, w2i, num_classes,
                   pool_type="mean", emb_dim=300, batch_size=256, epochs=5, lr=1e-3):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = len(w2i)

    model = BaselinePool(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        num_classes=num_classes,
        pool_type=pool_type
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_train_t, y_train_t, x_val_t, y_val_t = prepare_data(x_train, y_train, x_val, y_val, w2i)
    train_loader, val_loader = make_loaders(x_train_t, y_train_t, x_val_t, y_val_t, batch_size)

    print(f"\n=== {dataset_name.upper()} — Pool: {pool_type} ===")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    return model


if __name__ == "__main__":
    # train the model on the imdb dataset with 3 global pooling options
    # (x_tr1, y_tr1), (x_va1, y_va1), (i2w1, w2i1), num_cls1 = load_imdb()

    # run the experiment with 3 global pooling options
    # for pool in ["mean", "max", "select"]:
    #     run_experiment(
    #         "imdb",
    #         x_tr1, y_tr1,
    #         x_va1, y_va1,
    #         w2i1, num_cls1,
    #         pool_type=pool
    #     )

    # # adjust the learning rate and batch size
    # run_experiment(
    #         "imdb",
    #         x_tr1, y_tr1,
    #         x_va1, y_va1,
    #         w2i1, num_cls1,
    #         pool_type="max",
    #         batch_size=256,
    #         lr=1e-2,
    # )

    # train the model on the imdb synth dataset with 3 global pooling options
    # (x_tr2, y_tr2), (x_va2, y_va2), (i2w2, w2i2), num_cls2 = load_imdb_synth()

    # # Inspecting data (because it threw an index out of bound error, I had to remove duplicate strings from i2w in load_imdb_synth())
    # print(len(i2w2))
    # print(len(w2i2))

    # max_token = max(max(seq) for seq in x_tr2)
    # vocab_size = len(w2i2)
    # print("max token:", max_token)
    # print("vocab size:", vocab_size)
    # assert max_token < vocab_size, "ERROR: token index exceeds vocabulary size!"

    # invalid_tokens = set(
    #     idx
    #     for seq in x_tr2
    #     for idx in seq
    #     if idx >= vocab_size
    # )
    # print("Invalid token IDs:", invalid_tokens)
    # for bad in invalid_tokens:
    #     print(bad, "→", i2w2[bad] if bad < len(i2w2) else "(not in i2w2)")


    # for pool in ["mean", "max", "select"]:
    #     run_experiment("imdb_synth",
    #                 x_tr2, y_tr2,
    #                 x_va2, y_va2,
    #                 w2i2, num_cls2,
    #                 pool_type=pool,
    #                 batch_size=256,
    #                 epochs=5,
    #                 lr=1e-2)

    # # train the model on the xor dataset with max pooling
    # (x_tr3, y_tr3), (x_va3, y_va3), (i2w3, w2i3), num_cls3 = load_xor()
    # for pool in ["mean", "max", "select"]:
    #     run_experiment(
    #             "xor",
    #             x_tr3, y_tr3,
    #             x_va3, y_va3,
    #             w2i3, num_cls3,
    #             pool_type=pool,
    #             batch_size=256,
    #             lr=1e-2,
    #     )
    pass