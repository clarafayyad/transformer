from data_trf import load_imdb, load_imdb_synth, load_xor

from question1 import pad_and_convert, prepare_data
from question3 import train_one_epoch, evaluate, make_loaders

import math
import torch
import torch.nn as nn

import warnings

warnings.filterwarnings("ignore")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=6):
        """
        Multi-head self-attention with learnable projections for q/k/v.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.to_queries = nn.Linear(embed_dim, embed_dim)
        self.to_keys = nn.Linear(embed_dim, embed_dim)
        self.to_values = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (batch, time, emb)
        """
        batch_size, time, _ = x.size()

        q = self.to_queries(x)
        k = self.to_keys(x)
        v = self.to_values(x)

        # reshape to (batch, num_heads, time, head_dim)
        def reshape_heads(tensor):
            tensor = tensor.view(batch_size, time, self.num_heads, self.head_dim)
            return tensor.permute(0, 2, 1, 3).contiguous()

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # scaled dot-product attention
        scaling = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling  # (batch, heads, time, time)

        attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, v)  # (batch, heads, time, head_dim)

        # reorder back to (batch, time, emb)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, time, self.embed_dim)

        return self.out_proj(context)


class MultiHeadAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, pad_idx,
                 max_len, pool_type="select", num_heads=6):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        # Learned positional embeddings so the model knows token order.
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.attn = MultiHeadSelfAttention(emb_dim, num_heads=num_heads)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.pad_idx = pad_idx
        self.pool_type = pool_type

    def forward(self, x):
        batch_size, seq_len = x.shape

        max_positions = self.pos_emb.num_embeddings
        # Create a [0..seq_len) position index row and wrap if seq exceeds table.
        positions = torch.arange(seq_len, device=x.device) % max_positions
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        # Token embedding + positional embedding -> order-aware representation.
        x = self.emb(x) + self.pos_emb(positions)
        x = self.attn(x)      # (batch, time, emb)

        if self.pool_type == "mean":
            pooled = x.mean(dim=1)
        elif self.pool_type == "max":
            pooled = x.max(dim=1).values
        else:  # "select"
            pooled = x[:, 0, :]

        logits = self.classifier(pooled)
        return logits


def run_experiment_q6(dataset_name, x_train, y_train, x_val, y_val, w2i, num_classes,
                      pool_type="select", emb_dim=300, num_heads=6, max_len=256,
                      batch_size=256, epochs=5, lr=1e-3, patience=10, min_delta=0.0):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_size = len(w2i)
    pad_idx = w2i[".pad"]

    model = MultiHeadAttentionClassifier(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        num_classes=num_classes,
        pad_idx=pad_idx,
        max_len=max_len,
        pool_type=pool_type,
        num_heads=num_heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_train_t, y_train_t, x_val_t, y_val_t = prepare_data(x_train, y_train, x_val, y_val, w2i)
    train_loader, val_loader = make_loaders(x_train_t, y_train_t, x_val_t, y_val_t, batch_size)

    print(f"\n=== {dataset_name.upper()} — Pool: {pool_type}, Heads: {num_heads} ===")

    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
                break

    return model


if __name__ == "__main__":
    # MAX_LEN = 256

    # (x_tr, y_tr), (x_va, y_va), (i2w, w2i), num_cls = load_imdb()
    # x_tr = [seq[:MAX_LEN] for seq in x_tr]
    # x_va = [seq[:MAX_LEN] for seq in x_va]

    # run_experiment_q6(
    #     "imdb_multihead",
    #     x_tr, y_tr,
    #     x_va, y_va,
    #     w2i, num_cls,
    #     emb_dim=300,
    #     num_heads=6,
    #     max_len=MAX_LEN,
    #     batch_size=256,
    #     epochs=5,
    #     pool_type="select",
    #     lr=1e-2,
    # )

    # # (x_tr2, y_tr2), (x_va2, y_va2), (i2w2, w2i2), num_cls2 = load_imdb_synth()

    # # run_experiment_q6(
    # #     "imdb_synth_attn_select_best",
    # #     x_tr2, y_tr2,
    # #     x_va2, y_va2,
    # #     w2i2, num_cls2,
    # #     emb_dim=300,
    # #     batch_size=64,
    # #     epochs=100,
    # #     lr=1e-2,
    # #     pool_type="select",
    # #     patience=30  # Early stopping patience
    # # )

    # (x_tr3, y_tr3), (x_va3, y_va3), (i2w3, w2i3), num_cls3 = load_xor()
    # run_experiment_q6(
    #         "xor",
    #         x_tr3, y_tr3,
    #         x_va3, y_va3,
    #         w2i3, num_cls3,
    #         emb_dim=300,
    #         batch_size=16,
    #         epochs=100,
    #         lr=1e-2,
    #         pool_type="select",
    #         patience=30,
    # )