from data_trf import load_toy

from question10 import sample_batch, batch_to_inputs_targets

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=6, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must divide num_heads evenly")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.to_queries = nn.Linear(embed_dim, embed_dim)
        self.to_keys = nn.Linear(embed_dim, embed_dim)
        self.to_values = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, tensor, batch_size, time):
        tensor = tensor.view(batch_size, time, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3).contiguous()  # -> (batch, heads, time, head_dim)

    def forward(self, x):
        batch_size, time, _ = x.size()

        # Independent q/k/v projections per head.
        q = self._reshape_heads(self.to_queries(x), batch_size, time)
        k = self._reshape_heads(self.to_keys(x), batch_size, time)
        v = self._reshape_heads(self.to_values(x), batch_size, time)

        scaling = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling  # (batch, heads, time, time)

        # Build a causal mask that sets positions above the diagonal to -inf.
        i, j = torch.triu_indices(time, time, offset=1, device=scores.device)
        scores[..., i, j] = float("-inf")

        attn = torch.softmax(scores, dim=-1)  # attention probabilities
        attn = self.dropout(attn)  # dropout on attention weights

        context = torch.matmul(attn, v)  # (batch, heads, time, head_dim)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, time, self.embed_dim)  # merge heads

        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        hidden_dim = embed_dim * 4
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=6, dropout=0.1):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)

        self.ff = FeedForward(embed_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.drop1(self.attn(x))
        x = self.norm1(x + attn_out)

        ff_out = self.drop2(self.ff(x))
        x = self.norm2(x + ff_out)
        return x


class AutoregressiveTransformerLM(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_len, num_heads=6, num_blocks=3, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device

        max_positions = self.pos_emb.num_embeddings
        positions = torch.arange(seq_len, device=device) % max_positions
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        x = self.emb(x) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)

        x = self.dropout(x)
        logits = self.decoder(x)  # (batch, time, vocab)
        return logits


def run_language_model(train_data, vocab_size, seq_len=128, steps=200,
                       batch_size=64, emb_dim=256, num_heads=4, num_blocks=3,
                       dropout=0.1, lr=3e-4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoregressiveTransformerLM(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        max_len=seq_len,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(1, steps + 1):
        batch = sample_batch(train_data, batch_size=batch_size, seq_len=seq_len, device=device)
        x, y = batch_to_inputs_targets(batch)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"Step {step:04d} | loss={loss.item():.4f}")

    return model


if __name__ == "__main__":
    (train, _), (i2c, c2i) = load_toy(final=False)
    train = train.to(torch.long)

    vocab_size = len(c2i)

    run_language_model(
        train_data=train,
        vocab_size=vocab_size,
        seq_len=128,
        steps=200,
        batch_size=64,
        emb_dim=256,
        num_heads=4,
        num_blocks=3,
        dropout=0.1,
        lr=3e-4,
    )
