from data_trf import load_toy

import torch


def sample_batch(data, batch_size, seq_len, device=None):
    """
    Slice out `batch_size` contiguous windows of length `seq_len + 1`
    from the 1-D integer tensor `data`.
    Returns a tensor of shape (batch_size, seq_len + 1).
    """
    if data.dim() != 1:
        raise ValueError("data must be a 1-D tensor of token ids")

    total_len = data.size(0)
    window = seq_len + 1  # need one extra step for shifted targets
    if window > total_len:
        raise ValueError("Requested sequence length exceeds data length")

    max_start = total_len - window
    if max_start < 0:
        raise ValueError("Not enough data to sample the requested window size")

    # Sample random starting indice
    starts = torch.randint(0, max_start + 1, (batch_size,), device=data.device)
    offsets = torch.arange(window, device=data.device)
    indices = starts.unsqueeze(1) + offsets.unsqueeze(0)

    batch = data[indices]
    if device is not None:
        batch = batch.to(device)
    return batch


def batch_to_inputs_targets(batch):
    """
    Split a (batch, L+1) batch into language-model inputs/targets.
    """
    return batch[:, :-1], batch[:, 1:]


if __name__ == "__main__":
    (train, _), (i2c, c2i) = load_toy(final=False)
    train = train.to(torch.long)

    batch = sample_batch(train, batch_size=32, seq_len=64)
    x, y = batch_to_inputs_targets(batch)

    print("Batch shape:", batch.shape)
    print("Input shape:", x.shape, "Target shape:", y.shape)
