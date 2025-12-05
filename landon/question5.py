from data_trf import load_imdb, load_imdb_synth, load_xor

from question1 import pad_and_convert, prepare_data
from question3 import train_one_epoch, evaluate, make_loaders

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings("ignore")

import optuna

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

x_train_t, y_train_t, x_val_t, y_val_t = prepare_data(x_train, y_train, x_val, y_val, w2i)

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        # print(f"Shape of x: {x.shape}")

        # 1. Compute attention scores: (batch, time, time)
        scores = torch.matmul(x, x.transpose(1, 2))

        # 4. Softmax over the time dimension
        attn = F.softmax(scores, dim=-1)
        print(f"Shape of attn: {attn.shape}")

        # 5. Weighted sum of values
        out = torch.matmul(attn, x)  # (batch, time, emb)
        return out

class BaselineWithAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.attn = SelfAttention()
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.emb(x)              # (batch, time, emb)
        x = self.attn(x)             # (batch, time, emb)
        x = x.max(dim=1).values      # global max pooling over time
        logits = self.classifier(x)  # (batch, num_classes)
        return logits

class AttentionModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_classes, pad_idx, pool_type="select"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.attn = SelfAttention()          # from Q4
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.pad_idx = pad_idx
        self.pool_type = pool_type

    def forward(self, x):
        mask = (x != self.pad_idx)  # (batch, time)
        x = self.emb(x)             # (batch, time, emb)
        x = self.attn(x, mask=mask) # (batch, time, emb)

        # change pooling type here
        if self.pool_type == "mean":
            pooled = x.mean(dim=1)
        elif self.pool_type == "max":
            pooled = x.max(dim=1).values
        elif self.pool_type == "select":
            pooled = x[:, 0, :]                    # (batch, emb)

        logits = self.classifier(pooled)          # (batch, num_classes)
        return logits

def run_experiment_q5(dataset_name, x_train, y_train, x_val, y_val, w2i, num_classes,
                   pool_type="mean", emb_dim=300, batch_size=256, epochs=5, lr=1e-3, 
                   patience=10, min_delta=0.0):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = len(w2i)
    pad_idx = w2i[".pad"]

    model = AttentionModel(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        num_classes=num_classes,
        pad_idx=pad_idx,
        pool_type=pool_type
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_train_t, y_train_t, x_val_t, y_val_t = prepare_data(x_train, y_train, x_val, y_val, w2i)
    train_loader, val_loader = make_loaders(x_train_t, y_train_t, x_val_t, y_val_t, batch_size)

    print(f"\n=== {dataset_name.upper()} — Pool: {pool_type} ===")
    
    # Early stopping variables
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Early stopping logic
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

# Tsrain on imdb
# MAX_LEN = 256

# (x_tr1, y_tr1), (x_va1, y_va1), (i2w1, w2i1), num_cls1 = load_imdb()
# x_tr1 = [seq[:MAX_LEN] for seq in x_tr1]
# x_va1   = [seq[:MAX_LEN] for seq in x_va1]

# run_experiment_q5(
#     "imdb_attn_select",
#     x_tr1, y_tr1,
#     x_va1, y_va1,
#     w2i1, num_cls1,
#     emb_dim=300,
#     batch_size=64,
#     epochs=5,
#     pool_type="select",
#     lr=1e-4
# )

# # Train on imdb synth (with optuna)
# (x_tr2, y_tr2), (x_va2, y_va2), (i2w2, w2i2), num_cls2 = load_imdb_synth()

# # Optuna hyperparameter tuning for learning rate and batch size
# def objective(trial):
#     # Suggest hyperparameters
#     lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)  # Log scale for learning rate
#     batch_size = trial.suggest_int("batch_size", 16, 256, step=16)  # Powers of 2-like values
    
#     # Fixed parameters (not tuning these)
#     emb_dim = 300
#     epochs = 100
#     pool_type = "select"
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     vocab_size = len(w2i2)
#     pad_idx = w2i2[".pad"]
    
#     # Create model
#     model = AttnSelectModel(
#         vocab_size=vocab_size,
#         emb_dim=emb_dim,
#         num_classes=num_cls2,
#         pad_idx=pad_idx
#     ).to(device)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
#     # Prepare data
#     x_train_t, y_train_t, x_val_t, y_val_t = prepare_data(x_tr2, y_tr2, x_va2, y_va2, w2i2)
#     train_loader, val_loader = make_loaders(x_train_t, y_train_t, x_val_t, y_val_t, batch_size)
    
#     # Train model with early stopping
#     best_val_acc = 0.0
#     patience = 30 # Stop if no improvement for 50 epochs
#     min_delta = 0.0  # Minimum change to qualify as improvement
#     patience_counter = 0
#     best_epoch = 0
    
#     for epoch in range(1, epochs + 1):
#         train_loss = train_one_epoch(model, train_loader, optimizer, device)
#         val_acc = evaluate(model, val_loader, device)
        
#         # Early stopping logic
#         if val_acc > best_val_acc + min_delta:
#             best_val_acc = val_acc
#             best_epoch = epoch
#             patience_counter = 0
#         else:
#             patience_counter += 1
        
#         # Report intermediate value for pruning
#         trial.report(val_acc, epoch)
        
#         # Handle pruning based on the intermediate value
#         if trial.should_prune():
#             raise optuna.TrialPruned()
        
#         # Early stopping
#         if patience_counter >= patience:
#             break
    
#     return best_val_acc

# # Create study and run optimization
# study = optuna.create_study(
#     direction="maximize",  # Maximize validation accuracy
#     study_name="lr_batch_size_tuning",
#     pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
# )

# print("Starting Optuna hyperparameter tuning...")
# print("Tuning: learning rate and batch size")
# print("Fixed: emb_dim=300, epochs=100, pool_type='select'\n")

# study.optimize(objective, n_trials=100)  # Run 20 trials

# # Print best results
# print("\n" + "="*50)
# print("Optimization finished!")
# print(f"Best trial:")
# trial = study.best_trial
# print(f"  Value (val_acc): {trial.value:.4f}")
# print(f"  Params:")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")
# print("="*50)

# # Optionally, run the best configuration
# print("\nTraining with best hyperparameters...")
# run_experiment_q5(
#     "imdb_synth_attn_select_best",
#     x_tr2, y_tr2,
#     x_va2, y_va2,
#     w2i2, num_cls2,
#     emb_dim=300,
#     batch_size=trial.params["batch_size"],
#     epochs=100,
#     lr=trial.params["lr"],
#     pool_type="select",
#     patience=30  # Early stopping patience
# )

# # Train on imdb synth (no optuna)
# (x_tr2, y_tr2), (x_va2, y_va2), (i2w2, w2i2), num_cls2 = load_imdb_synth()

# run_experiment_q5(
#     "imdb_synth_attn_select_best",
#     x_tr2, y_tr2,
#     x_va2, y_va2,
#     w2i2, num_cls2,
#     emb_dim=300,
#     batch_size=16,
#     epochs=100,
#     lr=1e-2,
#     pool_type="select",
#     patience=30  # Early stopping patience
# )

# # train the model on the xor dataset with select pooling
# (x_tr3, y_tr3), (x_va3, y_va3), (i2w3, w2i3), num_cls3 = load_xor()
# run_experiment_q5(
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