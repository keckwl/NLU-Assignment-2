import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import load_names, build_vocab, NamesDataset, collate_fn
from models  import VanillaRNN, BidirectionalLSTM, RNNWithAttention

HYPERPARAMS = {
    "embed_dim"  : 32,
    "hidden_size": 64,
    "num_layers" : 1,
    "dropout"    : 0.5,
    "lr"         : 0.001,
    "batch_size" : 32,
    "epochs"     : 30,
}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits, _ = model(x_batch)
        loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # prevent exploding gradients
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_model(model, model_name, loader, epochs, lr, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD tokens in loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss, loss_history = float("inf"), []

    print(f"Training {model_name} | params: {model.count_parameters():,}")
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_best.pt"))
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")
    print(f"  Best loss: {best_loss:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), loss_history)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Training Loss — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_loss.png"), dpi=120)
    plt.close("all")

    return loss_history, best_loss


def run_training(names_path="TrainingNames.txt", output_dir="outputs"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    names              = load_names(names_path)
    char2idx, idx2char = build_vocab(names)
    vocab_size         = len(char2idx)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump({"char2idx": char2idx, "idx2char": idx2char}, f, indent=2)

    loader = DataLoader(NamesDataset(names, char2idx), batch_size=HYPERPARAMS["batch_size"],
                        shuffle=True, collate_fn=collate_fn)

    model_configs = {
        "VanillaRNN"      : VanillaRNN,
        "BLSTM"           : BidirectionalLSTM,
        "RNNWithAttention": RNNWithAttention,
    }

    summary = {}
    for model_name, model_class in model_configs.items():
        model = model_class(
            vocab_size=vocab_size,
            embed_dim=HYPERPARAMS["embed_dim"],
            hidden_size=HYPERPARAMS["hidden_size"],
            num_layers=HYPERPARAMS["num_layers"],
            dropout=HYPERPARAMS["dropout"],
        ).to(device)

        _, best_loss = train_model(model, model_name, loader,
                                   HYPERPARAMS["epochs"], HYPERPARAMS["lr"],
                                   device, output_dir)
        summary[model_name] = {
            "best_loss"      : best_loss,
            "num_parameters" : model.count_parameters(),
            "hyperparameters": HYPERPARAMS,
        }
        # Free GPU memory before training the next model
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return char2idx, idx2char


if __name__ == "__main__":
    run_training()
