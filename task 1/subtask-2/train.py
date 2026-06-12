import torch
import torch.nn as nn
import os
import csv

def train_epoch(model, loader, optimizer, criterion, clip, device):
    model.train()
    total_loss = 0
    
    for buggy, fixed in loader:
        buggy, fixed = buggy.to(device), fixed.to(device)

        optimizer.zero_grad()
        predictions = model(buggy, fixed)

        loss = criterion(predictions.reshape(-1, predictions.shape[-1]), fixed[:, 1:].reshape(-1)) # cross entropy expects (batch, vocab_size, seq_len)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip) # parameters are scaled down till norm <= clip
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0
    
    for buggy, fixed in loader:
        buggy, fixed = buggy.to(device), fixed.to(device)

        predictions = model(buggy, fixed, teacher_forcing_ratio=0.0)

        loss = criterion(predictions.reshape(-1, predictions.shape[-1]), fixed[:, 1:].reshape(-1))

        total_loss += loss.item()

    return total_loss / len(loader)


def train(model, train_loader, val_loader, optimizer, criterion, clip, epochs, device, experiment_name):
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, clip, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)

        history.append({
            "experiment": experiment_name,
            "epoch": epoch,
            "train_loss": round(train_loss, 4), # .4f is only for displaying
            "val_loss": round(val_loss, 4),
        })

        print(f"epoch {epoch}/{epochs} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"checkpoints/{experiment_name}_best.pt") # state_dict is dict that maps each layer to its parameter tensor
            print(f"  saved checkpoint (val_loss={best_val_loss:.4f})")

    write_header = not os.path.exists("results/metrics.csv") # if file exists, no need of header (LLM-inspired)
    with open("results/metrics.csv", "a", newline="") as f: # "a" appends
        writer = csv.DictWriter(f, fieldnames=["experiment", "epoch", "train_loss", "val_loss"])
        if write_header:
            writer.writeheader()
        writer.writerows(history)

    return history