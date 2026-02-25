# src/federated_client.py
import torch
from torch import nn
from torch.utils.data import DataLoader

def train_local(model: nn.Module, loader: DataLoader, lr: float, epochs: int, device: str = "cpu") -> float:
    model.train()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    last_loss = 0.0

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())

    return last_loss

def train_local_fedprox(
    model, global_model, loader, lr, epochs, mu=0.01, device="cpu"
):
    model.train()
    model.to(device)
    global_model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            prox = 0.0
            for w, w0 in zip(model.parameters(), global_model.parameters()):
                prox += ((w - w0) ** 2).sum()

            loss = loss + (mu / 2.0) * prox
            loss.backward()
            opt.step()