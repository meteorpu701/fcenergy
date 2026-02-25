# src/federated_server.py
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

def fedavg(state_dicts, weights):
    avg = copy.deepcopy(state_dicts[0])
    for k in avg.keys():
        avg[k] = avg[k] * weights[0]
        for i in range(1, len(state_dicts)):
            avg[k] = avg[k] + state_dicts[i][k] * weights[i]
    return avg

@torch.no_grad()
def eval_rmse(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    model.to(device)
    preds, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        preds.append(pred)
        ys.append(yb.numpy())
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    return float(np.sqrt(np.mean((preds - ys) ** 2)))