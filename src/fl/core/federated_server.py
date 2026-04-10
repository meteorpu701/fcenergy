# src/fl/core/federated_server.py
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from src.fl.algorithms import AGGREGATORS


class FederatedServer:
    def __init__(self, model: nn.Module, algorithm: str = "fedavg", agg_kwargs: dict | None = None):
        self.model = model
        self.algorithm = str(algorithm).strip().lower()

        if self.algorithm not in AGGREGATORS:
            raise ValueError(f"Unknown algorithm={self.algorithm}. Available={list(AGGREGATORS.keys())}")

        self.aggregator = AGGREGATORS[self.algorithm](**(agg_kwargs or {}))

        self.server_state = {"weights": model.state_dict()}

        if self.algorithm == "scaffold":
            self.server_state["c"] = {
                name: torch.zeros_like(p.data)
                for name, p in model.named_parameters()
            }

    def aggregate(self, client_updates):
        self.server_state = self.aggregator.aggregate(self.server_state, client_updates)
        self.model.load_state_dict(self.server_state["weights"])

    @torch.no_grad()
    def eval_rmse(self, loader: DataLoader, device: str = "cpu") -> float:
        self.model.eval()
        self.model.to(device)

        preds, ys = [], []
        for xb, yb in loader:
            xb = xb.to(device)
            pred = self.model(xb).cpu().numpy()
            preds.append(pred)
            ys.append(yb.numpy())

        preds = np.concatenate(preds)
        ys = np.concatenate(ys)
        return float(np.sqrt(np.mean((preds - ys) ** 2)))