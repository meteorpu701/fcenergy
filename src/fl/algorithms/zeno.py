# src/fl/algorithms/zeno.py
import copy
from typing import Dict, List

import torch


class Zeno:
    """
    Zeno-style robust aggregation via update filtering.

    Expected per-client update dict:
      - "weights": state_dict  (optional if "delta" provided)
      - "delta": state_dict of local - global (preferred)
      - "n_samples": int
      - "train_loss": float   (proxy score term)

    Scoring (proxy):
      score_i = -(train_loss_i) - rho * ||delta_i||^2

    Keep top-k by score, then FedAvg over survivors (by n_samples).

    Notes:
      - This is a pragmatic proxy that doesn't require server-side data.
      - Non-floating tensors are kept from server (like FedNova/FedAvg patterns).
    """

    name = "zeno"

    def __init__(self, rho: float = 1e-3, keep_frac: float = 0.6, min_keep: int = 1):
        self.rho = float(rho)
        self.keep_frac = float(keep_frac)
        self.min_keep = int(min_keep)

    def aggregate(self, server_state: Dict, client_updates: List[Dict]) -> Dict:
        if not client_updates:
            return server_state

        global_w = server_state["weights"]

        # ---- build deltas + scores ----
        scored = []
        for i, u in enumerate(client_updates):
            if "n_samples" not in u:
                raise KeyError(f"client_updates[{i}] missing n_samples")
            if "train_loss" not in u:
                raise KeyError(f"client_updates[{i}] missing train_loss (Zeno proxy needs it)")

            # delta preferred, else compute from weights
            if "delta" in u and u["delta"] is not None:
                delta = u["delta"]
            else:
                if "weights" not in u:
                    raise KeyError(f"client_updates[{i}] needs 'delta' or 'weights'")
                local = u["weights"]
                delta = {param_name: (local[param_name] - global_w[param_name]) for param_name in global_w.keys()}

            # L2 norm over floating tensors only
            sq = 0.0
            for param_name, g in global_w.items():
                if not torch.is_tensor(g) or (not torch.is_floating_point(g)):
                    continue
                dk = delta[param_name]
                dk = dk.to(device=g.device, dtype=g.dtype) if torch.is_tensor(dk) else torch.tensor(dk, device=g.device, dtype=g.dtype)
                sq += float(torch.sum(dk * dk).detach().cpu().item())

            train_loss = float(u["train_loss"])
            score = -(train_loss) - self.rho * sq

            scored.append((score, u, delta))

        # ---- keep top-k by score ----
        scored.sort(key=lambda t: t[0], reverse=True)

        n = len(scored)
        k_keep = int(self.keep_frac * n)  # floor
        k_keep = max(self.min_keep, k_keep)
        k_keep = min(n, k_keep)

        survivors = scored[:k_keep]

        # ---- weighted average of deltas (FedAvg on survivors) ----
        sample_counts = [int(u["n_samples"]) for (_, u, _) in survivors]
        total = sum(sample_counts)
        if total <= 0:
            weights = [1.0 / len(sample_counts)] * len(sample_counts)
        else:
            weights = [n / total for n in sample_counts]

        new_w = copy.deepcopy(global_w)

        for kparam, w0 in global_w.items():
            if not torch.is_tensor(w0) or (not torch.is_floating_point(w0)):
                new_w[kparam] = w0
                continue

            acc = torch.zeros_like(w0)
            for j, (_, u, delta) in enumerate(survivors):
                dj = delta[kparam]
                dj = dj.to(device=w0.device, dtype=w0.dtype) if torch.is_tensor(dj) else torch.tensor(dj, device=w0.device, dtype=w0.dtype)
                acc = acc + weights[j] * dj

            new_w[kparam] = w0 + acc

        # optional debug stats saved in server_state (won't break anything)
        server_state = dict(server_state)
        server_state["zeno_kept"] = k_keep
        server_state["zeno_total"] = len(client_updates)
        server_state["zeno_scores_top"] = [float(s) for (s, _, _) in survivors[: min(5, len(survivors))]]
        return {
            "weights": new_w,
            **{k: v for k, v in server_state.items() if k.startswith("zeno_")}
        }