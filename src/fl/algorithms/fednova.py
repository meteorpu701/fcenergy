# src/fl/algorithms/fednova.py
import copy
from typing import Dict, List

import torch


class FedNova:
    """
    FedNova (Normalized Averaging) aggregation (practical delta-based variant).

    Client update expected:
      - "delta": state_dict of (local_weights - global_weights)
      - "n_steps": int  (number of optimizer steps taken locally)
      - "n_samples": int

    We compute:
      p_i = n_samples_i / sum_j n_samples_j
      a_i = max(n_steps_i, 1)
      tau_eff = sum_i p_i * a_i

    Then:
      w_{t+1} = w_t + tau_eff * sum_i p_i * (delta_i / a_i)

    Key property:
      - If all clients have the same a_i = a, then tau_eff = a,
        and the update reduces to: w_{t+1} = w_t + sum_i p_i * delta_i  (FedAvg on deltas).
    """

    name = "fednova"

    def aggregate(self, server_state: Dict, client_updates: List[Dict]) -> Dict:
        if not client_updates:
            return server_state

        global_w = server_state["weights"]

        # Validate required keys
        for i, u in enumerate(client_updates):
            if "n_samples" not in u or "n_steps" not in u:
                raise KeyError(f"client_updates[{i}] missing n_samples or n_steps (FedNova needs both)")
            if "delta" not in u and "weights" not in u:
                raise KeyError(f"client_updates[{i}] must contain 'delta' or 'weights'")

        # Sample weights p_i
        sample_counts = [int(u["n_samples"]) for u in client_updates]
        total = sum(sample_counts)
        if total <= 0:
            p = [1.0 / len(sample_counts)] * len(sample_counts)
        else:
            p = [n / total for n in sample_counts]

        # Step counts a_i
        a = [max(int(u["n_steps"]), 1) for u in client_updates]

        # Effective steps
        tau_eff = sum(p_i * a_i for p_i, a_i in zip(p, a))
        if tau_eff <= 0:
            tau_eff = 1.0

        # Helper: get delta dict for a client
        def _get_delta(u: Dict) -> Dict[str, torch.Tensor]:
            if "delta" in u and u["delta"] is not None:
                return u["delta"]
            # fallback: delta = local - global (if only weights provided)
            local = u["weights"]
            d = {}
            for k in global_w.keys():
                d[k] = local[k] - global_w[k]
            return d

        # Aggregate
        new_w = copy.deepcopy(global_w)

        for k, w0 in global_w.items():
            # Keep non-float tensors unchanged (e.g. BatchNorm counters)
            if not torch.is_tensor(w0) or (not torch.is_floating_point(w0)):
                new_w[k] = w0
                continue

            acc = torch.zeros_like(w0)

            for i, u in enumerate(client_updates):
                delta_i = _get_delta(u)[k]

                if not torch.is_tensor(delta_i):
                    delta_i = torch.tensor(delta_i, device=w0.device, dtype=w0.dtype)
                else:
                    delta_i = delta_i.to(device=w0.device, dtype=w0.dtype)

                acc = acc + (p[i] * (delta_i / float(a[i])))

            # scale back by tau_eff
            new_w[k] = w0 + (tau_eff * acc)

        return {"weights": new_w}