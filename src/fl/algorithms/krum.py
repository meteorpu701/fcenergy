# src/fl/algorithms/krum.py
import copy
from typing import Dict, List, Tuple

import torch


class Krum:

    name = "krum"

    def __init__(self, f: int = 0):
        self.f = int(f)

    @staticmethod
    def _l2_sq(delta_a: Dict, delta_b: Dict, template: Dict) -> float:
        s = 0.0
        for k, g in template.items():
            if not torch.is_tensor(g) or (not torch.is_floating_point(g)):
                continue
            da = delta_a[k].to(device=g.device, dtype=g.dtype)
            db = delta_b[k].to(device=g.device, dtype=g.dtype)
            d = da - db
            s += float(torch.sum(d * d).detach().cpu().item())
        return s

    def aggregate(self, server_state: Dict, client_updates: List[Dict]) -> Dict:
        if not client_updates:
            return server_state

        global_w = server_state["weights"]
        n = len(client_updates)
        f = self.f

        if n < 2 * f + 3:
            raise ValueError(f"Krum requires n >= 2f + 3. Got n={n}, f={f}.")

        deltas: List[Dict[str, torch.Tensor]] = []
        for i, u in enumerate(client_updates):
            if "delta" in u and u["delta"] is not None:
                delta = u["delta"]
            else:
                if "weights" not in u:
                    raise KeyError(f"client_updates[{i}] needs 'delta' or 'weights'")
                local = u["weights"]
                delta = {}
                for k, g in global_w.items():
                    if not torch.is_tensor(g) or (not torch.is_floating_point(g)):
                        continue
                    delta[k] = (local[k].to(device=g.device, dtype=g.dtype) - g).detach()
            for k, g in global_w.items():
                if torch.is_tensor(g) and torch.is_floating_point(g) and k not in delta:
                    delta[k] = torch.zeros_like(g)
            deltas.append(delta)

        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = self._l2_sq(deltas[i], deltas[j], global_w)
                dist[i][j] = d
                dist[j][i] = d

        m = n - f - 2
        scores: List[Tuple[float, int]] = []
        for i in range(n):
            dists = sorted(dist[i][j] for j in range(n) if j != i)
            score = sum(dists[:m])
            scores.append((score, i))

        scores.sort(key=lambda t: t[0])
        best_i = scores[0][1]

        new_w = copy.deepcopy(global_w)
        for k, g in global_w.items():
            if not torch.is_tensor(g) or (not torch.is_floating_point(g)):
                new_w[k] = g
                continue
            new_w[k] = g + deltas[best_i][k].to(device=g.device, dtype=g.dtype)

        out = dict(server_state)
        out["krum_n"] = n
        out["krum_f"] = f
        out["krum_m"] = m
        out["krum_chosen"] = int(best_i)
        out["krum_best_score"] = float(scores[0][0])

        return {"weights": new_w, **{k: v for k, v in out.items() if k.startswith("krum_")}}