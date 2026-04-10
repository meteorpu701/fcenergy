# src/fl/algorithms/scaffold.py

import copy
from typing import Dict, List
import torch


class Scaffold:
    name = "scaffold"

    def aggregate(self, server_state: Dict, client_updates: List[Dict]) -> Dict:

        if not client_updates:
            return server_state

        if "c" not in server_state:
            raise KeyError("server_state missing 'c' for Scaffold.")

        state_dicts = [u["weights"] for u in client_updates]
        sample_counts = [int(u.get("n_samples", 0)) for u in client_updates]
        total = sum(sample_counts)

        if total <= 0:
            weights = [1.0 / len(state_dicts)] * len(state_dicts)
        else:
            weights = [n / total for n in sample_counts]

        new_w = copy.deepcopy(state_dicts[0])

        for k in new_w.keys():
            new_w[k] = new_w[k] * weights[0]
            for i in range(1, len(state_dicts)):
                new_w[k] = new_w[k] + state_dicts[i][k] * weights[i]

        c_global = server_state["c"]
        new_c = {k: v.clone() for k, v in c_global.items()}

        for k in new_c.keys():
            acc = torch.zeros_like(new_c[k])
            m = 0
            for u in client_updates:
                if "c_delta" in u:
                    acc += u["c_delta"][k].to(acc.device, acc.dtype)
                    m += 1
            if m > 0:
                new_c[k] = new_c[k] + acc / float(m)

        return {
            "weights": new_w,
            "c": new_c,
        }