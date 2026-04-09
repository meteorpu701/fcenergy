# src/fl/algorithms/fedavg.py
import copy
from typing import Dict, List


class FedAvg:
    name = "fedavg"

    def aggregate(self, server_state: Dict, client_updates: List[Dict]) -> Dict:
        """
        server_state: {"weights": state_dict}
        client_updates: [{"weights": state_dict, "n_samples": int}, ...]
        """
        if not client_updates:
            return server_state

        for i, u in enumerate(client_updates):
            if "weights" not in u or "n_samples" not in u:
                raise KeyError(f"client_updates[{i}] must have keys: weights, n_samples")

        state_dicts = [u["weights"] for u in client_updates]
        sample_counts = [int(u["n_samples"]) for u in client_updates]

        total = sum(sample_counts)
        if total <= 0:
            # fallback: uniform average if sample counts are unusable
            weights = [1.0 / len(sample_counts)] * len(sample_counts)
        else:
            weights = [n / total for n in sample_counts]

        avg = copy.deepcopy(state_dicts[0])
        for k in avg.keys():
            avg[k] = avg[k] * weights[0]
            for i in range(1, len(state_dicts)):
                avg[k] = avg[k] + state_dicts[i][k] * weights[i]

        return {"weights": avg}