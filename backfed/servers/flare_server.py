"""Implementation of FLARE server for federated learning."""

import math
from typing import Dict, List, Tuple

import torch
from logging import INFO, WARNING

from backfed.servers.defense_categories import RobustAggregationServer
from backfed.utils.logging_utils import log


class FlareServer(RobustAggregationServer):
    """
    FLARE server implementation that uses Maximum Mean Discrepancy (MMD)
    to detect and filter malicious updates.

    This is a hybrid defense that combines anomaly detection (MMD-based detection)
    with robust aggregation (weighted aggregation based on trust scores).
    """

    def __init__(
        self,
        server_config,
        server_type: str = "flare",
        voting_threshold: float = 0.5,
        temperature: float = 1.0,
        eta: float = 0.1,
    ):
        super().__init__(server_config, server_type, eta)
        self.voting_threshold = voting_threshold
        self.temperature = max(float(temperature), 1e-6)
        log(
            INFO,
            "Initialized FLARE server with voting_threshold=%s, temperature=%s",
            voting_threshold,
            self.temperature,
        )

    def _kernel_function(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix between two sets of vectors."""
        sigma = 1.0
        return torch.exp(-torch.cdist(x, y, p=2).pow(2) / (2 * sigma**2))

    def _compute_mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy between two sets of features."""
        m, n = x.size(0), y.size(0)

        if m == 0 or n == 0:
            return torch.tensor(0.0, device=x.device if m else y.device)

        xx_kernel = self._kernel_function(x, x)
        yy_kernel = self._kernel_function(y, y)
        xy_kernel = self._kernel_function(x, y)

        if m > 1:
            xx_sum = (xx_kernel.sum() - torch.diagonal(xx_kernel).sum()) / (m * (m - 1))
        else:
            xx_sum = torch.tensor(0.0, device=xx_kernel.device)

        if n > 1:
            yy_sum = (yy_kernel.sum() - torch.diagonal(yy_kernel).sum()) / (n * (n - 1))
        else:
            yy_sum = torch.tensor(0.0, device=yy_kernel.device)

        xy_sum = xy_kernel.sum() / (m * n)

        return xx_sum + yy_sum - 2 * xy_sum

    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, Dict]],
                               client_features: List[torch.Tensor]) -> bool:
        """
        Aggregate client updates using FLARE mechanism.

        Args:
            client_updates: List of (client_id, num_examples, model_update)
            client_features: List of feature representations from clients
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False

        if not client_features:
            log(INFO, "FLARE: No client features available, using standard FedAvg")
            return super().aggregate_client_updates(client_updates)

        num_clients = len(client_updates)

        if len(client_features) != num_clients:
            log(
                WARNING,
                "FLARE: Mismatch between client updates (%d) and features (%d), falling back to FedAvg",
                num_clients,
                len(client_features),
            )
            return super().aggregate_client_updates(client_updates)

        distance_matrix = torch.zeros((num_clients, num_clients), dtype=torch.float32)
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                mmd_score = self._compute_mmd(client_features[i], client_features[j]).item()
                distance_matrix[i, j] = distance_matrix[j, i] = mmd_score

        log(INFO, "FLARE distances: %s", distance_matrix.tolist())

        neighbor_count = max(1, int(math.ceil(self.voting_threshold * (num_clients - 1))))
        vote_counter = torch.zeros(num_clients, dtype=torch.float32)

        for i in range(num_clients):
            distances = distance_matrix[i]
            sorted_indices = torch.argsort(distances)
            neighbor_indices = [idx.item() for idx in sorted_indices if idx != i][:neighbor_count]
            for neighbor in neighbor_indices:
                vote_counter[neighbor] += 1

        trust_scores = torch.softmax(vote_counter / self.temperature, dim=0)
        log(INFO, "FLARE trust scores: %s", trust_scores.tolist())

        global_state_dict = self.global_model.state_dict()
        weight_accumulator: Dict[str, torch.Tensor] = {}

        for name, param in global_state_dict.items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            weight_accumulator[name] = torch.zeros_like(
                param, device=self.device, dtype=torch.float32
            )

        for weight, (_, _, client_state) in zip(trust_scores.tolist(), client_updates):
            trust_weight = float(weight)
            for name, param in client_state.items():
                if any(pattern in name for pattern in self.ignore_weights):
                    continue
                client_param = param.to(device=self.device, dtype=torch.float32)
                global_param = global_state_dict[name].to(device=self.device, dtype=torch.float32)
                diff = client_param - global_param
                weight_accumulator[name].add_(diff * trust_weight)

        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)

        return True
