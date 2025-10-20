"""
WeakDP server implementation for differential privacy with fixed clipping.
"""
import torch
from typing import List, Tuple
from logging import INFO, WARNING

from backfed.servers.defense_categories import RobustAggregationServer
from backfed.utils.logging_utils import log
from backfed.const import StateDict, client_id, num_examples

class NormClippingServer(RobustAggregationServer):
    """
    Server that clips the norm of client updates to defend against poisoning attacks.
    """
    def __init__(self, server_config, server_type="norm_clipping", clipping_norm=5.0, eta=0.1, verbose=True):
        """
        Args:
            server_config: Configuration for the server.
            server_type: Type of server.
            clipping_norm: Clipping norm for the norm clipping.
            eta: Learning rate for the server.
            verbose: Whether to log verbose information.
        """
        super(NormClippingServer, self).__init__(server_config, server_type, eta)
        self.clipping_norm = clipping_norm
        self.verbose = verbose
        log(INFO, f"Initialized NormClipping server with clipping_norm={clipping_norm}, eta={eta}")

    def clip_updates_inplace(self, client_ids: List[client_id], client_diffs: List[StateDict]) -> None:
        """
        Clip the norm of client_diffs (L_i - G) in-place based on trainable parameters only.

        Args:
            client_ids: List of client IDs
            client_diffs: List of client_diffs (state dicts)
        """
        for client_id, client_diff in zip(client_ids, client_diffs):
            flatten_weights = []
            for name, param in client_diff.items():
                # Skip buffers (non-trainable parameters)
                if "running" in name or "num_batches_tracked" in name:
                    continue
                flatten_weights.append(param.view(-1))

            if not flatten_weights:
                continue

            flatten_weights = torch.cat(flatten_weights)
            weight_diff_norm = torch.linalg.norm(flatten_weights, ord=2)

            if weight_diff_norm > self.clipping_norm:
                if self.verbose:
                    log(INFO, f"Client {client_id} weight diff norm {weight_diff_norm} -> {self.clipping_norm}")
            
                scaling_factor = self.clipping_norm / weight_diff_norm
                for name, param in client_diff.items():
                    # Only scale trainable parameters
                    if "running" in name or "num_batches_tracked" in name:
                        continue
                    if any(pattern in name for pattern in self.ignore_weights):
                        continue
                    client_diff[name].mul_(scaling_factor)
            else:
                if self.verbose:
                    log(INFO, f"Client {client_id} weight diff norm {weight_diff_norm} within the clipping norm.")

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> StateDict:
        """Aggregate client updates with norm clipping."""
        if len(client_updates) == 0:
            log(WARNING, "NormClipping: No client updates found")
            return False

        # Clip updates
        client_diffs = []
        client_ids = []
        global_state_dict = dict(self.global_model.named_parameters())

        for client_id, num_examples, client_params in client_updates:
            diff_dict = {}
            for name, param in client_params.items():
                if any(pattern in name for pattern in self.ignore_weights):
                    continue
                if name in global_state_dict:
                    diff_dict[name] = param.to(self.device) - global_state_dict[name]
            client_diffs.append(diff_dict)
            client_ids.append(client_id)

        self.clip_updates_inplace(client_ids, client_diffs)

        client_weight = 1 / len(client_updates)

        # Update global model with clipped weight differences
        for i, client_diff in enumerate(client_diffs):
            for name, diff in client_diff.items():
                if name in global_state_dict:
                    global_state_dict[name].data.add_(diff * client_weight * self.eta)

        return True

class WeakDPServer(NormClippingServer):
    """
    Server that implements differential privacy with fixed clipping and Gaussian noise.
    """
    def __init__(self, server_config, server_type="weakdp", strategy="unweighted_fedavg",
                 std_dev=0.025, clipping_norm=5.0, eta=0.1):

        """
        Args:
            server_config: Configuration for the server.
            server_type: Type of server.
            strategy: Strategy for the server.
            std_dev: Standard deviation for the Gaussian noise.
            clipping_norm: Clipping norm for the Gaussian noise.
        """
        super(WeakDPServer, self).__init__(server_config, server_type, clipping_norm=clipping_norm, eta=eta)

        if std_dev < 0:
            raise ValueError("The std_dev should be a non-negative value.")
        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")

        self.std_dev = std_dev
        self.strategy = strategy
        log(INFO, f"Initialized WeakDP server with std_dev={std_dev}, clipping_norm={clipping_norm}")

    @torch.no_grad()
    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> StateDict:
        """Aggregate client updates with DP guarantees by adding Gaussian noise to trainable parameters."""
        super().aggregate_client_updates(client_updates)

        for name, param in self.global_model.named_parameters():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            noise = torch.normal(0, self.std_dev, param.shape, device=param.device) * self.eta
            param.data.add_(noise)

        return True

    def __repr__(self) -> str:
        return f"WeakDP(strategy={self.strategy}, std_dev={self.std_dev}, clipping_norm={self.clipping_norm})"