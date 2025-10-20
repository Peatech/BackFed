"""
Implementation of FLTrust server for federated learning.
"""

import torch
import torch.nn.functional as F
import copy

from typing import Dict, List, Tuple
from logging import INFO
from torch.utils.data import DataLoader, TensorDataset
from backfed.datasets import FL_DataLoader
from backfed.servers.defense_categories import RobustAggregationServer
from backfed.utils.logging_utils import log
from hydra.utils import instantiate

class FLTrustServer(RobustAggregationServer):
    """
    FLTrust server implementation that uses cosine similarity with trusted data
    to assign trust scores to client updates.
    """

    def __init__(self, 
        server_config, 
        server_type = "fltrust", 
        eta: float = 0.1,
        m: int = 100, # Number of samples in server's root dataset
    ):
        self.m = m
        
        super().__init__(server_config, server_type, eta) # Setup datasets and so on
        
        self.global_lr = self.config.client_config.lr
        self.global_epochs = 1 # Follow original paper

    def _prepare_dataset(self):
        """Very hacky. We override the _prepare_dataset function to load auxiliary clean data for the defense."""
        super()._prepare_dataset()
                                    
        if self.m > len(self.testset):
            raise ValueError(f"FLTrust: m ({self.m}) is larger than test set size ({len(self.testset)})")

        random_indices = torch.randperm(len(self.testset))[:self.m]

        self.server_root_data = TensorDataset(torch.stack([self.normalization(self.testset[i][0]) for i in random_indices]),
                                                torch.tensor([self.testset[i][1] for i in random_indices]))
        self.server_dataloader = DataLoader(self.server_root_data, 
                                    batch_size=self.config.client_config.batch_size, # Follow client batch size
                                    shuffle=False, 
                                    num_workers=self.config.num_workers,
                                    pin_memory=self.config.pin_memory,
                                )

    def _central_update(self):
        """Perform update on the server's root dataset to obtain the central update."""
        ref_model = copy.deepcopy(self.global_model)
        ref_model.to(self.device)
        ref_model.train()

        # Create server optimizer
        server_optimizer = instantiate(self.config.client_config.optimizer, 
                                       params=ref_model.parameters())
        
        loss_func = torch.nn.CrossEntropyLoss()
        for epoch in range(self.global_epochs):
            for data, label in self.server_dataloader:
                data, label = data.to(self.device), label.to(self.device)
                server_optimizer.zero_grad()
                preds = ref_model(data)
                loss = loss_func(preds, label)
                loss.backward()
                server_optimizer.step()

        return self.parameters_dict_to_vector(ref_model.state_dict()) - self.parameters_dict_to_vector(self.global_model.state_dict())
    
    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, Dict]]) -> bool:
        """
        Aggregate client updates using FLTrust mechanism.

        Args:
            client_updates: List of (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False
        
        central_update = self._central_update()
        central_norm = torch.linalg.norm(central_update)

        score_list = []
        client_ids = []
        total_score = 0
        sum_parameters = {}

        global_vector = self.parameters_dict_to_vector(self.global_model.state_dict())
        for client_id, _, local_update in client_updates:
            # Convert local update to vector
            local_vector = self.parameters_dict_to_vector(local_update) - global_vector

            # Calculate cosine similarity and trust score
            client_cos = F.cosine_similarity(central_update, local_vector, dim=0)
            client_cos = max(client_cos.item(), 0) # ReLU
            local_norm = torch.linalg.norm(local_vector)
            client_norm_ratio = central_norm / (local_norm + 1e-12)

            score_list.append(client_cos)
            client_ids.append(client_id)
            total_score += client_cos

        # If all scores are 0, return current global model
        if total_score == 0:
            log(INFO, "FLTrust: All trust scores are 0, keeping current model")
            return False

        fltrust_weights = [score/total_score for score in score_list]
        if self.verbose:
            log(INFO, f"FLTrust weights (client_id, weight): {list(zip(client_ids, fltrust_weights))}")

        weight_accumulator = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.global_model.state_dict().items()
        }

        global_state_dict = self.global_model.state_dict()
        for weight, (cid, num_samples, client_state) in zip(fltrust_weights, client_updates):
            for name, param in client_state.items():
                if any(pattern in name for pattern in self.ignore_weights):
                    continue
                if name in global_state_dict:
                    diff = param.to(self.device) - global_state_dict[name]
                    weight_accumulator[name].add_(diff * weight)

        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)

        return True    
