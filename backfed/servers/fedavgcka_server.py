"""
FedAvgCKA server implementation for FL.

This implements the FedAvgCKA defense as described in "Exploiting Layerwise Feature 
Representation Similarity For Backdoor Defence in Federated Learning" by Walter et al. (ESORICS 2024).

FedAvgCKA is a pre-aggregation defense that computes Centered Kernel Alignment (CKA) 
similarity between client updates to identify and filter out potentially malicious clients.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import copy
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque

from backfed.servers.defense_categories import RobustAggregationServer
from backfed.utils.logging_utils import log
from backfed.const import StateDict, client_id, num_examples
from logging import INFO, WARNING, ERROR


class FedAvgCKAServer(RobustAggregationServer):
    """
    Server implementing FedAvgCKA defense - a pre-aggregation filter based on 
    Centered Kernel Alignment similarity between client model activations.
    
    The defense works by:
    1. Extracting activations from a reference dataset using each client model
    2. Computing pairwise CKA similarities between client activations
    3. Filtering out clients with low average CKA similarity (potential attackers)
    4. Applying standard FedAvg aggregation to filtered client updates
    """
    
    defense_categories = ["robust_aggregation", "pre_aggregation"]

    def __init__(self, server_config, server_type="fedavgcka", eta=0.1, 
                 trim_fraction=0.3, layer_comparison="penultimate", 
                 root_dataset_size=64, root_sampling_strategy="class_balanced",
                 multi_layer_weights=None, log_scores=False, **kwargs):
        """
        Initialize FedAvgCKA server.
        
        Args:
            server_config: Server configuration
            server_type: Type identifier
            eta: Learning rate for aggregation
            trim_fraction: Fraction of clients to exclude (default: 0.3)
            layer_comparison: Which layer(s) to use for CKA ("penultimate", "layer2", "layer3", "multi_layer")
            root_dataset_size: Size of reference dataset for CKA computation
            root_sampling_strategy: How to sample reference data ("random" or "class_balanced")
            multi_layer_weights: Weights for multi-layer comparison (dict)
            log_scores: Whether to log detailed CKA scores
        """
        super().__init__(server_config, server_type, eta, **kwargs)
        
        self.trim_fraction = trim_fraction
        self.layer_comparison = layer_comparison
        self.root_dataset_size = root_dataset_size
        self.root_sampling_strategy = root_sampling_strategy
        self.log_scores = log_scores
        
        # Multi-layer weights (layer_name -> weight)
        if multi_layer_weights is None:
            self.multi_layer_weights = {}
        else:
            self.multi_layer_weights = multi_layer_weights
            
        # Initialize root dataset
        self._init_root_dataset()
        
        log(INFO, f"Initialized FedAvgCKA server:")
        log(INFO, f"  - trim_fraction: {self.trim_fraction}")
        log(INFO, f"  - layer_comparison: {self.layer_comparison}")
        log(INFO, f"  - root_dataset_size: {self.root_dataset_size}")
        log(INFO, f"  - root_sampling_strategy: {self.root_sampling_strategy}")

    def _init_root_dataset(self):
        """Initialize the root dataset for CKA computation."""
        try:
            self.root_dataset = self._create_root_dataset(
                size=self.root_dataset_size,
                strategy=self.root_sampling_strategy
            )
            log(INFO, f"Created root dataset with {self.root_dataset_size} samples")
        except Exception as e:
            log(ERROR, f"Failed to create root dataset: {e}")
            # Fallback to smaller dataset
            try:
                self.root_dataset = self._create_root_dataset(size=32, strategy="random")
                log(WARNING, "Using fallback root dataset with 32 random samples")
            except Exception as e2:
                log(ERROR, f"Failed to create fallback root dataset: {e2}")
                self.root_dataset = None

    def _create_root_dataset(self, size: int = 64, strategy: str = "class_balanced") -> DataLoader:
        """Create root dataset from test data."""
        if size <= 0:
            raise ValueError(f"Root dataset size must be positive, got {size}")
            
        if strategy not in ["random", "class_balanced"]:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        
        test_dataset = self.testset
        total_samples = len(test_dataset)
        
        if size > total_samples:
            log(WARNING, f"Requested size {size} > available samples {total_samples}. Using all samples.")
            size = total_samples
        
        if strategy == "random":
            indices = np.random.choice(total_samples, size=size, replace=False)
            
        elif strategy == "class_balanced":
            try:
                # Group samples by class
                class_to_indices = defaultdict(list)
                for idx in range(total_samples):
                    data, label = test_dataset[idx]
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    class_to_indices[label].append(idx)
                
                n_classes = len(class_to_indices)
                samples_per_class = size // n_classes
                remaining_samples = size % n_classes
                
                indices = []
                for class_idx, (label, class_indices) in enumerate(class_to_indices.items()):
                    class_size = samples_per_class + (1 if class_idx < remaining_samples else 0)
                    class_size = min(class_size, len(class_indices))
                    
                    selected = np.random.choice(class_indices, size=class_size, replace=False)
                    indices.extend(selected)
                
                log(INFO, f"Created class-balanced root dataset: {len(indices)} samples across {n_classes} classes")
                
            except Exception as e:
                log(WARNING, f"Class-balanced sampling failed: {e}. Falling back to random sampling.")
                indices = np.random.choice(total_samples, size=size, replace=False)
        
        root_subset = Subset(test_dataset, indices)
        root_loader = DataLoader(
            root_subset, 
            batch_size=size,
            shuffle=False,
            num_workers=0
        )
        
        return root_loader

    def linear_cka(self, X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
        """
        Compute linear CKA similarity between two activation matrices.
        
        Args:
            X: Activation matrix [k, d1] 
            Y: Activation matrix [k, d2] 
            eps: Small constant for numerical stability
            
        Returns:
            CKA similarity score in [0, 1]
        """
        assert X.shape[0] == Y.shape[0], f"Batch size mismatch: {X.shape[0]} vs {Y.shape[0]}"
        
        n = X.shape[0]
        if n <= 1:
            log(WARNING, f"CKA computation with n={n} samples may be unreliable")
            return 0.0
        
        # Center the matrices (H = I - (1/n)*11^T)
        H = torch.eye(n, device=X.device) - torch.ones(n, n, device=X.device) / n
        
        # Apply centering: X_centered = HX, Y_centered = HY  
        X_centered = torch.mm(H, X)
        Y_centered = torch.mm(H, Y)
        
        # Compute HSIC values using linear kernel
        hsic_xy = torch.trace(torch.mm(X_centered, Y_centered.t())) / ((n - 1) ** 2)
        hsic_xx = torch.trace(torch.mm(X_centered, X_centered.t())) / ((n - 1) ** 2)  
        hsic_yy = torch.trace(torch.mm(Y_centered, Y_centered.t())) / ((n - 1) ** 2)
        
        # Compute normalized CKA score
        denominator = torch.sqrt(hsic_xx * hsic_yy)
        if denominator < eps:
            log(WARNING, f"Small CKA denominator: {denominator}. Returning 0.0")
            return 0.0
            
        cka_score = hsic_xy / denominator
        cka_score = torch.clamp(cka_score, 0.0, 1.0)
        
        return float(cka_score)

    def get_layer_activations(self, model: nn.Module, data_loader: DataLoader, layer_name: str) -> torch.Tensor:
        """Extract activations from a specific layer of the model."""
        model.eval()
        model.to(self.device)
        
        activations = []
        
        def hook_fn(module, input, output):
            # Flatten spatial dimensions if needed
            if output.dim() > 2:
                output = output.view(output.size(0), -1)
            activations.append(output.detach().cpu())
        
        # Find and register hook
        target_layer = None
        for name, module in model.named_modules():
            if name == layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            available_layers = [name for name, _ in model.named_modules()]
            raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available_layers}")
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                for batch_data in data_loader:
                    if isinstance(batch_data, (list, tuple)):
                        inputs = batch_data[0].to(self.device)
                    else:
                        inputs = batch_data.to(self.device)
                    
                    _ = model(inputs)
            
            if not activations:
                raise RuntimeError("No activations captured. Check layer name and data loader.")
                
            activation_matrix = torch.cat(activations, dim=0)
            
            # Row-center the activation matrix
            activation_matrix = activation_matrix - activation_matrix.mean(dim=0, keepdim=True)
            
            return activation_matrix
            
        finally:
            handle.remove()

    def get_penultimate_layer_name(self, model: nn.Module) -> str:
        """Automatically determine the penultimate layer name."""
        # Check for ResNet architecture
        if hasattr(model, 'fc') and hasattr(model, 'avgpool'):
            return 'avgpool'
        
        # Check for SimpleNet architecture  
        if hasattr(model, 'fc2') and hasattr(model, 'fc1'):
            return 'fc1'
            
        # Check for VGG-like architectures
        if hasattr(model, 'classifier') and hasattr(model, 'features'):
            return 'features'
        
        # Generic fallback - find the second-to-last named module
        layers = list(model.named_modules())
        if len(layers) < 2:
            raise ValueError("Model too simple to determine penultimate layer")
            
        penultimate_name = layers[-2][0]
        log(INFO, f"Auto-detected penultimate layer: '{penultimate_name}'")
        return penultimate_name

    def get_layer_names_for_comparison(self, model: nn.Module) -> List[str]:
        """Get layer names based on comparison configuration."""
        if self.layer_comparison == "penultimate":
            return [self.get_penultimate_layer_name(model)]
        
        elif self.layer_comparison == "layer2":
            if hasattr(model, 'layer2'):
                return ['layer2']
            elif hasattr(model, 'conv2'):
                return ['conv2']
            else:
                raise ValueError("Model does not have layer2 or conv2 for layer2 comparison")
                
        elif self.layer_comparison == "layer3":
            if hasattr(model, 'layer3'):
                return ['layer3']
            elif hasattr(model, 'fc1'):
                return ['fc1']
            else:
                raise ValueError("Model does not have layer3 or fc1 for layer3 comparison")
                
        elif self.layer_comparison == "multi_layer":
            layers = []
            
            try:
                layers.append(self.get_penultimate_layer_name(model))
            except ValueError:
                log(WARNING, "Could not find penultimate layer for multi-layer comparison")
            
            if hasattr(model, 'layer3'):
                layers.append('layer3')
            elif hasattr(model, 'fc1'):
                layers.append('fc1')
                
            if hasattr(model, 'layer2'):
                layers.append('layer2')
            elif hasattr(model, 'conv2'):
                layers.append('conv2')
            
            if not layers:
                raise ValueError("Could not find any suitable layers for multi-layer comparison")
                
            return layers
        
        else:
            raise ValueError(f"Unknown layer_comparison option: {self.layer_comparison}")

    def rank_clients_by_cka(self, activations: Dict[int, torch.Tensor]) -> Tuple[List[int], List[int], Dict[int, float]]:
        """Rank clients by CKA similarity and determine exclusions."""
        if not activations:
            return [], [], {}
            
        client_ids = list(activations.keys())
        n_clients = len(client_ids)
        
        if n_clients == 1:
            return client_ids, [], {client_ids[0]: 1.0}
        
        # Compute pairwise CKA scores
        cka_matrix = torch.zeros(n_clients, n_clients)
        
        log(INFO, f"Computing pairwise CKA scores for {n_clients} clients...")
        
        for i in range(n_clients):
            for j in range(i, n_clients):
                if i == j:
                    cka_score = 1.0
                else:
                    client_i, client_j = client_ids[i], client_ids[j]
                    cka_score = self.linear_cka(activations[client_i], activations[client_j])
                
                cka_matrix[i, j] = cka_score
                cka_matrix[j, i] = cka_score
        
        # Calculate average CKA score for each client
        avg_cka_scores = {}
        for i, client_id in enumerate(client_ids):
            if n_clients > 1:
                mask = torch.ones(n_clients, dtype=torch.bool)
                mask[i] = False
                avg_score = cka_matrix[i, mask].mean().item()
            else:
                avg_score = 1.0
            avg_cka_scores[client_id] = avg_score
        
        # Sort clients by average CKA score (ascending order - low similarity clients first)
        sorted_clients = sorted(client_ids, key=lambda cid: avg_cka_scores[cid])
        
        # Determine exclusions
        n_exclude = int(self.trim_fraction * n_clients)
        n_exclude = min(n_exclude, n_clients - 1)  # Keep at least one client
        
        excluded_clients = sorted_clients[:n_exclude]
        selected_clients = sorted_clients[n_exclude:]
        
        log(INFO, f"CKA ranking: selected {len(selected_clients)}, excluded {len(excluded_clients)}")
        return selected_clients, excluded_clients, avg_cka_scores

    def apply_fedavgcka_filter(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> Tuple[List[Tuple[client_id, num_examples, StateDict]], Dict[str, Any]]:
        """Apply FedAvgCKA filtering before aggregation."""
        if not client_updates or self.root_dataset is None:
            return client_updates, {"error": "No client updates or root dataset"}
        
        start_time = time.time()
        
        try:
            # Extract client models from updates  
            client_models = {}
            for client_id, num_examples, client_state in client_updates:
                # Create model copy and load client state
                client_model = copy.deepcopy(self.global_model)
                client_model.load_state_dict(client_state)
                client_models[client_id] = client_model
            
            sample_model = next(iter(client_models.values()))
            layer_names = self.get_layer_names_for_comparison(sample_model)
            
            client_ids = list(client_models.keys())
            
            if self.layer_comparison == "multi_layer":
                # Multi-layer comparison
                combined_scores = self._compute_multi_layer_cka_scores(
                    client_models, 
                    self.root_dataset, 
                    layer_names
                )
                
                sorted_clients = sorted(client_ids, key=lambda cid: combined_scores.get(cid, 0.0))
                
                n_exclude = int(self.trim_fraction * len(client_ids))
                n_exclude = min(n_exclude, len(client_ids) - 1)
                
                excluded_clients = sorted_clients[:n_exclude]
                selected_clients = sorted_clients[n_exclude:]
                cka_scores = combined_scores
                
            else:
                # Single layer comparison
                layer_name = layer_names[0]
                log(INFO, f"Extracting activations from layer: {layer_name}")
                
                activations = {}
                failed_clients = []
                
                for client_id, model in client_models.items():
                    try:
                        client_activations = self.get_layer_activations(model, self.root_dataset, layer_name)
                        activations[client_id] = client_activations
                    except Exception as e:
                        log(ERROR, f"Failed to extract activations for client {client_id}: {e}")
                        failed_clients.append(client_id)
                
                if len(activations) < 2:
                    log(WARNING, "Too few clients with valid activations. Skipping FedAvgCKA filtering.")
                    return client_updates, {
                        "error": "Insufficient valid activations",
                        "failed_clients": failed_clients,
                        "selected_clients": [cid for cid, _, _ in client_updates],
                        "excluded_clients": []
                    }
                
                selected_clients, excluded_clients, cka_scores = self.rank_clients_by_cka(activations)
            
            # Filter client updates
            filtered_updates = [
                client_update 
                for client_update in client_updates 
                if client_update[0] in selected_clients
            ]
            
            compute_time = time.time() - start_time
            
            telemetry = {
                "selected_clients": selected_clients,
                "excluded_clients": excluded_clients,
                "cka_scores": cka_scores,
                "layer_names": layer_names,
                "n_selected": len(selected_clients),
                "n_excluded": len(excluded_clients), 
                "trim_fraction": self.trim_fraction,
                "compute_time_s": compute_time
            }
            
            if self.log_scores:
                log(INFO, f"FedAvgCKA filtering complete:")
                log(INFO, f"  Selected: {len(selected_clients)} clients {selected_clients}")
                log(INFO, f"  Excluded: {len(excluded_clients)} clients {excluded_clients}")
                log(INFO, f"  Compute time: {compute_time:.2f} s")
            
            return filtered_updates, telemetry
            
        except Exception as e:
            log(ERROR, f"FedAvgCKA filtering failed: {e}")
            return client_updates, {
                "error": str(e),
                "selected_clients": [cid for cid, _, _ in client_updates],
                "excluded_clients": []
            }

    def _compute_multi_layer_cka_scores(self, client_models: Dict[int, nn.Module], root_loader: DataLoader, layer_names: List[str]) -> Dict[int, float]:
        """Compute weighted combination of CKA scores across multiple layers."""
        client_ids = list(client_models.keys())
        combined_scores = {client_id: 0.0 for client_id in client_ids}
        
        log(INFO, f"Computing multi-layer CKA scores for layers: {layer_names}")
        
        for layer_name in layer_names:
            log(INFO, f"Processing layer: {layer_name}")
            
            layer_activations = {}
            for client_id, model in client_models.items():
                try:
                    activations = self.get_layer_activations(model, root_loader, layer_name)
                    layer_activations[client_id] = activations
                except Exception as e:
                    log(ERROR, f"Failed to extract activations from {layer_name} for client {client_id}: {e}")
                    continue
            
            if len(layer_activations) < 2:
                log(WARNING, f"Too few clients ({len(layer_activations)}) have valid activations for {layer_name}, skipping")
                continue
            
            _, _, layer_cka_scores = self.rank_clients_by_cka(layer_activations)
            
            weight = self.multi_layer_weights.get(layer_name, 1.0 / len(layer_names))
            
            for client_id in layer_cka_scores:
                if client_id in combined_scores:
                    combined_scores[client_id] += weight * layer_cka_scores[client_id]
        
        log(INFO, f"Multi-layer CKA computation complete. Combined scores: {combined_scores}")
        return combined_scores

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> bool:
        """
        Aggregate client updates using FedAvgCKA filtering.
        
        This method implements the core FedAvgCKA defense:
        1. Apply CKA-based filtering to exclude suspicious clients
        2. Pass filtered updates to standard FedAvg aggregation
        """
        if not client_updates:
            log(WARNING, "No client updates found, using global model")
            return False
        
        # Apply FedAvgCKA filtering
        filtered_updates, telemetry = self.apply_fedavgcka_filter(client_updates)
        
        # Log filtering results
        if "error" not in telemetry:
            log(INFO, f"FedAvgCKA filtered {len(client_updates)} → {len(filtered_updates)} clients")
            
            # Log to CSV/WandB if enabled
            if self.config.save_logging in ["csv", "both"]:
                round_metrics = {
                    "fedavgcka_n_selected": telemetry["n_selected"],
                    "fedavgcka_n_excluded": telemetry["n_excluded"],
                    "fedavgcka_compute_time": telemetry["compute_time_s"]
                }
                self.csv_logger.log(round_metrics, step=self.current_round)
                
        else:
            log(WARNING, f"FedAvgCKA filtering failed: {telemetry['error']}")
        
        # Apply standard FedAvg aggregation to filtered updates
        return super().aggregate_client_updates(filtered_updates)
