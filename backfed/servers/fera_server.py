"""
FeRA (Feature Representation Anomaly) Defense Server Implementation

This defense detects malicious clients by analyzing feature representations using:
1. Spectral norm: Measures concentration of change in representations
2. Delta norm: Measures total deviation from global representations

Key features:
- Multi-layer feature extraction and analysis
- Configurable signal weighting
- Robust normalization using median + IQR
- Top-K% threshold-based detection
- Graceful error handling

Author: AI Assistant
Date: 2025-10-20
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from logging import INFO, WARNING

from backfed.servers.defense_categories import AnomalyDetectionServer
from backfed.utils.system_utils import log
from backfed.const import client_id, num_examples, StateDict


class FeRAServer(AnomalyDetectionServer):
    """
    FeRA Defense Server: Detects malicious clients using spectral and delta norm signals.
    
    The defense computes:
    1. Spectral norm: Largest eigenvalue of delta covariance matrix
    2. Delta norm: Frobenius norm of representation differences
    
    These signals are normalized and combined to produce anomaly scores.
    """
    
    defense_categories = ["anomaly_detection"]
    
    def __init__(
        self,
        server_config,
        server_type: str = "fera",
        eta: float = 0.5,
        # Signal weights
        spectral_weight: float = 0.6,
        delta_weight: float = 0.4,
        # Detection parameters
        top_k_percent: float = 0.5,
        # Outlier removal (for scaled norm attacks like Anticipate)
        remove_outliers: bool = True,
        outlier_threshold: float = 3.0,
        # Multi-layer options
        use_multi_layer: bool = False,
        layers: List[str] = None,
        combine_layers_method: str = 'mean',
        # Root dataset
        root_size: int = 64,
        use_ood_root_dataset: bool = False,
        **kwargs
    ):
        """
        Initialize FeRA server.
        
        Args:
            server_config: Configuration dictionary
            server_type: Type of server (default: "fera")
            eta: Learning rate for aggregation
            spectral_weight: Weight for spectral norm signal
            delta_weight: Weight for delta norm signal
            top_k_percent: Percentage of clients to flag as malicious (0.0 to 1.0)
            remove_outliers: Whether to remove extreme outliers from benign cluster
            outlier_threshold: Z-score threshold for outlier removal (e.g., 3.0 = 3 std devs)
            use_multi_layer: Whether to use multi-layer analysis
            layers: List of layer names to extract features from
            combine_layers_method: How to combine multi-layer scores ('mean', 'max', 'vote')
            root_size: Size of root dataset for feature extraction
            use_ood_root_dataset: Whether to use out-of-distribution data
        """
        super().__init__(server_config, server_type, eta, **kwargs)
        
        # Validate weights sum to 1.0
        total_weight = spectral_weight + delta_weight
        if not np.isclose(total_weight, 1.0):
            log(WARNING, f"Signal weights sum to {total_weight}, normalizing to 1.0")
            spectral_weight = spectral_weight / total_weight
            delta_weight = delta_weight / total_weight
        
        # Store parameters
        self.spectral_weight = spectral_weight
        self.delta_weight = delta_weight
        self.top_k_percent = np.clip(top_k_percent, 0.0, 1.0)
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        self.use_multi_layer = use_multi_layer
        self.layers = layers if layers is not None else ['penultimate']
        self.combine_layers_method = combine_layers_method
        self.root_size = root_size
        self.use_ood_root_dataset = use_ood_root_dataset
        
        # Create root dataset loader
        self.root_loader = self._create_root_loader()
        
        log(INFO, f"═══ Initialized FeRA Defense ═══")
        log(INFO, f"  Detection strategy: Flag BOTTOM {self.top_k_percent:.0%} (consistency-based)")
        log(INFO, f"  Rationale: Backdoors are more consistent than natural variance")
        log(INFO, f"  Spectral weight: {self.spectral_weight:.2f}")
        log(INFO, f"  Delta weight: {self.delta_weight:.2f}")
        log(INFO, f"  Outlier removal: {self.remove_outliers} (threshold: {self.outlier_threshold}σ)")
        log(INFO, f"  Multi-layer: {self.use_multi_layer}")
        if self.use_multi_layer:
            log(INFO, f"  Layers: {self.layers}")
            log(INFO, f"  Combine method: {self.combine_layers_method}")
        log(INFO, f"  Root dataset size: {self.root_size}")
        log(INFO, f"═══════════════════════════════════")
    
    def _create_root_loader(self):
        """Create root dataset loader for feature extraction."""
        from torch.utils.data import DataLoader, Subset
        import random
        
        # Sample subset of test data
        indices = list(range(len(self.testset)))
        random.seed(self.config.seed)
        random.shuffle(indices)
        subset_indices = indices[:self.root_size]
        
        root_dataset = Subset(self.testset, subset_indices)
        
        root_loader = DataLoader(
            root_dataset,
            batch_size=min(self.root_size, 64),
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        return root_loader
    
    def detect_anomalies(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect malicious clients using FeRA method.
        
        Args:
            client_updates: List of (client_id, num_examples, model_state_dict)
        
        Returns:
            Tuple of (malicious_client_ids, benign_client_ids)
        """
        # Handle edge cases
        if len(client_updates) < 2:
            log(WARNING, "FeRA: Less than 2 clients, cannot perform detection")
            return [], [cid for cid, _, _ in client_updates]
        
        try:
            # Load client models
            client_models = self._load_client_models(client_updates)
            
            if self.use_multi_layer:
                # Multi-layer analysis
                malicious_clients, benign_clients = self._detect_multi_layer(client_models)
            else:
                # Single-layer analysis
                malicious_clients, benign_clients = self._detect_single_layer(client_models)
            
            return malicious_clients, benign_clients
        
        except Exception as e:
            log(WARNING, f"FeRA: Detection failed with error: {str(e)}")
            log(WARNING, "FeRA: Falling back to no detection (all clients benign)")
            return [], [cid for cid, _, _ in client_updates]
    
    def _load_client_models(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Dict[int, nn.Module]:
        """Load client models from state dictionaries."""
        client_models = {}
        
        for cid, _, state_dict in client_updates:
            # Create a new model instance
            from backfed.utils import get_model
            model = get_model(
                model_name=self.config.model,
                num_classes=self.config.num_classes,
                dataset_name=self.config.dataset
            )
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            client_models[cid] = model
        
        return client_models
    
    def _detect_single_layer(
        self,
        client_models: Dict[int, nn.Module]
    ) -> Tuple[List[int], List[int]]:
        """Perform detection using single layer."""
        layer_name = self.layers[0]
        
        # Extract representations
        client_representations = self._extract_representations(
            client_models, layer_name
        )
        
        # Extract global representation
        global_representation = self._extract_global_representation(layer_name)
        
        # Compute signals
        spectral_scores = self._compute_spectral_norms(
            client_representations, global_representation
        )
        delta_scores = self._compute_delta_norms(
            client_representations, global_representation
        )
        
        # Normalize signals
        normalized_spectral = self._normalize_scores_robust(spectral_scores)
        normalized_delta = self._normalize_scores_robust(delta_scores)
        
        # Combine signals
        combined_scores = self._combine_signals(
            normalized_spectral, normalized_delta
        )
        
        # Apply two-sided adaptive filtering
        # Pass raw scores for outlier detection (preserves signal)
        malicious_clients, benign_clients = self._apply_threshold(
            combined_scores,
            spectral_scores,  # Raw norms before normalization
            delta_scores      # Raw norms before normalization
        )
        
        # Log results
        self._log_detection_results(
            spectral_scores, delta_scores,
            normalized_spectral, normalized_delta,
            combined_scores, malicious_clients, benign_clients
        )
        
        return malicious_clients, benign_clients
    
    def _detect_multi_layer(
        self,
        client_models: Dict[int, nn.Module]
    ) -> Tuple[List[int], List[int]]:
        """Perform detection using multiple layers."""
        layer_scores = {}
        layer_raw_spectral = {}
        layer_raw_delta = {}
        layer_detections = {}
        
        log(INFO, f"FeRA: Performing multi-layer analysis on {len(self.layers)} layers")
        
        # Analyze each layer
        for layer_name in self.layers:
            log(INFO, f"FeRA: Analyzing layer '{layer_name}'")
            
            try:
                # Extract representations
                client_representations = self._extract_representations(
                    client_models, layer_name
                )
                
                # Extract global representation
                global_representation = self._extract_global_representation(layer_name)
                
                # Compute signals
                spectral_scores = self._compute_spectral_norms(
                    client_representations, global_representation
                )
                delta_scores = self._compute_delta_norms(
                    client_representations, global_representation
                )
                
                # Store raw scores for later outlier detection
                layer_raw_spectral[layer_name] = spectral_scores
                layer_raw_delta[layer_name] = delta_scores
                
                # Normalize signals
                normalized_spectral = self._normalize_scores_robust(spectral_scores)
                normalized_delta = self._normalize_scores_robust(delta_scores)
                
                # Combine signals
                combined_scores = self._combine_signals(
                    normalized_spectral, normalized_delta
                )
                
                layer_scores[layer_name] = combined_scores
                
                # Note: Per-layer detection not done here to avoid confusion
                # We'll do final detection after combining all layers
                
                log(INFO, f"  Layer '{layer_name}' analyzed successfully")
            
            except Exception as e:
                log(WARNING, f"FeRA: Failed to analyze layer '{layer_name}': {str(e)}")
                continue
        
        # Combine layer scores
        if not layer_scores:
            log(WARNING, "FeRA: No valid layer scores, falling back to all benign")
            return [], list(client_models.keys())
        
        combined_scores = self._combine_layer_scores(layer_scores)
        
        # Combine raw scores across layers (use mean for outlier detection)
        combined_spectral = self._combine_layer_raw_scores(layer_raw_spectral)
        combined_delta = self._combine_layer_raw_scores(layer_raw_delta)
        
        # Apply two-sided adaptive filtering with combined raw scores
        malicious_clients, benign_clients = self._apply_threshold(
            combined_scores,
            combined_spectral,
            combined_delta
        )
        
        # Log multi-layer results
        log(INFO, f"FeRA: Multi-layer combined detection flagged {len(malicious_clients)} clients: {malicious_clients}")
        log(INFO, f"FeRA: Per-layer detections:")
        for layer_name, detection in layer_detections.items():
            log(INFO, f"  {layer_name}: {detection['malicious']}")
        
        return malicious_clients, benign_clients
    
    def _extract_representations(
        self,
        client_models: Dict[int, nn.Module],
        layer_name: str
    ) -> Dict[int, torch.Tensor]:
        """
        Extract feature representations from specified layer for all clients.
        
        Args:
            client_models: Dictionary of client models
            layer_name: Name of layer to extract from
        
        Returns:
            Dictionary mapping client_id to representation tensor [n_samples, d_features]
        """
        client_representations = {}
        
        for cid, model in client_models.items():
            representations = []
            
            with torch.no_grad():
                for batch in self.root_loader:
                    inputs, _ = batch
                    inputs = inputs.to(self.device)
                    
                    # Extract features from specified layer
                    features = self._extract_features(model, inputs, layer_name)
                    representations.append(features.cpu())
            
            # Concatenate all batches: [n_samples, d_features]
            client_representations[cid] = torch.cat(representations, dim=0)
        
        return client_representations
    
    def _extract_global_representation(self, layer_name: str) -> torch.Tensor:
        """Extract representation from global model."""
        representations = []
        
        with torch.no_grad():
            for batch in self.root_loader:
                inputs, _ = batch
                inputs = inputs.to(self.device)
                
                features = self._extract_features(self.global_model, inputs, layer_name)
                representations.append(features.cpu())
        
        return torch.cat(representations, dim=0)
    
    def _extract_features(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """
        Extract features from a specific layer using hooks.
        
        Args:
            model: Neural network model
            inputs: Input tensor
            layer_name: Layer to extract from ('penultimate', 'layer2', 'layer3', etc.)
        
        Returns:
            Extracted features tensor
        """
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        # Find and register hook on target layer
        target_layer = self._get_target_layer(model, layer_name)
        
        if target_layer is None:
            log(WARNING, f"FeRA: Could not find layer '{layer_name}', using final output")
            output = model(inputs)
            return output.flatten(1)
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            _ = model(inputs)
            
            if not features:
                raise RuntimeError(f"Hook did not capture features for layer '{layer_name}'")
            
            # Flatten features if needed
            output_features = features[0]
            if len(output_features.shape) > 2:
                output_features = output_features.flatten(1)
            
            return output_features
        
        finally:
            handle.remove()
    
    def _get_target_layer(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """
        Get the target layer from model based on layer name.
        
        Args:
            model: Neural network model
            layer_name: Name of layer ('penultimate', 'layer2', 'layer3', etc.)
        
        Returns:
            Target layer module or None if not found
        """
        if layer_name == 'penultimate':
            # Get second-to-last layer
            layers = list(model.children())
            if len(layers) >= 2:
                return layers[-2]
        
        elif layer_name == 'layer2':
            # For ResNet-like architectures
            if hasattr(model, 'layer2'):
                return model.layer2
        
        elif layer_name == 'layer3':
            # For ResNet-like architectures
            if hasattr(model, 'layer3'):
                return model.layer3
        
        elif layer_name == 'layer4':
            # For ResNet-like architectures
            if hasattr(model, 'layer4'):
                return model.layer4
        
        elif layer_name == 'all':
            # Use penultimate as default for 'all'
            layers = list(model.children())
            if len(layers) >= 2:
                return layers[-2]
        
        # Try to access as attribute
        if hasattr(model, layer_name):
            return getattr(model, layer_name)
        
        return None
    
    def _compute_spectral_norms(
        self,
        client_representations: Dict[int, torch.Tensor],
        global_representation: torch.Tensor
    ) -> Dict[int, float]:
        """
        Compute spectral norms for each client.
        
        Spectral norm = largest eigenvalue of delta covariance matrix
        where delta = client_rep - global_rep
        
        Args:
            client_representations: Dict mapping client_id to representations
            global_representation: Global model representations
        
        Returns:
            Dict mapping client_id to spectral norm score
        """
        spectral_norms = {}
        
        for cid, client_rep in client_representations.items():
            try:
                # Compute delta
                delta = client_rep - global_representation
                # Shape: [n_samples, d_features]
                
                n_samples = delta.shape[0]
                
                # Check sufficient samples
                if n_samples <= 1:
                    log(WARNING, f"FeRA: Client {cid} has insufficient samples ({n_samples}), setting spectral norm to 0")
                    spectral_norms[cid] = 0.0
                    continue
                
                # Compute delta covariance matrix
                delta_centered = delta - delta.mean(dim=0, keepdim=True)
                cov_matrix = (delta_centered.T @ delta_centered) / (n_samples - 1)
                # Shape: [d_features, d_features]
                
                # Compute eigenvalues (use eigvalsh for symmetric matrices)
                eigenvalues = torch.linalg.eigvalsh(cov_matrix)
                # Returns: tensor of shape [d_features], sorted in ascending order
                
                # Extract spectral norm (largest eigenvalue)
                spectral_norm = eigenvalues[-1].item()
                
                # Handle numerical issues
                if spectral_norm < 0:
                    log(WARNING, f"FeRA: Negative eigenvalue for client {cid}, clamping to 0")
                    spectral_norm = 0.0
                
                spectral_norms[cid] = float(spectral_norm)
            
            except Exception as e:
                log(WARNING, f"FeRA: Failed to compute spectral norm for client {cid}: {str(e)}")
                spectral_norms[cid] = 0.0
        
        return spectral_norms
    
    def _compute_delta_norms(
        self,
        client_representations: Dict[int, torch.Tensor],
        global_representation: torch.Tensor
    ) -> Dict[int, float]:
        """
        Compute delta norms (Frobenius norm) for each client.
        
        Delta norm = ||client_rep - global_rep||_F
        
        Args:
            client_representations: Dict mapping client_id to representations
            global_representation: Global model representations
        
        Returns:
            Dict mapping client_id to delta norm score
        """
        delta_norms = {}
        
        for cid, client_rep in client_representations.items():
            try:
                # Compute delta
                delta = client_rep - global_representation
                
                # Compute Frobenius norm
                delta_norm = torch.norm(delta, p='fro').item()
                
                delta_norms[cid] = float(delta_norm)
            
            except Exception as e:
                log(WARNING, f"FeRA: Failed to compute delta norm for client {cid}: {str(e)}")
                delta_norms[cid] = 0.0
        
        return delta_norms
    
    def _normalize_scores_robust(
        self,
        scores: Dict[int, float],
        baseline_stats: Optional[Tuple[float, float]] = None
    ) -> Dict[int, float]:
        """
        Normalize scores to [0, 1] using robust statistics (median + IQR).
        
        Args:
            scores: Dictionary of raw scores
            baseline_stats: Optional (median, iqr) tuple for normalization
        
        Returns:
            Dictionary of normalized scores
        """
        if not scores:
            return {}
        
        score_values = np.array(list(scores.values()))
        
        # Compute robust statistics
        if baseline_stats is None:
            median = np.median(score_values)
            q75 = np.percentile(score_values, 75)
            q25 = np.percentile(score_values, 25)
            iqr = q75 - q25
        else:
            median, iqr = baseline_stats
        
        # Normalize using median + IQR
        normalized_scores = {}
        for cid, score in scores.items():
            if iqr > 0:
                # Normalize to [0, 1] range
                normalized = (score - median) / iqr
                # Clip to reasonable range
                normalized = np.clip(normalized, 0.0, 1.0)
            else:
                # All scores are the same
                normalized = 0.5
            
            normalized_scores[cid] = float(normalized)
        
        return normalized_scores
    
    def _combine_signals(
        self,
        spectral_scores: Dict[int, float],
        delta_scores: Dict[int, float]
    ) -> Dict[int, float]:
        """
        Combine spectral and delta signals using weighted sum.
        
        Args:
            spectral_scores: Normalized spectral norm scores
            delta_scores: Normalized delta norm scores
        
        Returns:
            Combined anomaly scores
        """
        combined_scores = {}
        
        # Get all client IDs
        all_clients = set(spectral_scores.keys()) | set(delta_scores.keys())
        
        for cid in all_clients:
            spectral = spectral_scores.get(cid, 0.0)
            delta = delta_scores.get(cid, 0.0)
            
            combined = self.spectral_weight * spectral + self.delta_weight * delta
            combined_scores[cid] = float(combined)
        
        return combined_scores
    
    def _combine_layer_scores(
        self,
        layer_scores: Dict[str, Dict[int, float]]
    ) -> Dict[int, float]:
        """
        Combine scores from multiple layers.
        
        Args:
            layer_scores: Dictionary mapping layer_name to client scores
        
        Returns:
            Combined scores across layers
        """
        if not layer_scores:
            return {}
        
        # Get all client IDs
        all_clients = set()
        for scores in layer_scores.values():
            all_clients.update(scores.keys())
        
        combined_scores = {}
        
        if self.combine_layers_method == 'mean':
            # Average scores across layers
            for cid in all_clients:
                scores = [layer_scores[layer].get(cid, 0.0) for layer in layer_scores]
                combined_scores[cid] = float(np.mean(scores))
        
        elif self.combine_layers_method == 'max':
            # Take maximum score across layers
            for cid in all_clients:
                scores = [layer_scores[layer].get(cid, 0.0) for layer in layer_scores]
                combined_scores[cid] = float(np.max(scores))
        
        elif self.combine_layers_method == 'vote':
            # Vote-based: count how many layers flag each client
            for cid in all_clients:
                scores = [layer_scores[layer].get(cid, 0.0) for layer in layer_scores]
                # Normalize by number of layers
                combined_scores[cid] = float(np.mean([s > 0.5 for s in scores]))
        
        else:
            log(WARNING, f"FeRA: Unknown combine method '{self.combine_layers_method}', using mean")
            for cid in all_clients:
                scores = [layer_scores[layer].get(cid, 0.0) for layer in layer_scores]
                combined_scores[cid] = float(np.mean(scores))
        
        return combined_scores
    
    def _combine_layer_raw_scores(
        self,
        layer_raw_scores: Dict[str, Dict[int, float]]
    ) -> Dict[int, float]:
        """
        Combine RAW scores from multiple layers for outlier detection.
        
        Uses simple mean to preserve the magnitude of raw norms.
        
        Args:
            layer_raw_scores: Dictionary mapping layer_name to raw scores
        
        Returns:
            Combined raw scores (mean across layers)
        """
        if not layer_raw_scores:
            return {}
        
        # Get all client IDs
        all_clients = set()
        for scores in layer_raw_scores.values():
            all_clients.update(scores.keys())
        
        combined_raw = {}
        
        # Always use mean for raw scores to preserve magnitude
        for cid in all_clients:
            scores = [layer_raw_scores[layer].get(cid, 0.0) for layer in layer_raw_scores]
            combined_raw[cid] = float(np.mean(scores))
        
        return combined_raw
    
    def _apply_threshold(
        self,
        combined_scores: Dict[int, float],
        spectral_scores: Dict[int, float],
        delta_scores: Dict[int, float]
    ) -> Tuple[List[int], List[int]]:
        """
        Apply two-sided adaptive filtering to detect malicious clients.
        
        **STAGE 1: Consistency-Based Detection (Inverted Filtering)**
        Backdoor attacks create MORE CONSISTENT feature representations than natural
        data variance. We flag the BOTTOM K% (lowest combined scores).
        
        **STAGE 2: Norm-Inflation Evasion Detection**
        Sophisticated attacks (e.g., Anticipate) create extremely large RAW norms
        (10^11) to escape detection. We compute Modified Z-scores on RAW spectral
        and delta norms using ALL clients as baseline, then flag extreme outliers.
        
        Args:
            combined_scores: Combined normalized anomaly scores [0,1] for ranking
            spectral_scores: Raw spectral norms (before normalization) for outlier detection
            delta_scores: Raw delta norms (before normalization) for outlier detection
        
        Returns:
            Tuple of (malicious_clients, benign_clients)
        """
        if not combined_scores:
            return [], []
        
        # STAGE 1: Consistency-Based Detection (Inverted Filtering)
        # Sort clients by combined score (ascending order)
        sorted_clients = sorted(combined_scores.items(), key=lambda x: x[1])
        
        # Determine number of malicious clients based on top_k_percent
        n_clients = len(sorted_clients)
        n_malicious = max(0, int(np.ceil(n_clients * self.top_k_percent)))
        
        # M_initial: Bottom K% (low variance = backdoor consistency)
        # B_initial: Top (1-K)% (high variance = normal diversity)
        malicious_clients_initial = [cid for cid, _ in sorted_clients[:n_malicious]]
        benign_clients_initial = [cid for cid, _ in sorted_clients[n_malicious:]]
        
        # STAGE 2: Norm-Inflation Evasion Detection
        # Detect extreme outliers in B_initial using raw norms and ALL clients as baseline
        if self.remove_outliers and len(benign_clients_initial) > 0:
            all_client_ids = malicious_clients_initial + benign_clients_initial
            
            outliers_flagged = self._detect_outliers(
                spectral_scores,
                delta_scores,
                benign_clients_initial,  # Check these for outliers
                all_client_ids,          # Use all as baseline
                threshold=self.outlier_threshold
            )
            
            if outliers_flagged:
                log(INFO, f"  Outlier detection: Flagging {len(outliers_flagged)} extreme outliers from benign cluster")
                log(INFO, f"  Outliers (norm-inflation attacks): {outliers_flagged}")
                
                # M_final = M_initial ∪ {flagged outliers}
                malicious_clients = malicious_clients_initial + outliers_flagged
                # B_final = B_initial \ {flagged outliers}
                benign_clients = [cid for cid in benign_clients_initial if cid not in outliers_flagged]
            else:
                malicious_clients = malicious_clients_initial
                benign_clients = benign_clients_initial
        else:
            malicious_clients = malicious_clients_initial
            benign_clients = benign_clients_initial
        
        return malicious_clients, benign_clients
    
    def _detect_outliers(
        self,
        spectral_scores: Dict[int, float],
        delta_scores: Dict[int, float],
        candidate_clients: List[int],
        all_client_ids: List[int],
        threshold: float = 3.0
    ) -> List[int]:
        """
        Detect extreme outliers using two-sided Modified Z-score on RAW norms.
        
        **KEY INSIGHT**: Operates on RAW spectral/delta norms BEFORE normalization
        to preserve the outlier signal. Anticipate creates norms of 10^11 which
        would be lost if we used normalized [0,1] scores.
        
        **BASELINE**: Computes MAD using ALL clients (M_initial ∪ B_initial) for
        robustness. Even if multiple attackers infiltrate B_initial, the median
        remains stable because low-norm consistency attackers anchor it.
        
        Args:
            spectral_scores: Raw spectral norms (largest eigenvalue of delta covariance)
            delta_scores: Raw delta norms (Frobenius norm of representation difference)
            candidate_clients: B_initial - clients to check for outliers
            all_client_ids: M_initial ∪ B_initial - all clients for baseline computation
            threshold: Modified Z-score threshold (default: 3.0σ)
        
        Returns:
            List of client IDs identified as outliers
        """
        if len(all_client_ids) <= 2 or len(candidate_clients) == 0:
            return []
        
        outliers = []
        outlier_info = {}  # Track which signal triggered each outlier
        
        # Compute Modified Z-scores for SPECTRAL norms
        spectral_vals = np.array([spectral_scores.get(cid, 0.0) for cid in all_client_ids])
        
        if len(spectral_vals) > 0 and not np.all(np.isnan(spectral_vals)):
            median_spectral = np.median(spectral_vals)
            mad_spectral = np.median(np.abs(spectral_vals - median_spectral))
            
            if mad_spectral == 0:
                # All scores identical - use absolute threshold
                for cid in candidate_clients:
                    score = spectral_scores.get(cid, 0.0)
                    if not np.isclose(score, median_spectral) and score > 1000 * max(abs(median_spectral), 1.0):
                        outliers.append(cid)
                        outlier_info[cid] = 'spectral (absolute)'
            else:
                # Compute Modified Z-scores for candidates
                for cid in candidate_clients:
                    score = spectral_scores.get(cid, 0.0)
                    z_score = abs(score - median_spectral) / (1.4826 * mad_spectral)
                    
                    if z_score > threshold:
                        outliers.append(cid)
                        outlier_info[cid] = f'spectral (z={z_score:.2f})'
        
        # Compute Modified Z-scores for DELTA norms
        delta_vals = np.array([delta_scores.get(cid, 0.0) for cid in all_client_ids])
        
        if len(delta_vals) > 0 and not np.all(np.isnan(delta_vals)):
            median_delta = np.median(delta_vals)
            mad_delta = np.median(np.abs(delta_vals - median_delta))
            
            if mad_delta == 0:
                # All scores identical - use absolute threshold
                for cid in candidate_clients:
                    score = delta_scores.get(cid, 0.0)
                    if not np.isclose(score, median_delta) and score > 1000 * max(abs(median_delta), 1.0):
                        if cid not in outliers:
                            outliers.append(cid)
                            outlier_info[cid] = 'delta (absolute)'
                        else:
                            outlier_info[cid] = 'both (absolute)'
            else:
                # Compute Modified Z-scores for candidates
                for cid in candidate_clients:
                    score = delta_scores.get(cid, 0.0)
                    z_score = abs(score - median_delta) / (1.4826 * mad_delta)
                    
                    if z_score > threshold:
                        if cid not in outliers:
                            outliers.append(cid)
                            outlier_info[cid] = f'delta (z={z_score:.2f})'
                        else:
                            # Already flagged by spectral
                            outlier_info[cid] = f'both (spectral + delta z={z_score:.2f})'
        
        # Log which signals triggered each outlier
        if outliers:
            for cid in outliers:
                trigger = outlier_info.get(cid, 'unknown')
                log(INFO, f"    Client {cid}: Outlier triggered by {trigger}")
        
        return outliers
    
    def _log_detection_results(
        self,
        spectral_scores: Dict[int, float],
        delta_scores: Dict[int, float],
        normalized_spectral: Dict[int, float],
        normalized_delta: Dict[int, float],
        combined_scores: Dict[int, float],
        malicious_clients: List[int],
        benign_clients: List[int]
    ):
        """Log detailed detection results."""
        log(INFO, f"═══ FeRA Detection Results (Round {self.current_round}) ═══")
        log(INFO, f"Total clients: {len(combined_scores)}")
        log(INFO, f"Flagged as malicious: {len(malicious_clients)}")
        log(INFO, f"Flagged as benign: {len(benign_clients)}")
        log(INFO, f"Detection strategy: Flag BOTTOM {self.top_k_percent:.0%} (low variance = backdoor consistency)")
        
        # Sort by combined score (ascending)
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1])
        
        log(INFO, f"\nClient scores (sorted by consistency, ascending - LOW=malicious):")
        log(INFO, f"{'Client':<8} {'Spectral':<10} {'Delta':<10} {'Norm_Spec':<12} {'Norm_Delta':<12} {'Combined':<10} {'Status':<10}")
        log(INFO, f"{'-'*80}")
        
        for cid, combined in sorted_results:
            spectral = spectral_scores.get(cid, 0.0)
            delta = delta_scores.get(cid, 0.0)
            norm_spec = normalized_spectral.get(cid, 0.0)
            norm_delta = normalized_delta.get(cid, 0.0)
            status = "MALICIOUS" if cid in malicious_clients else "benign"
            
            log(INFO, f"{cid:<8} {spectral:<10.4f} {delta:<10.4f} {norm_spec:<12.4f} {norm_delta:<12.4f} {combined:<10.4f} {status:<10}")
        
        log(INFO, f"═══════════════════════════════════════════════════")

