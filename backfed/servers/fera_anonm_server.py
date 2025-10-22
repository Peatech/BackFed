"""
FeRA_anonm (Feature Representation Anomaly - Anomaly Methods) Defense Server

This enhanced version of FeRA adds 5 intuition-based detection methods:
1. Parameter Inactivity Score - Unlearning resistance on clean data
2. Cross-Task Learning Speed - Fast convergence on synthetic backdoors
3. Decision Boundary Distance - Boundary proximity to target class
4. Prediction Stability Score - Robustness under noise
5. Combined Multi-Signal - Weighted fusion of all methods

Architecture:
- Phase 1: Original FeRA clustering (50% split)
- Phase 2: Multi-signal analysis on ALL clients in BOTH clusters
- Comprehensive telemetry with comparison tables and JSON export

Author: AI Assistant
Date: 2025-10-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from logging import INFO, WARNING
from concurrent.futures import ThreadPoolExecutor, as_completed

from backfed.servers.fera_server import FeRAServer
from backfed.utils.system_utils import log
from backfed.const import client_id, num_examples, StateDict


class FeRAAnomServer(FeRAServer):
    """
    Enhanced FeRA Defense with 5 intuition-based detection methods.
    
    Extends FeRAServer with additional behavioral analysis methods that
    test all clients after initial clustering to provide deeper insights
    into malicious behavior patterns.
    """
    
    defense_categories = ["anomaly_detection"]
    
    def __init__(
        self,
        server_config,
        server_type: str = "fera_anonm",
        eta: float = 0.5,
        **kwargs
    ):
        """
        Initialize FeRA_anonm server with all original FeRA parameters
        plus additional method-specific parameters.
        """
        # Initialize parent FeRA server
        super().__init__(server_config, server_type=server_type, eta=eta, **kwargs)
        
        # Method enable flags
        self.enable_method_1 = kwargs.get('enable_method_1', True)
        self.enable_method_2 = kwargs.get('enable_method_2', True)
        self.enable_method_3 = kwargs.get('enable_method_3', True)
        self.enable_method_4 = kwargs.get('enable_method_4', True)
        self.enable_method_5 = kwargs.get('enable_method_5', True)
        self.use_combined_only = kwargs.get('use_combined_only', False)
        
        # Method 1: Unlearning parameters
        self.unlearning_epochs = kwargs.get('unlearning_epochs', 5)
        self.unlearning_lr = kwargs.get('unlearning_lr', 0.01)
        self.unlearning_batch_size = kwargs.get('unlearning_batch_size', 64)
        
        # Method 2: Learning speed parameters
        self.learning_speed_iters = kwargs.get('learning_speed_iters', 20)
        self.learning_speed_lr = kwargs.get('learning_speed_lr', 0.01)
        self.learning_speed_synthetic_trigger = kwargs.get('learning_speed_synthetic_trigger', 'random')
        
        # Method 3: Boundary distance parameters
        self.boundary_attack_steps = kwargs.get('boundary_attack_steps', 10)
        self.boundary_epsilon = kwargs.get('boundary_epsilon', 0.1)
        self.boundary_attack_type = kwargs.get('boundary_attack_type', 'fgsm')
        
        # Method 4: Stability parameters
        self.stability_noise_levels = kwargs.get('stability_noise_levels', [0.01, 0.05, 0.1])
        self.stability_samples = kwargs.get('stability_samples', 100)
        self.stability_trigger_types = kwargs.get('stability_trigger_types', ['pattern', 'pixel'])
        
        # Method 5: Combined weights
        self.combined_weight_unlearning = kwargs.get('combined_weight_unlearning', 0.35)
        self.combined_weight_speed = kwargs.get('combined_weight_speed', 0.30)
        self.combined_weight_boundary = kwargs.get('combined_weight_boundary', 0.20)
        self.combined_weight_stability = kwargs.get('combined_weight_stability', 0.15)
        
        # Telemetry options
        self.save_detailed_json = kwargs.get('save_detailed_json', True)
        self.json_output_dir = kwargs.get('json_output_dir', 'outputs/fera_anonm_analysis')
        self.print_comparison_table = kwargs.get('print_comparison_table', True)
        self.print_per_method_tables = kwargs.get('print_per_method_tables', True)
        
        # Create output directory
        if self.save_detailed_json:
            Path(self.json_output_dir).mkdir(parents=True, exist_ok=True)
        
        log(INFO, "═══ Initialized FeRA_anonm Defense ═══")
        log(INFO, f"  Method 1 (Unlearning): {'Enabled' if self.enable_method_1 else 'Disabled'}")
        log(INFO, f"  Method 2 (Learning Speed): {'Enabled' if self.enable_method_2 else 'Disabled'}")
        log(INFO, f"  Method 3 (Boundary Distance): {'Enabled' if self.enable_method_3 else 'Disabled'}")
        log(INFO, f"  Method 4 (Stability): {'Enabled' if self.enable_method_4 else 'Disabled'}")
        log(INFO, f"  Method 5 (Combined): {'Enabled' if self.enable_method_5 else 'Disabled'}")
        log(INFO, f"  Use combined only: {self.use_combined_only}")
        log(INFO, "═════════════════════════════════════")
    
    def _detect_malicious_clients(
        self,
        client_updates: List[Tuple[StateDict, int]],
        round_number: int
    ) -> Tuple[List[int], List[int]]:
        """
        Enhanced detection with two phases:
        1. Original FeRA clustering
        2. Multi-signal analysis on all clients
        
        Returns:
            Tuple of (predicted_malicious, predicted_benign) client IDs
        """
        # Phase 1: Run original FeRA detection
        log(INFO, "═══ Phase 1: Original FeRA Clustering ═══")
        fera_malicious, fera_benign = super()._detect_malicious_clients(
            client_updates, round_number
        )
        
        log(INFO, f"  Suspected malicious cluster (bottom 50%): {sorted(fera_malicious)}")
        log(INFO, f"  Suspected benign cluster (top 50%): {sorted(fera_benign)}")
        
        # Phase 2: Multi-signal analysis
        log(INFO, "═══ Phase 2: Multi-Signal Analysis ═══")
        all_client_ids = [cid for _, cid in client_updates]
        
        # Run all enabled methods
        method_scores = self._run_multi_signal_analysis(
            client_updates, round_number
        )
        
        # Generate telemetry
        if self.print_per_method_tables or self.print_comparison_table:
            self._output_telemetry(
                method_scores, fera_malicious, fera_benign,
                all_client_ids, round_number
            )
        
        # Export JSON
        if self.save_detailed_json:
            self._export_json(
                method_scores, fera_malicious, fera_benign,
                all_client_ids, round_number
            )
        
        # Determine final predictions
        final_malicious, final_benign = self._finalize_predictions(
            method_scores, fera_malicious, fera_benign, all_client_ids
        )
        
        return final_malicious, final_benign
    
    def _run_multi_signal_analysis(
        self,
        client_updates: List[Tuple[StateDict, int]],
        round_number: int
    ) -> Dict[str, Dict[int, float]]:
        """
        Run all enabled detection methods in parallel.
        
        Returns:
            Dictionary mapping method names to client scores
        """
        method_scores = {}
        
        # Method 1: Unlearning Resistance
        if self.enable_method_1:
            log(INFO, "  Computing Method 1: Unlearning Resistance...")
            method_scores['unlearning'] = self._compute_unlearning_score(client_updates)
        
        # Method 2: Learning Speed
        if self.enable_method_2:
            log(INFO, "  Computing Method 2: Cross-Task Learning Speed...")
            method_scores['learning_speed'] = self._compute_learning_speed_score(client_updates)
        
        # Method 3: Boundary Distance
        if self.enable_method_3:
            log(INFO, "  Computing Method 3: Decision Boundary Distance...")
            method_scores['boundary'] = self._compute_boundary_distance_score(client_updates)
        
        # Method 4: Stability
        if self.enable_method_4:
            log(INFO, "  Computing Method 4: Prediction Stability...")
            method_scores['stability'] = self._compute_stability_score(client_updates)
        
        # Method 5: Combined
        if self.enable_method_5 and len(method_scores) > 0:
            log(INFO, "  Computing Method 5: Combined Multi-Signal...")
            method_scores['combined'] = self._compute_combined_score(method_scores)
        
        return method_scores
    
    def _compute_unlearning_score(
        self,
        client_updates: List[Tuple[StateDict, int]]
    ) -> Dict[int, float]:
        """
        Method 1: Parameter Inactivity Score (Unlearning Resistance)
        
        Intuition: Backdoor parameters are "sticky" - they resist unlearning on clean data.
        
        For each client update:
        1. Apply update to global model (temp copy)
        2. Fine-tune on root dataset (clean) for N epochs
        3. Measure loss improvement: backdoor updates show less improvement
        4. Score = 1 - (loss_improvement / max_improvement)
        
        Returns:
            Dict[client_id, score] where higher score = more malicious
        """
        scores = {}
        
        # Get root dataset loader
        root_loader = self._get_root_dataloader(self.unlearning_batch_size)
        
        # Compute baseline: fine-tune global model
        baseline_improvement = self._finetune_and_measure(
            self.global_model, root_loader, self.unlearning_epochs, self.unlearning_lr
        )
        
        for update, cid in client_updates:
            # Create temp model with client update
            temp_model = self._apply_client_update_temp(update)
            
            # Fine-tune and measure improvement
            improvement = self._finetune_and_measure(
                temp_model, root_loader, self.unlearning_epochs, self.unlearning_lr
            )
            
            # Score: lower improvement = more resistant = more malicious
            if baseline_improvement > 0:
                score = 1.0 - (improvement / (baseline_improvement + 1e-10))
            else:
                score = 0.5  # Neutral score if no baseline
            
            scores[cid] = max(0.0, min(1.0, score))  # Clip to [0,1]
        
        # Normalize scores
        scores = self._normalize_scores_robust(scores)
        
        return scores
    
    def _compute_learning_speed_score(
        self,
        client_updates: List[Tuple[StateDict, int]]
    ) -> Dict[int, float]:
        """
        Method 2: Cross-Task Learning Speed (Fast Convergence)
        
        Intuition: Backdoored models learn new backdoors faster (feature hijacking).
        
        For each client update:
        1. Apply update to global model (temp copy)
        2. Create synthetic backdoor task (random pattern -> class 0)
        3. Fine-tune for N iterations
        4. Measure convergence rate (loss decrease per iteration)
        5. Score = normalized_convergence_rate
        
        Returns:
            Dict[client_id, score] where higher score = faster convergence = more malicious
        """
        scores = {}
        
        # Create synthetic backdoor dataset
        synthetic_loader = self._create_synthetic_backdoor_loader()
        
        for update, cid in client_updates:
            # Create temp model with client update
            temp_model = self._apply_client_update_temp(update)
            
            # Measure convergence speed
            convergence_rate = self._measure_convergence_speed(
                temp_model, synthetic_loader, self.learning_speed_iters, self.learning_speed_lr
            )
            
            scores[cid] = convergence_rate
        
        # Normalize scores
        scores = self._normalize_scores_robust(scores)
        
        return scores
    
    def _compute_boundary_distance_score(
        self,
        client_updates: List[Tuple[StateDict, int]]
    ) -> Dict[int, float]:
        """
        Method 3: Decision Boundary Distance (Boundary Proximity)
        
        Intuition: Backdoor clients push decision boundaries closer to trigger space.
        
        For each client update:
        1. Apply update to global model (temp copy)
        2. Sample from root dataset (non-target classes)
        3. Use FGSM/PGD to find minimal perturbation to target class
        4. Measure average perturbation magnitude (L2 norm)
        5. Score = 1 / (perturbation_magnitude + epsilon)
        
        Returns:
            Dict[client_id, score] where higher score = closer boundary = more malicious
        """
        scores = {}
        
        # Get root dataset loader
        root_loader = self._get_root_dataloader(batch_size=32)
        
        # Get target class from attack config
        target_class = self.server_config.atk_config.get('target_class', 0)
        
        for update, cid in client_updates:
            # Create temp model with client update
            temp_model = self._apply_client_update_temp(update)
            
            # Measure boundary distance
            avg_perturbation = self._measure_boundary_distance(
                temp_model, root_loader, target_class,
                self.boundary_attack_steps, self.boundary_epsilon
            )
            
            # Score: smaller perturbation = closer boundary = more malicious
            score = 1.0 / (avg_perturbation + 1e-3)
            scores[cid] = score
        
        # Normalize scores
        scores = self._normalize_scores_robust(scores)
        
        return scores
    
    def _compute_stability_score(
        self,
        client_updates: List[Tuple[StateDict, int]]
    ) -> Dict[int, float]:
        """
        Method 4: Prediction Stability Score (Robustness)
        
        Intuition: Backdoor predictions are brittle under noise.
        
        For each client update:
        1. Apply update to global model (temp copy)
        2. Create triggered samples (apply known patterns)
        3. Add Gaussian noise at multiple levels
        4. Measure prediction consistency (entropy, agreement)
        5. Score = 1 - normalized_stability
        
        Returns:
            Dict[client_id, score] where higher score = more unstable = more malicious
        """
        scores = {}
        
        # Create triggered samples
        triggered_samples, target_labels = self._create_triggered_samples(
            self.stability_samples
        )
        
        for update, cid in client_updates:
            # Create temp model with client update
            temp_model = self._apply_client_update_temp(update)
            
            # Measure stability under noise
            stability = self._measure_prediction_stability(
                temp_model, triggered_samples, target_labels, self.stability_noise_levels
            )
            
            # Score: lower stability = more brittle = more malicious
            score = 1.0 - stability
            scores[cid] = max(0.0, min(1.0, score))
        
        # Normalize scores
        scores = self._normalize_scores_robust(scores)
        
        return scores
    
    def _compute_combined_score(
        self,
        method_scores: Dict[str, Dict[int, float]]
    ) -> Dict[int, float]:
        """
        Method 5: Combined Multi-Signal Score
        
        Weighted fusion of all available methods.
        
        Returns:
            Dict[client_id, combined_score]
        """
        combined = {}
        
        # Get all client IDs
        all_clients = set()
        for scores in method_scores.values():
            if scores:
                all_clients.update(scores.keys())
        
        # Combine scores with weights
        for cid in all_clients:
            total_weight = 0.0
            weighted_sum = 0.0
            
            if 'unlearning' in method_scores:
                weighted_sum += self.combined_weight_unlearning * method_scores['unlearning'].get(cid, 0.5)
                total_weight += self.combined_weight_unlearning
            
            if 'learning_speed' in method_scores:
                weighted_sum += self.combined_weight_speed * method_scores['learning_speed'].get(cid, 0.5)
                total_weight += self.combined_weight_speed
            
            if 'boundary' in method_scores:
                weighted_sum += self.combined_weight_boundary * method_scores['boundary'].get(cid, 0.5)
                total_weight += self.combined_weight_boundary
            
            if 'stability' in method_scores:
                weighted_sum += self.combined_weight_stability * method_scores['stability'].get(cid, 0.5)
                total_weight += self.combined_weight_stability
            
            combined[cid] = weighted_sum / (total_weight + 1e-10)
        
        return combined
    
    # ========== Helper Methods ==========
    
    def _apply_client_update_temp(self, client_update: StateDict) -> nn.Module:
        """Create temporary model with client update applied."""
        temp_model = copy.deepcopy(self.global_model)
        temp_model.load_state_dict(client_update)
        temp_model.to(self.device)
        temp_model.eval()
        return temp_model
    
    def _finetune_and_measure(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
        lr: float
    ) -> float:
        """
        Fine-tune model and measure loss improvement.
        
        Returns:
            Loss improvement (initial_loss - final_loss)
        """
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Measure initial loss
        initial_loss = self._evaluate_loss(model, dataloader, criterion)
        
        # Fine-tune
        for epoch in range(epochs):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Measure final loss
        final_loss = self._evaluate_loss(model, dataloader, criterion)
        
        improvement = initial_loss - final_loss
        return max(0.0, improvement)
    
    def _evaluate_loss(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> float:
        """Evaluate average loss on dataloader."""
        model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                count += 1
        
        return total_loss / (count + 1e-10)
    
    def _create_synthetic_backdoor_loader(self) -> torch.utils.data.DataLoader:
        """
        Create synthetic backdoor dataset for learning speed test.
        
        Uses random pattern or checkerboard pattern as trigger.
        """
        # Get a batch from root dataset
        root_loader = self._get_root_dataloader(batch_size=128)
        inputs, labels = next(iter(root_loader))
        
        # Apply synthetic trigger (random noise pattern)
        trigger_size = (1, inputs.shape[2] // 4, inputs.shape[3] // 4)
        trigger = torch.randn(trigger_size)
        
        # Add trigger to bottom-right corner
        triggered_inputs = inputs.clone()
        triggered_inputs[:, :, -trigger_size[1]:, -trigger_size[2]:] = trigger
        
        # Map all to class 0
        triggered_labels = torch.zeros(len(labels), dtype=torch.long)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(triggered_inputs, triggered_labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        return loader
    
    def _measure_convergence_speed(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        iterations: int,
        lr: float
    ) -> float:
        """
        Measure how quickly model learns synthetic backdoor.
        
        Returns:
            Convergence rate (average loss decrease per iteration)
        """
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        
        for i in range(iterations):
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                break  # One batch per iteration
        
        # Compute convergence rate
        if len(losses) > 1:
            initial_loss = np.mean(losses[:3])
            final_loss = np.mean(losses[-3:])
            convergence_rate = (initial_loss - final_loss) / len(losses)
            return max(0.0, convergence_rate)
        
        return 0.0
    
    def _measure_boundary_distance(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        target_class: int,
        steps: int,
        epsilon: float
    ) -> float:
        """
        Measure average perturbation needed to flip prediction to target class.
        
        Uses FGSM-style adversarial attack.
        
        Returns:
            Average L2 norm of perturbations
        """
        model.eval()
        perturbations = []
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Only use samples not already in target class
            mask = labels != target_class
            if mask.sum() == 0:
                continue
            
            inputs = inputs[mask]
            target_labels = torch.full((len(inputs),), target_class, dtype=torch.long).to(self.device)
            
            # FGSM attack
            inputs.requires_grad = True
            
            for step in range(steps):
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, target_labels)
                model.zero_grad()
                loss.backward()
                
                # Gradient step
                with torch.no_grad():
                    perturbation = epsilon * inputs.grad.sign()
                    inputs = inputs + perturbation
                    inputs.requires_grad = True
                
                # Check if flipped
                with torch.no_grad():
                    preds = model(inputs).argmax(dim=1)
                    if (preds == target_labels).all():
                        break
            
            # Measure perturbation magnitude
            with torch.no_grad():
                pert_norm = torch.norm(perturbation.view(len(inputs), -1), p=2, dim=1).mean().item()
                perturbations.append(pert_norm)
            
            break  # One batch is enough
        
        return np.mean(perturbations) if perturbations else 1.0
    
    def _create_triggered_samples(
        self,
        num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create samples with known trigger patterns.
        
        Returns:
            (triggered_inputs, target_labels)
        """
        # Get samples from root dataset
        root_loader = self._get_root_dataloader(batch_size=num_samples)
        inputs, _ = next(iter(root_loader))
        
        # Apply pattern trigger (white square in bottom-right)
        trigger_size = 3
        triggered_inputs = inputs.clone()
        triggered_inputs[:, :, -trigger_size:, -trigger_size:] = 1.0
        
        # Target class
        target_class = self.server_config.atk_config.get('target_class', 0)
        target_labels = torch.full((len(inputs),), target_class, dtype=torch.long)
        
        return triggered_inputs, target_labels
    
    def _measure_prediction_stability(
        self,
        model: nn.Module,
        triggered_samples: torch.Tensor,
        target_labels: torch.Tensor,
        noise_levels: List[float]
    ) -> float:
        """
        Measure prediction consistency under noise.
        
        Returns:
            Stability score (0-1, higher = more stable)
        """
        model.eval()
        triggered_samples = triggered_samples.to(self.device)
        target_labels = target_labels.to(self.device)
        
        agreements = []
        
        with torch.no_grad():
            # Original predictions
            original_preds = model(triggered_samples).argmax(dim=1)
            
            # Test under different noise levels
            for noise_level in noise_levels:
                noise = torch.randn_like(triggered_samples) * noise_level
                noisy_samples = triggered_samples + noise
                noisy_preds = model(noisy_samples).argmax(dim=1)
                
                # Measure agreement with original
                agreement = (noisy_preds == original_preds).float().mean().item()
                agreements.append(agreement)
        
        # Average stability across noise levels
        stability = np.mean(agreements)
        return stability
    
    def _normalize_scores_robust(self, scores: Dict[int, float]) -> Dict[int, float]:
        """
        Normalize scores to [0,1] using robust statistics (median + IQR).
        """
        if not scores:
            return scores
        
        values = list(scores.values())
        
        if len(values) < 2:
            return {k: 0.5 for k in scores.keys()}
        
        median = np.median(values)
        q25, q75 = np.percentile(values, [25, 75])
        iqr = q75 - q25
        
        if iqr < 1e-10:
            return {k: 0.5 for k in scores.keys()}
        
        normalized = {}
        for cid, score in scores.items():
            # Clip outliers
            clipped = np.clip(score, q25 - 1.5 * iqr, q75 + 1.5 * iqr)
            # Normalize
            norm_score = (clipped - q25) / (iqr + 1e-10)
            normalized[cid] = max(0.0, min(1.0, norm_score))
        
        return normalized
    
    def _output_telemetry(
        self,
        method_scores: Dict[str, Dict[int, float]],
        fera_malicious: List[int],
        fera_benign: List[int],
        all_clients: List[int],
        round_number: int
    ):
        """
        Print comprehensive telemetry with comparison tables.
        """
        log(INFO, "")
        log(INFO, f"═══ FeRA_anonm Multi-Signal Detection (Round {round_number}) ═══")
        log(INFO, "")
        log(INFO, "[Phase 1] Original FeRA Clustering:")
        log(INFO, f"  Suspected Malicious Cluster (bottom 50%): {sorted(fera_malicious)}")
        log(INFO, f"  Suspected Benign Cluster (top 50%): {sorted(fera_benign)}")
        log(INFO, "")
        
        if self.print_per_method_tables:
            log(INFO, "[Phase 2] Multi-Signal Analysis:")
            log(INFO, "")
            
            # Method 1
            if 'unlearning' in method_scores:
                self._print_method_table("Method 1: Parameter Inactivity (Unlearning Resistance)", 
                                        method_scores['unlearning'], all_clients)
            
            # Method 2
            if 'learning_speed' in method_scores:
                self._print_method_table("Method 2: Cross-Task Learning Speed", 
                                        method_scores['learning_speed'], all_clients)
            
            # Method 3
            if 'boundary' in method_scores:
                self._print_method_table("Method 3: Decision Boundary Distance", 
                                        method_scores['boundary'], all_clients)
            
            # Method 4
            if 'stability' in method_scores:
                self._print_method_table("Method 4: Prediction Stability", 
                                        method_scores['stability'], all_clients)
            
            # Method 5 - Combined
            if 'combined' in method_scores:
                self._print_combined_table(method_scores, all_clients)
        
        if self.print_comparison_table:
            self._print_comparison_matrix(method_scores, fera_malicious, all_clients)
        
        log(INFO, "═════════════════════════════════════════════════")
        log(INFO, "")
    
    def _print_method_table(
        self,
        method_name: str,
        scores: Dict[int, float],
        all_clients: List[int]
    ):
        """Print table for a single method."""
        log(INFO, method_name)
        log(INFO, "  Client   Norm_Score   Rank   Status")
        log(INFO, "  -------  -----------  -----  --------")
        
        # Sort by score (descending)
        sorted_clients = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (cid, score) in enumerate(sorted_clients[:10], 1):  # Top 10
            status = "MALICIOUS ⚠️" if score > 0.5 else "benign"
            log(INFO, f"  {cid:<7}  {score:.4f}       {rank:<5}  {status}")
        
        log(INFO, "")
    
    def _print_combined_table(
        self,
        method_scores: Dict[str, Dict[int, float]],
        all_clients: List[int]
    ):
        """Print combined scores with breakdown."""
        log(INFO, "Method 5: Combined Multi-Signal")
        log(INFO, "  Client   Unlearn  Speed    Boundary Stability Combined  Rank  Status")
        log(INFO, "  -------  -------  -------  -------- ---------  --------  ----  ------")
        
        combined = method_scores.get('combined', {})
        sorted_clients = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (cid, combined_score) in enumerate(sorted_clients[:10], 1):
            unlearn = method_scores.get('unlearning', {}).get(cid, 0.0)
            speed = method_scores.get('learning_speed', {}).get(cid, 0.0)
            boundary = method_scores.get('boundary', {}).get(cid, 0.0)
            stability = method_scores.get('stability', {}).get(cid, 0.0)
            
            # Status indicators
            indicators = sum([unlearn > 0.5, speed > 0.5, boundary > 0.5, stability > 0.5])
            status = "MALICIOUS " + "⚠️" * min(indicators, 3) if combined_score > 0.5 else "benign"
            
            log(INFO, f"  {cid:<7}  {unlearn:.2f}     {speed:.2f}     {boundary:.2f}     {stability:.2f}       {combined_score:.2f}      {rank:<4}  {status}")
        
        log(INFO, "")
    
    def _print_comparison_matrix(
        self,
        method_scores: Dict[str, Dict[int, float]],
        fera_malicious: List[int],
        all_clients: List[int]
    ):
        """Print method agreement matrix."""
        log(INFO, "[Comparison Matrix]")
        log(INFO, "Method Agreement Analysis:")
        log(INFO, "  Client   FeRA  Method1  Method2  Method3  Method4  Combined  Consensus")
        log(INFO, "  -------  ----  -------  -------  -------  -------  --------  ---------")
        
        for cid in sorted(all_clients)[:15]:  # Top 15 clients
            fera_flag = "MAL" if cid in fera_malicious else "BEN"
            
            m1_flag = "MAL" if method_scores.get('unlearning', {}).get(cid, 0.0) > 0.5 else "BEN"
            m2_flag = "MAL" if method_scores.get('learning_speed', {}).get(cid, 0.0) > 0.5 else "BEN"
            m3_flag = "MAL" if method_scores.get('boundary', {}).get(cid, 0.0) > 0.5 else "BEN"
            m4_flag = "MAL" if method_scores.get('stability', {}).get(cid, 0.0) > 0.5 else "BEN"
            combined_flag = "MAL" if method_scores.get('combined', {}).get(cid, 0.0) > 0.5 else "BEN"
            
            # Count consensus
            mal_count = [fera_flag, m1_flag, m2_flag, m3_flag, m4_flag, combined_flag].count("MAL")
            consensus = f"{mal_count}/6"
            if mal_count >= 5:
                consensus += " ✓✓✓"
            elif mal_count >= 4:
                consensus += " ✓✓"
            elif mal_count >= 3:
                consensus += " ✓"
            
            log(INFO, f"  {cid:<7}  {fera_flag}   {m1_flag}      {m2_flag}      {m3_flag}      {m4_flag}      {combined_flag}       {consensus}")
        
        log(INFO, "")
        
        # High confidence detections
        high_confidence = []
        for cid in all_clients:
            mal_count = 0
            if cid in fera_malicious:
                mal_count += 1
            for method in ['unlearning', 'learning_speed', 'boundary', 'stability', 'combined']:
                if method_scores.get(method, {}).get(cid, 0.0) > 0.5:
                    mal_count += 1
            
            if mal_count >= 4:
                high_confidence.append((cid, mal_count))
        
        high_confidence.sort(key=lambda x: x[1], reverse=True)
        log(INFO, f"[High-Confidence Detections]")
        log(INFO, f"Flagged by 4+ methods: {[cid for cid, _ in high_confidence]}")
        log(INFO, "")
    
    def _export_json(
        self,
        method_scores: Dict[str, Dict[int, float]],
        fera_malicious: List[int],
        fera_benign: List[int],
        all_clients: List[int],
        round_number: int
    ):
        """Export detailed JSON with all scores and predictions."""
        output = {
            "round": round_number,
            "initial_clustering": {
                "suspected_malicious": sorted(fera_malicious),
                "suspected_benign": sorted(fera_benign)
            },
            "method_scores": {},
            "predictions": {},
            "agreement_matrix": {},
            "ground_truth": sorted(self.server_config.client_manager.get_malicious_clients()),
        }
        
        # Add method scores with rankings
        for method_name, scores in method_scores.items():
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            output["method_scores"][method_name] = {
                str(cid): {
                    "score": float(score),
                    "rank": rank + 1
                }
                for rank, (cid, score) in enumerate(sorted_scores)
            }
        
        # Add predictions
        output["predictions"]["fera_original"] = sorted(fera_malicious)
        for method_name, scores in method_scores.items():
            predicted = [cid for cid, score in scores.items() if score > 0.5]
            output["predictions"][f"method_{method_name}"] = sorted(predicted)
        
        # Add agreement matrix
        for cid in all_clients:
            mal_count = 0
            if cid in fera_malicious:
                mal_count += 1
            for scores in method_scores.values():
                if scores.get(cid, 0.0) > 0.5:
                    mal_count += 1
            
            consensus = "high" if mal_count >= 5 else "medium" if mal_count >= 3 else "low"
            output["agreement_matrix"][str(cid)] = {
                "methods_flagging": mal_count,
                "consensus": consensus
            }
        
        # Save JSON
        json_path = Path(self.json_output_dir) / f"fera_anonm_round_{round_number}.json"
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        log(INFO, f"Detailed scores exported to: {json_path}")
    
    def _finalize_predictions(
        self,
        method_scores: Dict[str, Dict[int, float]],
        fera_malicious: List[int],
        fera_benign: List[int],
        all_clients: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Determine final predictions based on configuration.
        
        If use_combined_only=True, use only combined scores.
        Otherwise, use original FeRA predictions.
        """
        if self.use_combined_only and 'combined' in method_scores:
            # Use combined scores
            combined = method_scores['combined']
            predicted_malicious = [cid for cid in all_clients if combined.get(cid, 0.0) > 0.5]
            predicted_benign = [cid for cid in all_clients if cid not in predicted_malicious]
        else:
            # Use original FeRA predictions
            predicted_malicious = fera_malicious
            predicted_benign = fera_benign
        
        return predicted_malicious, predicted_benign

