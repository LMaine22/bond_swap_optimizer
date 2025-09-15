# src/robustness_framework.py
"""
Diversity and Selection Framework for Bond Swap Optimization

Implements the comprehensive framework for:
1. Diversity-aware representative selection
2. Micro-menus for different risk appetites
3. Time-budget management
4. Audit and governance tracking
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    TARGET_BUY_BACK_YIELD,
    MAX_RECOVERY_PERIOD_MONTHS,
    DIVERSITY_MIN_JACCARD,
    DIVERSITY_K_RANGE,
    DIVERSITY_SILHOUETTE_SAMPLES,
    REVIEW_TIME_MODE,
    REVIEW_TIME_PRESETS,
    MICRO_MENU_SIZE,
    MICRO_MENU_PERCENTILE,
)
from src.genetic_algorithm import SwapMetrics, _compute_metrics


@dataclass
class ExecutiveMenuItem:
    """A selected item for the executive menu"""
    option_id: str
    mask: np.ndarray
    base_metrics: SwapMetrics
    cluster_id: Optional[int] = None
    diversity_score: float = 0.0
    hypervolume_contribution: float = 0.0
    is_corner: bool = False
    corner_type: Optional[str] = None


@dataclass
class MicroMenu:
    """A micro-menu for specific risk appetites"""
    menu_type: str  # "Conservative", "Fast-Recovery", "Max Pick-Up"
    items: List[ExecutiveMenuItem]
    description: str


class DiversityFramework:
    """Main framework class for diverse bond swap optimization"""
    
    def __init__(self, bond_candidates_df: pd.DataFrame):
        self.bond_candidates_df = bond_candidates_df
        self.full_pareto: List[Dict] = []
        self.executive_menu: List[ExecutiveMenuItem] = []
        self.micro_menus: List[MicroMenu] = []
        self.governance_stats: Dict[str, Any] = {}
        
    def create_executive_menu(self, pareto_set: List[Dict]) -> List[ExecutiveMenuItem]:
        """Create diversity-aware executive menu using k-medoids clustering"""
        if not pareto_set:
            return []
        
        print(f"\n--- Executive Menu Selection ---")
        print(f"Creating executive menu from {len(pareto_set)} Pareto options...")
        
        # Normalize objectives for clustering
        normalized_data = self._normalize_objectives(pareto_set)
        
        # Find optimal k using silhouette score
        optimal_k = self._find_optimal_k(normalized_data)
        print(f"Optimal k for clustering: {optimal_k}")
        
        # Perform k-medoids clustering
        cluster_centers, cluster_labels = self._k_medoids_clustering(normalized_data, optimal_k)
        
        # Select representatives from each cluster
        representatives = self._select_cluster_representatives(
            pareto_set, cluster_labels, cluster_centers, normalized_data
        )
        
        # Force-include corner solutions
        corner_items = self._force_include_corners(pareto_set)
        
        # Combine and enforce diversity
        executive_items = self._enforce_diversity(representatives, corner_items)
        
        # Apply hypervolume trimming if needed
        if len(executive_items) > REVIEW_TIME_PRESETS[REVIEW_TIME_MODE]["executive_max"]:
            executive_items = self._hypervolume_trim(executive_items)
        
        self.executive_menu = executive_items
        print(f"Executive menu created: {len(executive_items)} items")
        
        return executive_items
    
    def _normalize_objectives(self, pareto_set: List[Dict]) -> np.ndarray:
        """Normalize objectives to [0,1] range for fair clustering"""
        objectives = []
        for option in pareto_set:
            m = option["metrics"]
            # Normalize: delta_income↑, loss↓, recovery↓, sold_wavg↓
            # Convert to minimization: -delta_income, loss, recovery, sold_wavg
            obj = [-m.delta_income, m.loss, m.recovery_months, m.sold_wavg]
            objectives.append(obj)
        
        objectives = np.array(objectives)
        
        # Min-max normalization
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(objectives)
        
        return normalized
    
    def _find_optimal_k(self, normalized_data: np.ndarray) -> int:
        """Find optimal k using silhouette score"""
        min_k, max_k = DIVERSITY_K_RANGE
        best_k = min_k
        best_score = -1
        
        for k in range(min_k, min(max_k + 1, len(normalized_data))):
            if k >= len(normalized_data):
                break
                
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(normalized_data)
                
                if len(set(cluster_labels)) > 1:  # Need at least 2 clusters
                    score = silhouette_score(normalized_data, cluster_labels, 
                                           sample_size=min(DIVERSITY_SILHOUETTE_SAMPLES, len(normalized_data)))
                    if score > best_score:
                        best_score = score
                        best_k = k
            except:
                continue
        
        return best_k
    
    def _k_medoids_clustering(self, normalized_data: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform k-medoids clustering (using k-means as approximation)"""
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(normalized_data)
        cluster_centers = kmeans.cluster_centers_
        
        return cluster_centers, cluster_labels
    
    def _select_cluster_representatives(self, pareto_set: List[Dict], 
                                      cluster_labels: np.ndarray, 
                                      cluster_centers: np.ndarray,
                                      normalized_data: np.ndarray) -> List[ExecutiveMenuItem]:
        """Select best representative from each cluster"""
        representatives = []
        
        for cluster_id in range(len(cluster_centers)):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            
            # Find closest to cluster center
            cluster_data = normalized_data[cluster_indices]
            distances = np.linalg.norm(cluster_data - cluster_centers[cluster_id], axis=1)
            best_idx = cluster_indices[np.argmin(distances)]
            
            option = pareto_set[best_idx]
            mask = option["mask"]
            base_metrics = option["metrics"]
            
            item = ExecutiveMenuItem(
                option_id=f"EXEC_{len(representatives)+1:02d}",
                mask=mask,
                base_metrics=base_metrics,
                cluster_id=cluster_id
            )
            representatives.append(item)
        
        return representatives
    
    def _force_include_corners(self, pareto_set: List[Dict]) -> List[ExecutiveMenuItem]:
        """Force-include corner solutions (max Δ-income, min recovery, etc.)"""
        if not pareto_set:
            return []
        
        corners = []
        
        # Max delta income
        max_delta_idx = max(range(len(pareto_set)), 
                           key=lambda i: pareto_set[i]["metrics"].delta_income)
        corners.append(("max_delta_income", max_delta_idx))
        
        # Min recovery
        min_recovery_idx = min(range(len(pareto_set)), 
                              key=lambda i: pareto_set[i]["metrics"].recovery_months)
        corners.append(("min_recovery", min_recovery_idx))
        
        # Min loss
        min_loss_idx = min(range(len(pareto_set)), 
                          key=lambda i: pareto_set[i]["metrics"].loss)
        corners.append(("min_loss", min_loss_idx))
        
        # Min sold wavg
        min_sold_wavg_idx = min(range(len(pareto_set)), 
                               key=lambda i: pareto_set[i]["metrics"].sold_wavg)
        corners.append(("min_sold_wavg", min_sold_wavg_idx))
        
        corner_items = []
        for corner_type, idx in corners:
            option = pareto_set[idx]
            mask = option["mask"]
            base_metrics = option["metrics"]
            
            item = ExecutiveMenuItem(
                option_id=f"CORNER_{corner_type}",
                mask=mask,
                base_metrics=base_metrics,
                is_corner=True,
                corner_type=corner_type
            )
            corner_items.append(item)
        
        return corner_items
    
    def _enforce_diversity(self, representatives: List[ExecutiveMenuItem], 
                          corner_items: List[ExecutiveMenuItem]) -> List[ExecutiveMenuItem]:
        """Enforce Jaccard diversity between selected items"""
        all_items = representatives + corner_items
        diverse_items = []
        diverse_masks = []
        
        for item in all_items:
            if not diverse_masks:
                diverse_items.append(item)
                diverse_masks.append(item.mask)
                continue
            
            # Check minimum Jaccard distance
            min_jaccard = min(self._jaccard_distance(item.mask, mask) for mask in diverse_masks)
            
            if min_jaccard >= DIVERSITY_MIN_JACCARD:
                diverse_items.append(item)
                diverse_masks.append(item.mask)
        
        # Calculate diversity scores
        for i, item in enumerate(diverse_items):
            if i == 0:
                item.diversity_score = 1.0
            else:
                similarities = [self._jaccard_distance(item.mask, other.mask) 
                               for other in diverse_items[:i]]
                item.diversity_score = 1.0 - max(similarities) if similarities else 1.0
        
        return diverse_items
    
    def _jaccard_distance(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate Jaccard distance between two masks"""
        set1 = set(np.where(mask1)[0])
        set2 = set(np.where(mask2)[0])
        
        if not set1 and not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return 1.0 - (intersection / union) if union > 0 else 0.0
    
    def _hypervolume_trim(self, items: List[ExecutiveMenuItem]) -> List[ExecutiveMenuItem]:
        """Trim items using hypervolume contribution (simplified)"""
        # For now, just keep the first N items
        max_items = REVIEW_TIME_PRESETS[REVIEW_TIME_MODE]["executive_max"]
        return items[:max_items]
    
    def create_micro_menus(self, pareto_set: List[Dict]) -> List[MicroMenu]:
        """Create micro-menus for different risk appetites"""
        if not pareto_set:
            return []
        
        print(f"\n--- Micro-Menus ---")
        
        micro_menus = []
        
        # Conservative / Low-Loss
        conservative = self._create_conservative_menu(pareto_set)
        micro_menus.append(conservative)
        
        # Fast-Recovery
        fast_recovery = self._create_fast_recovery_menu(pareto_set)
        micro_menus.append(fast_recovery)
        
        # Max Pick-Up (Δ-income)
        max_pickup = self._create_max_pickup_menu(pareto_set)
        micro_menus.append(max_pickup)
        
        self.micro_menus = micro_menus
        return micro_menus
    
    def _create_conservative_menu(self, pareto_set: List[Dict]) -> MicroMenu:
        """Create conservative menu (low loss)"""
        # Sort by loss (ascending), then delta income (descending), then recovery (ascending)
        sorted_options = sorted(pareto_set, 
                              key=lambda x: (x["metrics"].loss, -x["metrics"].delta_income, x["metrics"].recovery_months))
        
        items = []
        for i, option in enumerate(sorted_options[:MICRO_MENU_SIZE]):
            mask = option["mask"]
            base_metrics = option["metrics"]
            
            item = ExecutiveMenuItem(
                option_id=f"CONS_{i+1:02d}",
                mask=mask,
                base_metrics=base_metrics
            )
            items.append(item)
        
        return MicroMenu(
            menu_type="Conservative",
            items=items,
            description="Low-loss options prioritizing capital preservation"
        )
    
    def _create_fast_recovery_menu(self, pareto_set: List[Dict]) -> MicroMenu:
        """Create fast recovery menu"""
        # Sort by recovery (ascending), then delta income (descending), then loss (ascending)
        sorted_options = sorted(pareto_set, 
                              key=lambda x: (x["metrics"].recovery_months, -x["metrics"].delta_income, x["metrics"].loss))
        
        items = []
        for i, option in enumerate(sorted_options[:MICRO_MENU_SIZE]):
            mask = option["mask"]
            base_metrics = option["metrics"]
            
            item = ExecutiveMenuItem(
                option_id=f"FAST_{i+1:02d}",
                mask=mask,
                base_metrics=base_metrics
            )
            items.append(item)
        
        return MicroMenu(
            menu_type="Fast-Recovery",
            items=items,
            description="Quick recovery options for rapid capital return"
        )
    
    def _create_max_pickup_menu(self, pareto_set: List[Dict]) -> MicroMenu:
        """Create max pickup menu (highest delta income)"""
        # Sort by delta income (descending), then recovery (ascending)
        sorted_options = sorted(pareto_set, 
                              key=lambda x: (-x["metrics"].delta_income, x["metrics"].recovery_months))
        
        items = []
        for i, option in enumerate(sorted_options[:MICRO_MENU_SIZE]):
            mask = option["mask"]
            base_metrics = option["metrics"]
            
            item = ExecutiveMenuItem(
                option_id=f"MAX_{i+1:02d}",
                mask=mask,
                base_metrics=base_metrics
            )
            items.append(item)
        
        return MicroMenu(
            menu_type="Max Pick-Up",
            items=items,
            description="Maximum income pickup options for highest yield enhancement"
        )
    
    def generate_audit_artifacts(self, output_dir: str) -> Dict[str, str]:
        """Generate comprehensive audit artifacts"""
        artifacts = {}
        
        # Full Pareto CSV
        if REVIEW_TIME_PRESETS[REVIEW_TIME_MODE]["full_pareto"]:
            full_pareto_path = self._save_full_pareto_csv(output_dir)
            artifacts["full_pareto"] = full_pareto_path
        
        # Executive Menu CSV
        executive_path = self._save_executive_menu_csv(output_dir)
        artifacts["executive_menu"] = executive_path
        
        # Micro-menus CSV
        micro_menus_path = self._save_micro_menus_csv(output_dir)
        artifacts["micro_menus"] = micro_menus_path
        
        # Manifest JSON
        manifest_path = self._save_manifest_json(output_dir)
        artifacts["manifest"] = manifest_path
        
        return artifacts
    
    def _save_full_pareto_csv(self, output_dir: str) -> str:
        """Save full Pareto set to CSV"""
        # Implementation for full Pareto CSV
        pass
    
    def _save_executive_menu_csv(self, output_dir: str) -> str:
        """Save executive menu to CSV"""
        # Implementation for executive menu CSV
        pass
    
    def _save_micro_menus_csv(self, output_dir: str) -> str:
        """Save micro-menus to CSV"""
        # Implementation for micro-menus CSV
        pass
    
    def _save_manifest_json(self, output_dir: str) -> str:
        """Save manifest JSON with all parameters and provenance"""
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "review_time_mode": REVIEW_TIME_MODE,
            "diversity_parameters": {
                "min_jaccard": DIVERSITY_MIN_JACCARD,
                "k_range": DIVERSITY_K_RANGE
            },
            "governance_stats": self.governance_stats
        }
        
        manifest_path = f"{output_dir}/manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest_path
