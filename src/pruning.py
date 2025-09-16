# src/pruning.py - Fixed version without .iloc issues
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List
import warnings

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

from src.config import (
    MIN_SWAP_SIZE_DOLLARS,
    MAX_SWAP_SIZE_DOLLARS,
    MAX_TOTAL_SWAP_LOSS_DOLLARS,
    MIN_RECOVERY_PERIOD_MONTHS,
    MAX_RECOVERY_PERIOD_MONTHS,
    SOLD_WAVG_PROJ_YIELD_MAX,
    ENFORCE_MIN_SWAP_LOSS,
    MIN_TOTAL_SWAP_LOSS_DOLLARS,
)


@dataclass
class SwapMetrics:
    par: float
    book: float
    market: float
    loss: float
    income: float
    delta_income: float
    sold_wavg: float
    proj_y: float
    recovery_months: float
    count: int


def _safe_sum(s) -> float:
    """Safely sum a series, handling NaN and type conversion"""
    return float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())


def compute_metrics(df: pd.DataFrame, mask: np.ndarray) -> SwapMetrics:
    """Compute comprehensive swap metrics for a bond selection"""
    # Convert mask to boolean array if needed
    if mask.dtype != bool:
        mask = mask.astype(bool)

    sel = df.loc[mask]
    if sel.empty:
        return SwapMetrics(0, 0, 0, 0, 0, 0, float("inf"), 0, float("inf"), 0)

    par = _safe_sum(sel.get("par", 0))
    book = _safe_sum(sel.get("book", 0))
    market = _safe_sum(sel.get("market", 0))
    loss = _safe_sum(sel.get("loss", 0))  # positive = loss
    income = _safe_sum(sel.get("income", 0))
    d_inc = _safe_sum(sel.get("delta_income", 0))

    sold_wavg = (income / par) if par > 0 else float("inf")
    rec_mo = (12.0 * loss / d_inc) if d_inc > 0 else float("inf")

    # Market value weighted projected yield
    if "proj_yield" in sel.columns and sel["proj_yield"].notna().any() and market > 0:
        proj_y_series = pd.to_numeric(sel["proj_yield"], errors="coerce")
        market_series = pd.to_numeric(sel["market"], errors="coerce")
        proj_y = float((proj_y_series * market_series).sum() / market)
    else:
        proj_y = (income / market) if market > 0 else 0.0

    return SwapMetrics(par, book, market, loss, income, d_inc, sold_wavg, proj_y, rec_mo, len(sel))


def feasible(m: SwapMetrics) -> bool:
    """Check if swap metrics meet all hard constraints"""
    # Size constraints
    if not (MIN_SWAP_SIZE_DOLLARS <= m.market <= MAX_SWAP_SIZE_DOLLARS):
        return False

    # Loss constraints
    if m.loss > MAX_TOTAL_SWAP_LOSS_DOLLARS:
        return False
    if ENFORCE_MIN_SWAP_LOSS and m.loss < MIN_TOTAL_SWAP_LOSS_DOLLARS:
        return False

    # Recovery period constraints (including scenario-specific minimums)
    if m.recovery_months < MIN_RECOVERY_PERIOD_MONTHS:
        return False
    if m.recovery_months > MAX_RECOVERY_PERIOD_MONTHS:
        return False

    # Yield constraint
    if m.sold_wavg > SOLD_WAVG_PROJ_YIELD_MAX:
        return False

    # Must have positive delta income
    if m.delta_income <= 0:
        return False

    return True


def prune_passengers(
        df: pd.DataFrame,
        mask_in: np.ndarray,
        *,
        abs_dinc_eps: float = 5_000.0,  # minimum absolute Δ-income drop that "earns a seat"
        rel_dinc_eps: float = 0.02,  # minimum relative Δ-income drop (2%) that "earns a seat"
        soldwavg_eps: float = 1e-9,  # allow no worsening of sold wavg (<= base + eps)
        recovery_eps: float = 1e-9,  # allow no worsening of recovery (<= base + eps)
) -> Tuple[np.ndarray, Dict]:
    """
    Iteratively remove "passenger" bonds that don't meaningfully contribute to the swap.

    A bond is considered a passenger if removing it:
    - Keeps the option feasible under all constraints
    - Doesn't worsen sold_wavg or recovery time (within eps tolerance)
    - Doesn't reduce Δ-income by more than the thresholds

    This is particularly important for scenario-based optimization where we want
    clean, focused swaps without unnecessary complexity.

    Returns:
        new_mask: Pruned mask with passengers removed
        audit: Dictionary with removal details and metrics
    """
    # Ensure we're working with boolean mask
    base_mask = mask_in.astype(bool).copy()
    base_m = compute_metrics(df, base_mask)

    removed: List[int] = []
    enablers: List[int] = []

    if base_m.count <= 1:  # Can't prune if only one bond
        return base_mask, {
            "base_metrics": asdict(base_m),
            "final_metrics": asdict(base_m),
            "removed_indices": removed,
            "kept_enablers": enablers,
            "pruning_summary": "No pruning possible with ≤1 bond"
        }

    changed = True
    current_mask = base_mask.copy()
    iteration = 0
    max_iterations = 20  # Prevent infinite loops

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        cur_m = compute_metrics(df, current_mask)

        # Get indices of currently selected bonds
        current_indices = np.where(current_mask)[0]
        passenger_candidates = []

        for idx in current_indices:
            # Test removal of this bond
            test_mask = current_mask.copy()
            test_mask[idx] = False

            m_wo = compute_metrics(df, test_mask)

            # Check if removal breaks feasibility - if so, it's an enabler
            if not feasible(m_wo):
                bond_idx = df.index[idx]
                if bond_idx not in enablers:
                    enablers.append(bond_idx)
                continue

            # Calculate impact of removal
            dinc_drop_abs = cur_m.delta_income - m_wo.delta_income
            dinc_drop_rel = dinc_drop_abs / max(cur_m.delta_income, 1e-9)

            # Check if other metrics don't worsen
            soldwavg_ok = (m_wo.sold_wavg <= (cur_m.sold_wavg + soldwavg_eps))
            recovery_ok = (m_wo.recovery_months <= (cur_m.recovery_months + recovery_eps))

            # Is this bond a passenger?
            is_passenger = (
                    (dinc_drop_abs < abs_dinc_eps) and
                    (dinc_drop_rel < rel_dinc_eps) and
                    soldwavg_ok and
                    recovery_ok
            )

            if is_passenger:
                passenger_candidates.append(idx)

        # Remove all passengers found in this iteration
        if passenger_candidates:
            for idx in passenger_candidates:
                current_mask[idx] = False
                bond_idx = df.index[idx]
                removed.append(bond_idx)
            changed = True

    final_m = compute_metrics(df, current_mask)

    # Safety check: ensure final solution is still feasible
    if not feasible(final_m):
        return base_mask, {
            "base_metrics": asdict(base_m),
            "final_metrics": asdict(base_m),
            "removed_indices": [],  # rollback
            "kept_enablers": enablers,
            "pruning_summary": "Rollback - final solution became infeasible"
        }

    # Build audit summary
    pruning_summary = f"Pruned {len(removed)} passengers in {iteration} iterations"
    if removed:
        delta_impact = base_m.delta_income - final_m.delta_income
        pruning_summary += f". Δ-income impact: ${delta_impact:,.0f}"

    return current_mask, {
        "base_metrics": asdict(base_m),
        "final_metrics": asdict(final_m),
        "removed_indices": removed,
        "kept_enablers": enablers,
        "pruning_summary": pruning_summary,
        "iterations": iteration,
        "bonds_removed": len(removed),
        "bonds_kept": final_m.count,
        "enabler_count": len(enablers)
    }