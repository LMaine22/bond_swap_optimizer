# src/pruning.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

from src.config import (
    MIN_SWAP_SIZE_DOLLARS,
    MAX_SWAP_SIZE_DOLLARS,
    MAX_TOTAL_SWAP_LOSS_DOLLARS,
    MAX_RECOVERY_PERIOD_MONTHS,
    SOLD_WAVG_PROJ_YIELD_MAX,
)

# -------------------- metrics & feasibility --------------------

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
    return float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())

def compute_metrics(df: pd.DataFrame, mask: np.ndarray) -> SwapMetrics:
    sel = df.loc[mask]
    if sel.empty:
        return SwapMetrics(0,0,0,0,0,0,float("inf"),0,float("inf"),0)

    par    = _safe_sum(sel.get("par", 0))
    book   = _safe_sum(sel.get("book", 0))
    market = _safe_sum(sel.get("market", 0))
    loss   = _safe_sum(sel.get("loss", 0))            # positive = loss
    income = _safe_sum(sel.get("income", 0))
    d_inc  = _safe_sum(sel.get("delta_income", 0))

    sold_wavg = (income / par) if par > 0 else float("inf")
    rec_mo    = (12.0 * loss / d_inc) if d_inc > 0 else float("inf")

    if "proj_yield" in sel.columns and sel["proj_yield"].notna().any() and market > 0:
        proj_y = float((pd.to_numeric(sel["proj_yield"], errors="coerce") * pd.to_numeric(sel["market"], errors="coerce")).sum() / market)
    else:
        proj_y = (income / market) if market > 0 else 0.0

    return SwapMetrics(par, book, market, loss, income, d_inc, sold_wavg, proj_y, rec_mo, len(sel))

def feasible(m: SwapMetrics) -> bool:
    if not (MIN_SWAP_SIZE_DOLLARS <= m.market <= MAX_SWAP_SIZE_DOLLARS):
        return False
    if m.loss > MAX_TOTAL_SWAP_LOSS_DOLLARS:
        return False
    if m.recovery_months > MAX_RECOVERY_PERIOD_MONTHS:
        return False
    if m.sold_wavg > SOLD_WAVG_PROJ_YIELD_MAX:
        return False
    if m.delta_income <= 0:
        return False
    return True

# -------------------- pruning --------------------

def prune_passengers(
    df: pd.DataFrame,
    mask_in: np.ndarray,
    *,
    abs_dinc_eps: float = 5_000.0,     # minimum absolute Δ-income drop that "earns a seat"
    rel_dinc_eps: float = 0.02,        # minimum relative Δ-income drop (2%) that "earns a seat"
    soldwavg_eps: float = 1e-9,        # allow no worsening of sold wavg (<= base + eps)
    recovery_eps: float = 1e-9,        # allow no worsening of recovery (<= base + eps)
) -> Tuple[np.ndarray, Dict]:
    """
    Iteratively remove "passenger" bonds: names whose removal
    (a) keeps the option feasible, and
    (b) does NOT worsen sold_wavg nor recovery (within eps),
    and (c) does NOT reduce Δ-income by at least abs/rel thresholds.

    Returns new_mask and an audit dict with removed/enablers lists + pre/post metrics.
    """
    base_mask = mask_in.astype(bool).copy()
    base_m = compute_metrics(df, base_mask)

    removed: List[int] = []
    enablers: List[int] = []

    if base_m.count <= 1:  # nothing to prune
        return base_mask, {
            "base_metrics": asdict(base_m),
            "final_metrics": asdict(base_m),
            "removed_indices": removed,
            "kept_enablers": enablers,
        }

    changed = True
    current_mask = base_mask.copy()
    while changed:
        changed = False
        cur_m = compute_metrics(df, current_mask)

        # iterate over a stable list of indices to test
        current_idxs = list(df.index[current_mask])
        passenger_candidates = []

        for idx in current_idxs:
            test_mask = current_mask.copy()
            test_mask[df.index.get_loc(idx)] = False  # turn off this bond

            m_wo = compute_metrics(df, test_mask)

            # If removal breaks feasibility -> it's an "enabler" (kept)
            if not feasible(m_wo):
                if idx not in enablers:
                    enablers.append(idx)
                continue

            # Marginal tests vs current option (not the original base)
            dinc_drop_abs = cur_m.delta_income - m_wo.delta_income
            dinc_drop_rel = dinc_drop_abs / max(cur_m.delta_income, 1e-9)

            soldwavg_ok = (m_wo.sold_wavg <= (cur_m.sold_wavg + soldwavg_eps))
            recovery_ok = (m_wo.recovery_months <= (cur_m.recovery_months + recovery_eps))

            # Passenger if Δ-income doesn't meaningfully drop AND no other metric worsens
            if (dinc_drop_abs < abs_dinc_eps) and (dinc_drop_rel < rel_dinc_eps) and soldwavg_ok and recovery_ok:
                passenger_candidates.append(idx)

        if passenger_candidates:
            # Remove all passengers found in this pass and loop again
            for idx in passenger_candidates:
                current_mask[df.index.get_loc(idx)] = False
                removed.append(idx)
            changed = True  # iterate again with a tighter set

    final_m = compute_metrics(df, current_mask)
    # Safety: ensure final remains feasible; if not, roll back (very unlikely due to checks)
    if not feasible(final_m):
        return base_mask, {
            "base_metrics": asdict(base_m),
            "final_metrics": asdict(base_m),
            "removed_indices": [],  # rollback removes nothing
            "kept_enablers": enablers,
        }

    return current_mask, {
        "base_metrics": asdict(base_m),
        "final_metrics": asdict(final_m),
        "removed_indices": removed,
        "kept_enablers": enablers,
    }
