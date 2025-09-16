# src/genetic_algorithm.py
"""
Simplified NSGA-II Pareto optimizer for bond swap optimization.

Focuses on the core objectives that matter most for bond swaps:
1. Maximize Δ-income
2. Minimize recovery time
3. Achieve good proceeds coverage across the target range

Supports scenario-based constraints including minimum recovery periods for tax loss swaps.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

from src.config import (
    MIN_SWAP_SIZE_DOLLARS,
    MAX_SWAP_SIZE_DOLLARS,
    ENFORCE_MIN_SWAP_LOSS,
    MIN_TOTAL_SWAP_LOSS_DOLLARS,
    MAX_TOTAL_SWAP_LOSS_DOLLARS,
    MIN_RECOVERY_PERIOD_MONTHS,
    MAX_RECOVERY_PERIOD_MONTHS,
    SOLD_WAVG_PROJ_YIELD_MAX,
    GA_POP_SIZE,
    GA_GENERATIONS,
    GA_MUTATION_RATE,
    GA_CROSSOVER_RATE,
    GA_SEED,
    PROCEEDS_ANCHOR_COUNT,
    PROCEEDS_ANCHORS,
    SCENARIO_MODE,
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


def _safe_num(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _compute_metrics(df: pd.DataFrame, mask: np.ndarray) -> SwapMetrics:
    """Compute swap metrics for a given bond selection"""
    sel = df.loc[mask]
    if sel.empty:
        return SwapMetrics(0, 0, 0, 0, 0, 0, float("inf"), 0, float("inf"), 0)

    nsum = lambda c: _safe_num(pd.to_numeric(sel.get(c, 0), errors="coerce").sum())
    par = nsum("par")
    book = nsum("book")
    market = nsum("market")
    loss = nsum("loss")  # positive = loss (book - market)
    income = nsum("income")  # proj_yield * par
    d_inc = nsum("delta_income")  # buyback_income - income

    sold_wavg = (income / par) if par > 0 else float("inf")
    rec_mo = (12.0 * loss / d_inc) if d_inc > 0 else float("inf")

    # MV-weighted projected yield
    if "proj_yield" in sel.columns and sel["proj_yield"].notna().any() and market > 0:
        proj_y = float(
            (pd.to_numeric(sel["proj_yield"], errors="coerce") * pd.to_numeric(sel["market"], errors="coerce")).sum()
            / market
        )
    else:
        proj_y = (income / market) if market > 0 else 0.0

    return SwapMetrics(par, book, market, loss, income, d_inc, sold_wavg, proj_y, rec_mo, len(sel))


def _feasible(m: SwapMetrics) -> bool:
    """Check if a swap meets all hard constraints"""
    # Size constraints
    if m.market < MIN_SWAP_SIZE_DOLLARS or m.market > MAX_SWAP_SIZE_DOLLARS:
        return False

    # Loss constraints
    if m.loss > MAX_TOTAL_SWAP_LOSS_DOLLARS:
        return False
    if ENFORCE_MIN_SWAP_LOSS and m.loss < MIN_TOTAL_SWAP_LOSS_DOLLARS:
        return False

    # Recovery period constraints (including minimum for tax loss swaps)
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


# Proceeds coverage objective helpers
def _normalized_proceeds(market_val: float) -> float:
    """Normalize market value to [0,1] range within swap size window"""
    rng = max(1.0, float(MAX_SWAP_SIZE_DOLLARS - MIN_SWAP_SIZE_DOLLARS))
    p = (float(market_val) - float(MIN_SWAP_SIZE_DOLLARS)) / rng
    return max(0.0, min(1.0, p))  # Clamp to [0,1]


def _proceeds_anchors() -> List[float]:
    """Get anchor positions for proceeds coverage objective"""
    if PROCEEDS_ANCHORS and len(PROCEEDS_ANCHORS) > 0:
        return list(PROCEEDS_ANCHORS)
    N = max(2, int(PROCEEDS_ANCHOR_COUNT))
    return [(i + 0.5) / N for i in range(N)]


def _coverage_to_nearest_anchor(p_norm: float, anchors: List[float]) -> float:
    """Distance to nearest anchor (0 = at anchor)"""
    return float(min(abs(p_norm - a) for a in anchors)) if anchors else 0.0


def _objectives(m: SwapMetrics, anchors: List[float]) -> Tuple[float, float, float]:
    """
    Build objective tuple for NSGA-II (larger = better):
    1. Δ-income (maximize)
    2. -recovery_months (minimize recovery time)
    3. -coverage_distance (get close to proceeds anchors for diversity)
    """
    rec = float(m.recovery_months if math.isfinite(m.recovery_months) else 1e9)
    p_norm = _normalized_proceeds(m.market)
    coverage = _coverage_to_nearest_anchor(p_norm, anchors)

    return (
        float(m.delta_income),
        float(-rec),
        float(-coverage),
    )


# NSGA-II Implementation
def _non_dominated_sort(pop_objs: List[Tuple[np.ndarray, Tuple[float, ...]]]) -> List[List[int]]:
    """Sort population into non-dominated fronts"""
    S = [set() for _ in pop_objs]
    n = [0] * len(pop_objs)
    fronts: List[List[int]] = [[]]

    for p, (_, op) in enumerate(pop_objs):
        for q, (_, oq) in enumerate(pop_objs):
            if p == q:
                continue
            # Larger is better dominance
            p_dom_q = all(a >= b for a, b in zip(op, oq)) and any(a > b for a, b in zip(op, oq))
            q_dom_p = all(b >= a for a, b in zip(op, oq)) and any(b > a for a, b in zip(op, oq))

            if p_dom_q:
                S[p].add(q)
            elif q_dom_p:
                n[p] += 1

        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts):
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(q)
        if nxt:
            fronts.append(nxt)
        i += 1

    return fronts


def _crowding_distance(front: List[int], objs: List[Tuple[float, ...]]) -> Dict[int, float]:
    """Calculate crowding distance for diversity"""
    if not front:
        return {}

    m = len(objs[0])
    dist = {i: 0.0 for i in front}

    for k in range(m):
        order = sorted(front, key=lambda i: objs[i][k])
        dist[order[0]] = dist[order[-1]] = float("inf")

        vmin, vmax = objs[order[0]][k], objs[order[-1]][k]
        if vmax == vmin:
            continue

        for j in range(1, len(order) - 1):
            prev, nxt = order[j - 1], order[j + 1]
            dist[order[j]] += (objs[nxt][k] - objs[prev][k]) / (vmax - vmin)

    return dist


def _crossover(rng: np.random.Generator, a: np.ndarray, b: np.ndarray, rate: float) -> np.ndarray:
    """Single-point crossover"""
    if rng.random() >= rate:
        return a.copy()
    cut = int(rng.integers(1, len(a) - 1))
    return np.concatenate([a[:cut], b[cut:]]).copy()


def _mutate(rng: np.random.Generator, mask: np.ndarray, rate: float) -> np.ndarray:
    """Bit-flip mutation"""
    if rng.random() >= rate:
        return mask.copy()
    n = len(mask)
    flips = max(1, int(n * 0.05))  # Flip ~5% of bits
    idx = rng.choice(n, size=flips, replace=False)
    child = mask.copy()
    child[idx] = ~child[idx]
    return child


def _repair(df: pd.DataFrame, mask: np.ndarray, max_iterations: int = 10) -> Optional[np.ndarray]:
    """Repair infeasible solutions to meet constraints"""
    m = _compute_metrics(df, mask)
    if _feasible(m):
        return mask

    work = mask.copy()

    for iteration in range(max_iterations):
        m = _compute_metrics(df, work)

        if _feasible(m):
            return work

        changed = False

        # Fix size constraints first
        if m.market < MIN_SWAP_SIZE_DOLLARS:
            # Add bonds with best delta_income/loss ratio
            candidates = (~work)
            if candidates.any():
                df_cand = df.loc[candidates].copy()
                df_cand["score"] = pd.to_numeric(df_cand["delta_income"], errors="coerce") / (
                        1.0 + pd.to_numeric(df_cand["loss"], errors="coerce").clip(lower=0.0)
                )
                df_cand = df_cand.replace([np.inf, -np.inf], np.nan).fillna(0.0).infer_objects(copy=False)
                best_idx = df_cand["score"].idxmax()
                if pd.notna(best_idx):
                    work[best_idx] = True
                    changed = True

        elif m.market > MAX_SWAP_SIZE_DOLLARS:
            # Remove bonds with largest market values
            if work.any():
                df_sel = df.loc[work].copy()
                worst_idx = pd.to_numeric(df_sel["market"], errors="coerce").idxmax()
                if pd.notna(worst_idx):
                    work[worst_idx] = False
                    changed = True

        # Fix loss constraints
        if m.loss > MAX_TOTAL_SWAP_LOSS_DOLLARS:
            # Remove bonds with highest loss
            if work.any():
                df_sel = df.loc[work].copy()
                worst_idx = pd.to_numeric(df_sel["loss"], errors="coerce").idxmax()
                if pd.notna(worst_idx):
                    work[worst_idx] = False
                    changed = True

        elif ENFORCE_MIN_SWAP_LOSS and m.loss < MIN_TOTAL_SWAP_LOSS_DOLLARS:
            # Add bonds to increase loss
            candidates = (~work)
            if candidates.any():
                df_cand = df.loc[candidates].copy()
                best_idx = pd.to_numeric(df_cand["loss"], errors="coerce").idxmax()
                if pd.notna(best_idx):
                    work[best_idx] = True
                    changed = True

        # Fix recovery period constraints
        if m.recovery_months < MIN_RECOVERY_PERIOD_MONTHS:
            # For tax loss swaps, need to increase recovery time
            # Add bonds with lower delta_income ratio to slow recovery
            candidates = (~work)
            if candidates.any():
                df_cand = df.loc[candidates].copy()
                df_cand["recovery_impact"] = pd.to_numeric(df_cand["loss"], errors="coerce") / (
                        pd.to_numeric(df_cand["delta_income"], errors="coerce") + 1e-6
                )
                best_idx = df_cand["recovery_impact"].idxmax()
                if pd.notna(best_idx):
                    work[best_idx] = True
                    changed = True

        elif m.recovery_months > MAX_RECOVERY_PERIOD_MONTHS:
            # Remove bonds that slow recovery most
            if work.any():
                df_sel = df.loc[work].copy()
                df_sel["recovery_impact"] = pd.to_numeric(df_sel["loss"], errors="coerce") / (
                        pd.to_numeric(df_sel["delta_income"], errors="coerce") + 1e-6
                )
                worst_idx = df_sel["recovery_impact"].idxmax()
                if pd.notna(worst_idx):
                    work[worst_idx] = False
                    changed = True

        # Fix yield constraint
        if m.sold_wavg > SOLD_WAVG_PROJ_YIELD_MAX:
            # Remove bonds with highest yields
            if work.any():
                df_sel = df.loc[work].copy()
                worst_idx = pd.to_numeric(df_sel["proj_yield"], errors="coerce").idxmax()
                if pd.notna(worst_idx):
                    work[worst_idx] = False
                    changed = True

        if not changed:
            break

    # Final check
    final_m = _compute_metrics(df, work)
    return work if _feasible(final_m) else None


def evolve_nsga(df: pd.DataFrame, seed: int = GA_SEED) -> List[Dict]:
    """
    Main NSGA-II evolution with scenario-aware constraints
    """
    print(f"Running NSGA-II optimization for {SCENARIO_MODE} scenario...")

    rng = np.random.default_rng(seed)
    n = len(df)
    anchors = _proceeds_anchors()

    # Initialize population
    pop: List[np.ndarray] = []
    tries = 0
    max_tries = GA_POP_SIZE * 20

    print("Initializing population...")
    while len(pop) < GA_POP_SIZE and tries < max_tries:
        # Random initial solution
        k = int(rng.integers(max(1, int(0.05 * n)), max(2, int(0.3 * n)) + 1))
        mask = np.zeros(n, dtype=bool)
        mask[rng.choice(n, size=k, replace=False)] = True

        # Repair to make feasible
        repaired = _repair(df, mask)
        if repaired is not None:
            pop.append(repaired)
        tries += 1

    if not pop:
        print("Could not generate any feasible initial solutions")
        return []

    print(f"Generated {len(pop)} initial solutions")

    def eval_mask(mask: np.ndarray) -> Tuple[Tuple[float, ...], SwapMetrics]:
        met = _compute_metrics(df, mask)
        if not _feasible(met):
            # Return dominated objectives for infeasible solutions
            return ((-1e12, -1e12, -1e12), met)
        return (_objectives(met, anchors), met)

    # Evolution loop
    for generation in tqdm(range(GA_GENERATIONS), desc="Evolving", unit="gen"):
        # Evaluate population
        objs: List[Tuple[float, ...]] = []
        mets: List[SwapMetrics] = []
        for mask in pop:
            o, m = eval_mask(mask)
            objs.append(o)
            mets.append(m)

        # Non-dominated sorting
        pop_objs = list(zip(pop, objs))
        fronts = _non_dominated_sort(pop_objs)

        # Select survivors using crowding distance
        new_pop: List[np.ndarray] = []
        for front in fronts:
            if not front:
                continue

            if len(new_pop) + len(front) <= GA_POP_SIZE:
                # Take entire front
                for i in front:
                    new_pop.append(pop[i])
            else:
                # Partial front selection using crowding distance
                d = _crowding_distance(front, objs)
                sorted_front = sorted(front, key=lambda i: d.get(i, 0.0), reverse=True)
                remaining = GA_POP_SIZE - len(new_pop)
                for i in sorted_front[:remaining]:
                    new_pop.append(pop[i])
                break

        pop = new_pop

        # Generate offspring
        children: List[np.ndarray] = []
        while len(children) < GA_POP_SIZE:
            # Tournament selection
            a, b = rng.choice(len(pop), size=2, replace=False)
            child = _crossover(rng, pop[a], pop[b], GA_CROSSOVER_RATE)
            child = _mutate(rng, child, GA_MUTATION_RATE)

            # Repair offspring
            repaired = _repair(df, child)
            if repaired is not None:
                children.append(repaired)
            else:
                # If repair fails, use parent
                children.append(pop[a].copy())

        pop = children

    # Final evaluation and extract Pareto front
    final_objs: List[Tuple[float, ...]] = []
    final_mets: List[SwapMetrics] = []
    for mask in pop:
        o, m = eval_mask(mask)
        final_objs.append(o)
        final_mets.append(m)

    final_fronts = _non_dominated_sort(list(zip(pop, final_objs)))
    pareto_indices = final_fronts[0] if final_fronts else []

    # Build results
    results = []
    for i in pareto_indices:
        m = final_mets[i]
        results.append({
            "mask": pop[i],
            "metrics": m,
            "objectives": final_objs[i],
        })

    # Sort by delta income (primary) then recovery time (secondary)
    results.sort(key=lambda r: (-r["metrics"].delta_income, r["metrics"].recovery_months))

    print(f"Found {len(results)} Pareto-optimal solutions")

    return results


# Legacy compatibility function
def run_optimizer(bond_candidates_df: pd.DataFrame, n_jobs: int = -1):
    """Backward-compatible entry point"""
    pareto = evolve_nsga(bond_candidates_df, seed=GA_SEED)
    if not pareto:
        return np.zeros(len(bond_candidates_df), dtype=bool), {
            "delta_income": 0.0, "loss": 0.0, "recovery_months": float("inf"),
            "sold_wavg": float("inf"), "par": 0.0, "book": 0.0, "market": 0.0,
            "income": 0.0, "proj_y": 0.0, "count": 0,
        }

    best = pareto[0]
    m: SwapMetrics = best["metrics"]
    return best["mask"], {
        "par": m.par, "book": m.book, "market": m.market, "loss": m.loss,
        "income": m.income, "delta_income": m.delta_income, "sold_wavg": m.sold_wavg,
        "proj_y": m.proj_y, "recovery_months": m.recovery_months, "count": m.count,
    }