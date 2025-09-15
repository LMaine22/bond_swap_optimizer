# src/genetic_algorithm.py
"""
NSGA-II Pareto optimizer for the bond swap project.

This version adds a proceeds-coverage objective so the frontier spreads across
the proceeds window [MIN_SWAP_SIZE_DOLLARS .. MAX_SWAP_SIZE_DOLLARS] without
penalties or multiple runs.

Objectives (larger-is-better in this implementation):
    (1)  Δ-income                      (maximize)
    (2)  -recovery_months              (minimize recovery)
    (3)  -coverage_to_proceeds_anchor  (minimize distance to nearest anchor)

Hard feasibility (discard/repair if violated):
    - Proceeds within [MIN, MAX]
    - Total loss ≤ MAX_TOTAL_SWAP_LOSS_DOLLARS
    - Recovery months ≤ MAX_RECOVERY_PERIOD_MONTHS
    - Sold WAVG projected yield ≤ SOLD_WAVG_PROJ_YIELD_MAX
    - Δ-income > 0

Public API:
    - evolve_nsga(df) -> List[Dict] with {'mask','metrics','objectives', ...}
    - collect_pareto_menu(pareto, df, k) -> diversified slice (legacy helper)
    - run_optimizer(df) -> (best_mask, metrics_dict) (legacy/compat)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress pandas warnings about deprecated fillna behavior
pd.set_option('future.no_silent_downcasting', True)

from src.config import (
    # Menu (legacy helper)
    MENU_NUM_OPTIONS,
    # Caps
    MIN_SWAP_SIZE_DOLLARS,
    MAX_SWAP_SIZE_DOLLARS,
    ENFORCE_MIN_SWAP_LOSS,
    MIN_TOTAL_SWAP_LOSS_DOLLARS,
    MAX_TOTAL_SWAP_LOSS_DOLLARS,
    MAX_RECOVERY_PERIOD_MONTHS,
    SOLD_WAVG_PROJ_YIELD_MAX,     # already decimalized in config
    # GA params
    GA_POP_SIZE,
    GA_GENERATIONS,
    GA_MUTATION_RATE,
    GA_CROSSOVER_RATE,
    GA_SEED,
    # Proceeds coverage objective
    PROCEEDS_ANCHOR_COUNT,
    PROCEEDS_ANCHORS,
)

# -------------------- metric helpers --------------------

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
    sel = df.loc[mask]
    if sel.empty:
        return SwapMetrics(0,0,0,0,0,0,float("inf"),0,float("inf"),0)

    nsum = lambda c: _safe_num(pd.to_numeric(sel.get(c, 0), errors="coerce").sum())
    par    = nsum("par")
    book   = nsum("book")
    market = nsum("market")
    loss   = nsum("loss")                   # positive = loss (book - market)
    income = nsum("income")                 # proj_yield * par
    d_inc  = nsum("delta_income")           # buyback_income - income

    sold_wavg = (income / par) if par > 0 else float("inf")
    rec_mo    = (12.0 * loss / d_inc) if d_inc > 0 else float("inf")

    # MV-weighted projected yield (fallback to income/market if proj_yield missing)
    if "proj_yield" in sel.columns and sel["proj_yield"].notna().any() and market > 0:
        proj_y = float(
            (pd.to_numeric(sel["proj_yield"], errors="coerce") * pd.to_numeric(sel["market"], errors="coerce")).sum()
            / market
        )
    else:
        proj_y = (income / market) if market > 0 else 0.0

    return SwapMetrics(par, book, market, loss, income, d_inc, sold_wavg, proj_y, rec_mo, len(sel))

def _feasible(m: SwapMetrics) -> bool:
    if m.market < MIN_SWAP_SIZE_DOLLARS or m.market > MAX_SWAP_SIZE_DOLLARS:
        return False
    if m.loss > MAX_TOTAL_SWAP_LOSS_DOLLARS:
        return False
    if ENFORCE_MIN_SWAP_LOSS and m.loss < MIN_TOTAL_SWAP_LOSS_DOLLARS:
        return False
    if m.recovery_months > MAX_RECOVERY_PERIOD_MONTHS:
        return False
    if m.sold_wavg > SOLD_WAVG_PROJ_YIELD_MAX:
        return False
    if m.delta_income <= 0:
        return False
    return True

# ---------- proceeds-coverage objective helpers ----------

def _normalized_proceeds(market_val: float) -> float:
    rng = max(1.0, float(MAX_SWAP_SIZE_DOLLARS - MIN_SWAP_SIZE_DOLLARS))
    p = (float(market_val) - float(MIN_SWAP_SIZE_DOLLARS)) / rng
    # clamp to [0,1] to be robust when repair nudges just outside window temporarily
    return 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)

def _proceeds_anchors() -> List[float]:
    """
    Return anchor positions in normalized [0,1].
    If PROCEEDS_ANCHORS is provided, use it (assumed already normalized).
    Else create midpoints of PROCEEDS_ANCHOR_COUNT equal bins.
    """
    if PROCEEDS_ANCHORS and len(PROCEEDS_ANCHORS) > 0:
        return list(PROCEEDS_ANCHORS)
    N = max(2, int(PROCEEDS_ANCHOR_COUNT))
    return [ (i + 0.5) / N for i in range(N) ]

def _coverage_to_nearest_anchor(p_norm: float, anchors: List[float]) -> float:
    """Absolute distance in normalized space to the closest anchor (0 = at anchor)."""
    return float(min(abs(p_norm - a) for a in anchors)) if anchors else 0.0

def _objectives(m: SwapMetrics, anchors: List[float]) -> Tuple[float,float,float]:
    """
    Build the objective tuple (larger = better).
      obj1 =  Δ-income
      obj2 = -recovery_months
      obj3 = -coverage_to_nearest_anchor  (smaller distance → larger objective)
    """
    rec = float(m.recovery_months if math.isfinite(m.recovery_months) else 1e9)
    p_norm = _normalized_proceeds(m.market)
    coverage = _coverage_to_nearest_anchor(p_norm, anchors)
    return (
        float(m.delta_income),
        float(-rec),
        float(-coverage),
    )

# -------------------- NSGA-II core --------------------

def _non_dominated_sort(pop_objs: List[Tuple[np.ndarray, Tuple[float,...]]]) -> List[List[int]]:
    S = [set() for _ in pop_objs]; n = [0]*len(pop_objs); fronts: List[List[int]] = [[]]
    for p, (_, op) in enumerate(pop_objs):
        for q, (_, oq) in enumerate(pop_objs):
            if p == q:
                continue
            # "larger is better" dominance
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

def _crowding_distance(front: List[int], objs: List[Tuple[float,...]]) -> Dict[int, float]:
    if not front:
        return {}
    m = len(objs[0]); dist = {i: 0.0 for i in front}
    for k in range(m):
        order = sorted(front, key=lambda i: objs[i][k])
        dist[order[0]] = dist[order[-1]] = float("inf")
        vmin, vmax = objs[order[0]][k], objs[order[-1]][k]
        if vmax == vmin:
            continue
        for j in range(1, len(order)-1):
            prev, nxt = order[j-1], order[j+1]
            dist[order[j]] += (objs[nxt][k] - objs[prev][k]) / (vmax - vmin)
    return dist

def _crossover(rng: np.random.Generator, a: np.ndarray, b: np.ndarray, rate: float) -> np.ndarray:
    if rng.random() >= rate:
        return a.copy()
    cut = int(rng.integers(1, len(a)-1))
    return np.concatenate([a[:cut], b[cut:]]).copy()

def _mutate(rng: np.random.Generator, mask: np.ndarray, rate: float) -> np.ndarray:
    if rng.random() >= rate:
        return mask.copy()
    n = len(mask)
    flips = max(1, int(n * 0.05))  # flip ~5% of bits
    idx = rng.choice(n, size=flips, replace=False)
    child = mask.copy(); child[idx] = ~child[idx]
    return child

def _repair(df: pd.DataFrame, mask: np.ndarray) -> Optional[np.ndarray]:
    """Greedy repair to satisfy hard constraints (caps/window)."""
    m = _compute_metrics(df, mask)
    if _feasible(m):
        return mask
    work = mask.copy()

    # If too small, greedily add best Δ-income / (1+loss) names
    if m.market < MIN_SWAP_SIZE_DOLLARS:
        candidates = (~work)
        df2 = df.loc[candidates].copy()
        df2["score"] = pd.to_numeric(df2["delta_income"], errors="coerce") / (
            1.0 + pd.to_numeric(df2["loss"], errors="coerce").clip(lower=0.0)
        )
        df2 = df2.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).fillna(0.0)
        for idx in df2.sort_values("score", ascending=False).index:
            work[idx] = True
            m = _compute_metrics(df, work)
            if m.market >= MIN_SWAP_SIZE_DOLLARS:
                break

    changed = True
    while changed and not _feasible(m):
        changed = False

        # Too much loss → drop worst (high loss – 0.2*Δinc)
        if m.loss > MAX_TOTAL_SWAP_LOSS_DOLLARS:
            sel = df.loc[work].copy()
            sel["score"] = pd.to_numeric(sel["loss"], errors="coerce") - 0.2 * pd.to_numeric(sel["delta_income"], errors="coerce")
            for idx in sel.sort_values("score", ascending=True).index[::-1]:
                work[idx] = False
                m = _compute_metrics(df, work); changed = True
                if m.loss <= MAX_TOTAL_SWAP_LOSS_DOLLARS:
                    break
        
        # Too little loss → add best (high Δinc / (1 + loss)) - only if enforcing min loss
        if ENFORCE_MIN_SWAP_LOSS and m.loss < MIN_TOTAL_SWAP_LOSS_DOLLARS:
            candidates = (~work)
            df2 = df.loc[candidates].copy()
            df2["score"] = pd.to_numeric(df2["delta_income"], errors="coerce") / (
                1.0 + pd.to_numeric(df2["loss"], errors="coerce").clip(lower=0.0)
            )
            df2 = df2.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).fillna(0.0)
            for idx in df2.sort_values("score", ascending=False).index:
                work[idx] = True
                m = _compute_metrics(df, work)
                if m.loss >= MIN_TOTAL_SWAP_LOSS_DOLLARS:
                    break

        # Sold wavg proj yield too high → drop highest proj_yield first
        if m.sold_wavg > SOLD_WAVG_PROJ_YIELD_MAX:
            sel = df.loc[work].copy()
            sel["score"] = pd.to_numeric(sel["proj_yield"], errors="coerce")
            for idx in sel.sort_values("score", ascending=False).index:
                work[idx] = False
                m = _compute_metrics(df, work); changed = True
                if m.sold_wavg <= SOLD_WAVG_PROJ_YIELD_MAX:
                    break

        # Too large → drop largest market values
        if m.market > MAX_SWAP_SIZE_DOLLARS:
            sel = df.loc[work].copy()
            sel["score"] = pd.to_numeric(sel["market"], errors="coerce")
            for idx in sel.sort_values("score", ascending=False).index:
                work[idx] = False
                m = _compute_metrics(df, work); changed = True
                if m.market <= MAX_SWAP_SIZE_DOLLARS:
                    break

    return work if _feasible(m) else None

# -------------------- public API --------------------

def evolve_nsga(df: pd.DataFrame, seed: int = GA_SEED) -> List[Dict]:
    rng = np.random.default_rng(seed)
    n = len(df)

    # Log anchors once for transparency
    anchors = _proceeds_anchors()
    #print(f"NSGA objective (proceeds coverage): {len(anchors)} anchors at {anchors}")

    # Initialize population with feasible masks
    pop: List[np.ndarray] = []
    tries = 0
    while len(pop) < GA_POP_SIZE and tries < GA_POP_SIZE * 50:
        k = int(rng.integers(max(1, int(0.1*n)), max(2, int(0.5*n))+1))
        mask = np.zeros(n, dtype=bool)
        mask[rng.choice(n, size=k, replace=False)] = True
        repaired = _repair(df, mask)
        if repaired is not None:
            pop.append(repaired)
        tries += 1
    if not pop:
        pop = [np.zeros(n, dtype=bool)]

    def eval_mask(mask: np.ndarray) -> Tuple[Tuple[float, ...], SwapMetrics]:
        met = _compute_metrics(df, mask)
        if not _feasible(met):
            # dominated "junk" vector; keep length == number of objectives (3)
            return ((-1e12, -1e12, -1e12), met)
        return (_objectives(met, anchors), met)

    # Evolution loop
    for generation in tqdm(range(GA_GENERATIONS), desc="Evolving NSGA-II", unit="gen"):
        objs: List[Tuple[float, ...]] = []
        mets: List[SwapMetrics] = []
        for mask in pop:
            o, m = eval_mask(mask); objs.append(o); mets.append(m)

        pop_objs = list(zip(pop, objs))
        fronts = _non_dominated_sort(pop_objs)

        # Crowding selection
        new_pop: List[np.ndarray] = []
        for front in fronts:
            if not front:
                continue
            d = _crowding_distance(front, objs)
            for i in sorted(front, key=lambda i: d.get(i, 0.0), reverse=True):
                new_pop.append(pop[i])
                if len(new_pop) >= GA_POP_SIZE:
                    break
            if len(new_pop) >= GA_POP_SIZE:
                break
        pop = new_pop

        # Crossover + mutation + repair
        children: List[np.ndarray] = []
        while len(children) < GA_POP_SIZE:
            a, b = rng.choice(len(pop), size=2, replace=False)
            child = _crossover(rng, pop[a], pop[b], GA_CROSSOVER_RATE)
            child = _mutate(rng, child, GA_MUTATION_RATE)
            repaired = _repair(df, child)
            if repaired is not None:
                children.append(repaired)
        pop = children

    # Final evaluation
    objs: List[Tuple[float, ...]] = []
    mets: List[SwapMetrics] = []
    for mask in pop:
        o, m = eval_mask(mask); objs.append(o); mets.append(m)
    pareto_fronts = _non_dominated_sort(list(zip(pop, objs)))
    pareto_idx = pareto_fronts[0] if pareto_fronts else []

    # Pack results (rank/crowding optional; main() defaults to 0/NaN if absent)
    results = []
    for i in pareto_idx:
        m = mets[i]
        results.append({
            "mask": pop[i],
            "metrics": m,
            "objectives": objs[i],
            # Optionally attach diagnostics if you want:
            # "rank": 0,
            # "crowding_distance": np.nan,
        })
    # Sort primarily by Δ-income desc, then recovery asc (presentation order)
    results.sort(key=lambda r: (-r["metrics"].delta_income, r["metrics"].recovery_months))
    return results

def collect_pareto_menu(pareto: List[Dict], df: pd.DataFrame, k: int | None = None) -> List[Dict]:
    """Pick a diversified list from the Pareto front (Jaccard ≥ 0.25)."""
    k = k or MENU_NUM_OPTIONS
    chosen: List[Dict] = []
    chosen_masks: List[np.ndarray] = []

    def jacc(a: np.ndarray, b: np.ndarray) -> float:
        a1, b1 = a.astype(bool), b.astype(bool)
        inter = np.logical_and(a1, b1).sum()
        union = np.logical_or(a1, b1).sum()
        return 1.0 - (inter / union if union > 0 else 0.0)

    for cand in pareto:
        msk = cand["mask"]
        if not chosen_masks:
            chosen.append(cand); chosen_masks.append(msk)
            if len(chosen) >= k: break
            continue
        dmin = min(jacc(msk, c) for c in chosen_masks)
        if dmin >= 0.25:
            chosen.append(cand); chosen_masks.append(msk)
            if len(chosen) >= k: break

    # If we still need more, just fill from the front in order
    i = 0
    while len(chosen) < k and i < len(pareto):
        cand = pareto[i]
        if cand not in chosen:
            chosen.append(cand)
        i += 1

    # Attach convenience fields
    menu = []
    for idx, item in enumerate(chosen, start=1):
        m = item["metrics"]
        rows = df.loc[item["mask"]].copy()
        rows["option_id"] = f"OPT_{idx:02d}"
        menu.append({
            "option_id": f"OPT_{idx:02d}",
            "mask": item["mask"],
            "metrics": m,
            "rows": rows
        })
    return menu

def run_optimizer(bond_candidates_df: pd.DataFrame, n_jobs: int = -1):
    """
    Backward-compatible entrypoint used by main.py:
    returns (best_mask, metrics_dict).
    """
    pareto = evolve_nsga(bond_candidates_df, seed=GA_SEED)
    if not pareto:
        return np.zeros(len(bond_candidates_df), dtype=bool), {
            "delta_income": 0.0, "loss": 0.0, "recovery_months": float("inf"), "sold_wavg": float("inf"),
            "par": 0.0, "book": 0.0, "market": 0.0, "income": 0.0, "proj_y": 0.0, "count": 0,
        }
    best = pareto[0]; m: SwapMetrics = best["metrics"]
    return best["mask"], {
        "par": m.par, "book": m.book, "market": m.market, "loss": m.loss, "income": m.income,
        "delta_income": m.delta_income, "sold_wavg": m.sold_wavg, "proj_y": m.proj_y,
        "recovery_months": m.recovery_months, "count": m.count,
    }
