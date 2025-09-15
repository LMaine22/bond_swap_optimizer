# src/tr_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from src.config import (
    TR_HORIZON_MONTHS,
    TR_PARALLEL_SHIFTS_BPS,
    TARGET_BUY_BACK_YIELD,
    ASSUMED_HOLD_DURATION_YEARS,
    ASSUMED_HOLD_CONVEXITY,
    ASSUMED_BUY_DURATION_YEARS,
    ASSUMED_BUY_CONVEXITY,
    KEY_RATE_BUCKETS,
    KEY_RATE_SHOCKS,
    KEY_RATE_KRD_COLUMN_MAP,
    BUY_KRD_WEIGHTS_DEFAULT,
    TR_INCLUDE_ROLL_ON_SWAP,  # NEW
    # New funding overlay imports
    FUNDING_OVERLAY_ENABLED,
    FUNDING_BASE_RATE_BPS,
    FUNDING_SPREAD_BPS,
    FUNDING_TERM_MONTHS,
    FUNDING_MULTIPLIER_OPTIONS,
    FUNDING_UNWIND_MONTH,
    WAIT_TO_BUY_ENABLED,
    WAIT_TO_BUY_MONTHS,
    CASH_PARKING_RATE_BPS,
    EXPECTED_RATE_CHANGES,
    EXPECTED_SPREAD_CHANGES,
)

# ---- column aliases for your sheet ----
DUR_COLS = ["Eff Dur", "mod_duration", "duration", "dur", "mduration"]
CONV_COLS = ["Eff Convex", "convexity", "conv"]
YTM_COLS = ["Proj Yield", "Proj Yield (TE)", "YTM", "YTM (TE)", "Market Yield", "Mrkt Yield (TE)", "Acctg Yield",
            "Acctg Yld (TE) (1Yr)"]
AVG_LIFE_COLS = ["Avg Life", "Average Life", "WAL"]


@dataclass
class TRBreakdown:
    income_hold: float
    price_hold: float
    roll_hold: float
    income_swap: float
    price_swap: float
    roll_swap: float
    realized_pl_today: float
    tr_hold_dollar: float
    tr_swap_dollar: float
    tr_excess_dollar: float
    tr_hold_pct: float
    tr_swap_pct: float
    tr_excess_pct: float


@dataclass
class OverlayBreakdown:
    """Extended breakdown for funding and wait-then-buy overlays"""
    # Standard swap components
    standard: TRBreakdown

    # Funding overlay components
    funding_cost: float = 0.0
    funded_income: float = 0.0
    funded_price: float = 0.0
    funded_unwind_gain: float = 0.0
    tr_funded_dollar: float = 0.0
    tr_funded_pct: float = 0.0
    tr_funded_excess: float = 0.0

    # Wait-then-buy components
    cash_income: float = 0.0
    delayed_income: float = 0.0
    delayed_price: float = 0.0
    tr_wait_dollar: float = 0.0
    tr_wait_pct: float = 0.0
    tr_wait_excess: float = 0.0

    # Comparative metrics
    funding_breakeven_rate: float = 0.0
    wait_advantage_bps: float = 0.0
    time_to_profit_months: float = 0.0


# ---- helpers ----
def _get_first_existing(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _mv_weights(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    mv = pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0).values
    tot = mv.sum()
    return (mv / tot) if tot > 0 else np.zeros_like(mv)


def _weighted_or_assumed(df: pd.DataFrame, mask: np.ndarray, cols: List[str], assumed: float) -> float:
    col = _get_first_existing(df, cols)
    if col is None:
        return float(assumed)
    s = pd.to_numeric(df.loc[mask, col], errors="coerce").astype(float)
    w = _mv_weights(df, mask)
    if w.sum() <= 0 or not np.isfinite(s).any():
        return float(assumed)
    return float(np.nansum(s.values * w))


def _income_from_sold(df: pd.DataFrame, mask: np.ndarray) -> float:
    if "income" in df.columns:
        return float(pd.to_numeric(df.loc[mask, "income"], errors="coerce").fillna(0.0).sum())
    ycol = _get_first_existing(df, YTM_COLS)
    if (ycol is not None) and ("par" in df.columns):
        y = pd.to_numeric(df.loc[mask, ycol], errors="coerce").fillna(0.0)
        par = pd.to_numeric(df.loc[mask, "par"], errors="coerce").fillna(0.0)
        return float((y * par).sum())
    mv = pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0).sum()
    avg_y = TARGET_BUY_BACK_YIELD if ycol is None else _weighted_or_assumed(df, mask, [ycol], TARGET_BUY_BACK_YIELD)
    return float(mv * avg_y)


def _years_to_maturity_like(df: pd.DataFrame, mask: np.ndarray, default_years: float = 5.0) -> pd.Series:
    if "years_to_maturity" in df.columns:
        y = pd.to_numeric(df.loc[mask, "years_to_maturity"], errors="coerce")
        return y.fillna(default_years).clip(lower=0.0)
    col = _get_first_existing(df, AVG_LIFE_COLS)
    if col is not None:
        y = pd.to_numeric(df.loc[mask, col], errors="coerce")
        return y.fillna(default_years).clip(lower=0.0)
    return pd.Series(np.full(mask.sum(), default_years), index=df.index[mask])


def _pull_to_par_roll(df: pd.DataFrame, mask: np.ndarray, horizon_years: float) -> float:
    if "par" not in df.columns or "market" not in df.columns:
        return 0.0
    ytm_like = _years_to_maturity_like(df, mask, default_years=5.0)
    par = pd.to_numeric(df.loc[mask, "par"], errors="coerce").fillna(0.0)
    mv = pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0)
    price_gap = par - mv
    with np.errstate(divide="ignore", invalid="ignore"):
        portion = (horizon_years / ytm_like).clip(lower=0.0)
    portion = portion.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return float((price_gap * portion).sum())


def _price_change_dollar(market_value: float, duration_years: float, convexity: float, shift_bps: float) -> float:
    dy = shift_bps / 10000.0
    return float(market_value * (-duration_years * dy + 0.5 * convexity * dy * dy))


def _dv01_from_duration(duration_years: float, market_value: float) -> float:
    return float(market_value * duration_years / 10000.0)


def _get_funding_rate() -> float:
    """Get current funding rate (base + spread)"""
    return (FUNDING_BASE_RATE_BPS + FUNDING_SPREAD_BPS) / 10000.0


def _get_cash_rate() -> float:
    """Get cash parking rate for wait-then-buy analysis"""
    return CASH_PARKING_RATE_BPS / 10000.0


# ---- PARALLEL TR (ORIGINAL) ----
def _compute_tr_components_parallel(df, mask, proceeds_mv, horizon_months, shift_bps, hold_dur, hold_conv, buy_dur,
                                    buy_conv):
    H = horizon_months / 12.0
    inc_hold_annual = _income_from_sold(df, mask)
    income_hold = inc_hold_annual * H
    roll_hold = _pull_to_par_roll(df, mask, horizon_years=H)
    price_hold = _price_change_dollar(proceeds_mv, hold_dur, hold_conv, shift_bps)

    realized_pl_today = - float(pd.to_numeric(df.loc[mask, "loss"], errors="coerce").fillna(0.0).sum())
    income_swap = float(proceeds_mv * TARGET_BUY_BACK_YIELD * H)
    roll_swap = roll_hold if TR_INCLUDE_ROLL_ON_SWAP else 0.0  # NEW
    price_swap = _price_change_dollar(proceeds_mv, buy_dur, buy_conv, shift_bps)

    tr_hold = income_hold + price_hold + roll_hold
    tr_swap = income_swap + price_swap + roll_swap + realized_pl_today
    tr_excess = tr_swap - tr_hold

    denom = proceeds_mv if proceeds_mv > 0 else 1.0
    return TRBreakdown(
        income_hold=income_hold, price_hold=price_hold, roll_hold=roll_hold,
        income_swap=income_swap, price_swap=price_swap, roll_swap=roll_swap,
        realized_pl_today=realized_pl_today,
        tr_hold_dollar=tr_hold, tr_swap_dollar=tr_swap, tr_excess_dollar=tr_excess,
        tr_hold_pct=tr_hold / denom, tr_swap_pct=tr_swap / denom, tr_excess_pct=tr_excess / denom
    )


def compute_parallel_tr_table(df: pd.DataFrame, mask: np.ndarray) -> Tuple[pd.DataFrame, Optional[float]]:
    proceeds_mv = float(pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0).sum())
    hold_dur = _weighted_or_assumed(df, mask, DUR_COLS, ASSUMED_HOLD_DURATION_YEARS)
    hold_conv = _weighted_or_assumed(df, mask, CONV_COLS, ASSUMED_HOLD_CONVEXITY)
    buy_dur, buy_conv = ASSUMED_BUY_DURATION_YEARS, ASSUMED_BUY_CONVEXITY
    horizon = TR_HORIZON_MONTHS

    rows = []
    for bps in TR_PARALLEL_SHIFTS_BPS:
        br = _compute_tr_components_parallel(df, mask, proceeds_mv, horizon, bps, hold_dur, hold_conv, buy_dur,
                                             buy_conv)
        rows.append({
            "shift_bps": bps,
            "TR_Hold_$": br.tr_hold_dollar, "TR_Swap_$": br.tr_swap_dollar, "Excess_$": br.tr_excess_dollar,
            "TR_Hold_%": br.tr_hold_pct, "TR_Swap_%": br.tr_swap_pct, "Excess_%": br.tr_excess_pct,
            "Income_Hold": br.income_hold, "Price_Hold": br.price_hold, "Roll_Hold": br.roll_hold,
            "Income_Swap": br.income_swap, "Price_Swap": br.price_swap, "Roll_Swap": br.roll_swap,
            "Realized_Today": br.realized_pl_today,
        })
    table = pd.DataFrame(rows)

    # linearized breakeven (fixed scaling)
    H = horizon / 12.0
    inc_hold_ann = _income_from_sold(df, mask)
    inc_swap_ann = proceeds_mv * TARGET_BUY_BACK_YIELD
    inc_diff = (inc_swap_ann - inc_hold_ann) * H

    dv01_hold = _dv01_from_duration(hold_dur, proceeds_mv)
    dv01_buy = _dv01_from_duration(buy_dur, proceeds_mv)
    dv01_diff = dv01_buy - dv01_hold

    realized = - float(pd.to_numeric(df.loc[mask, "loss"], errors="coerce").fillna(0.0).sum())
    roll_diff = (table.loc[0, "Roll_Swap"] - table.loc[0, "Roll_Hold"]) if not table.empty else 0.0

    breakeven_bps = None
    if abs(dv01_diff) > 1e-9:
        dy_star = (inc_diff + realized + roll_diff) / dv01_diff
        breakeven_bps = float(dy_star * 10000.0)

    return table, breakeven_bps


# ---- FUNDING OVERLAY FUNCTIONS ----

def _compute_funding_overlay_components(df, mask, proceeds_mv, horizon_months, shift_bps, hold_dur, hold_conv, buy_dur,
                                        buy_conv, funding_multiple=1.0):
    """
    Compute funding overlay: keep existing bonds, borrow to buy new bonds
    """
    H = horizon_months / 12.0
    funding_rate = _get_funding_rate()
    unwind_month = min(FUNDING_UNWIND_MONTH, horizon_months)
    unwind_years = unwind_month / 12.0

    # Standard components for held bonds
    inc_hold_annual = _income_from_sold(df, mask)
    income_hold = inc_hold_annual * H
    roll_hold = _pull_to_par_roll(df, mask, horizon_years=H)
    price_hold = _price_change_dollar(proceeds_mv, hold_dur, hold_conv, shift_bps)

    # Funding overlay components
    funding_amount = proceeds_mv * funding_multiple
    funding_cost = funding_amount * funding_rate * (unwind_years)  # Cost until unwind

    # Income from funded purchase
    funded_income = funding_amount * TARGET_BUY_BACK_YIELD * H

    # Price change on funded purchase
    funded_price = _price_change_dollar(funding_amount, buy_dur, buy_conv, shift_bps)

    # Unwind the held bonds at unwind_month (capture price appreciation)
    unwind_shift_factor = unwind_years / H  # Proportion of shift realized by unwind
    unwind_shift = shift_bps * unwind_shift_factor
    funded_unwind_gain = _price_change_dollar(proceeds_mv, hold_dur, hold_conv, unwind_shift)

    # Roll on funded buy side
    funded_roll = _pull_to_par_roll(df, mask, horizon_years=H) if TR_INCLUDE_ROLL_ON_SWAP else 0.0

    # Total returns
    tr_hold = income_hold + price_hold + roll_hold
    tr_funded = (income_hold + funded_income + funded_roll +
                 funded_price + funded_unwind_gain - funding_cost)
    tr_funded_excess = tr_funded - tr_hold

    denom = proceeds_mv if proceeds_mv > 0 else 1.0

    # Calculate breakeven funding rate
    net_benefit = funded_income + funded_price + funded_unwind_gain + funded_roll
    breakeven_rate = net_benefit / (funding_amount * unwind_years) if funding_amount * unwind_years > 0 else 0.0

    return OverlayBreakdown(
        standard=TRBreakdown(
            income_hold=income_hold, price_hold=price_hold, roll_hold=roll_hold,
            income_swap=0, price_swap=0, roll_swap=0, realized_pl_today=0,
            tr_hold_dollar=tr_hold, tr_swap_dollar=tr_hold, tr_excess_dollar=0,
            tr_hold_pct=tr_hold / denom, tr_swap_pct=tr_hold / denom, tr_excess_pct=0
        ),
        funding_cost=funding_cost,
        funded_income=funded_income,
        funded_price=funded_price,
        funded_unwind_gain=funded_unwind_gain,
        tr_funded_dollar=tr_funded,
        tr_funded_pct=tr_funded / denom,
        tr_funded_excess=tr_funded_excess,
        funding_breakeven_rate=breakeven_rate,
        time_to_profit_months=_calculate_time_to_profit(funding_amount, funded_income, funding_cost, H)
    )


def _compute_wait_then_buy_components(df, mask, proceeds_mv, horizon_months, shift_bps, wait_months, buy_dur, buy_conv):
    """
    Compute wait-then-buy: park in cash, then buy after wait period
    """
    H = horizon_months / 12.0
    W = wait_months / 12.0
    remaining_years = (horizon_months - wait_months) / 12.0

    if remaining_years <= 0:
        # Can't wait longer than horizon
        return OverlayBreakdown(standard=TRBreakdown(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    cash_rate = _get_cash_rate()

    # Income from cash while waiting
    cash_income = proceeds_mv * cash_rate * W

    # Expected rate environment when we buy
    expected_rate_change = EXPECTED_RATE_CHANGES.get(wait_months, 0)
    expected_spread_change = EXPECTED_SPREAD_CHANGES.get(wait_months, 0)

    # Effective shift includes both the parallel shift scenario AND expected rate changes
    effective_shift = shift_bps + expected_rate_change

    # Buy at new yield level after waiting
    new_buy_yield = TARGET_BUY_BACK_YIELD + (expected_rate_change + expected_spread_change) / 10000.0

    # Income from delayed purchase (at potentially better yield)
    delayed_income = proceeds_mv * new_buy_yield * remaining_years

    # Price change from purchase point to horizon (only on remaining shift)
    remaining_shift = shift_bps  # The additional shift beyond what we expected
    delayed_price = _price_change_dollar(proceeds_mv, buy_dur, buy_conv, remaining_shift)

    tr_wait = cash_income + delayed_income + delayed_price

    # Compare to buying now
    buy_now_income = proceeds_mv * TARGET_BUY_BACK_YIELD * H
    buy_now_price = _price_change_dollar(proceeds_mv, buy_dur, buy_conv, shift_bps)
    tr_buy_now = buy_now_income + buy_now_price

    wait_advantage = tr_wait - tr_buy_now

    denom = proceeds_mv if proceeds_mv > 0 else 1.0

    return OverlayBreakdown(
        standard=TRBreakdown(0, 0, 0, 0, 0, 0, 0, tr_buy_now, tr_buy_now, 0, tr_buy_now / denom, tr_buy_now / denom, 0),
        cash_income=cash_income,
        delayed_income=delayed_income,
        delayed_price=delayed_price,
        tr_wait_dollar=tr_wait,
        tr_wait_pct=tr_wait / denom,
        tr_wait_excess=wait_advantage,
        wait_advantage_bps=wait_advantage / proceeds_mv * 10000 if proceeds_mv > 0 else 0
    )


def _calculate_time_to_profit(funding_amount: float, annual_income: float, funding_cost: float,
                              horizon_years: float) -> float:
    """Calculate months until funding overlay becomes profitable"""
    if annual_income <= 0:
        return float('inf')

    monthly_net_income = (annual_income / 12.0) - (funding_cost / (horizon_years * 12.0))
    if monthly_net_income <= 0:
        return float('inf')

    # Break-even when cumulative net income > 0
    return max(0.0, 1.0)  # Simplified: assume 1 month minimum


def compute_parallel_tr_table_with_overlays(df: pd.DataFrame, mask: np.ndarray) -> Tuple[pd.DataFrame, Dict]:
    """
    Enhanced parallel TR table with funding and wait-then-buy overlays
    """
    if not (FUNDING_OVERLAY_ENABLED or WAIT_TO_BUY_ENABLED):
        # Fall back to standard analysis
        standard_table, breakeven = compute_parallel_tr_table(df, mask)
        return standard_table, {"breakeven_bps": breakeven}

    proceeds_mv = float(pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0).sum())
    hold_dur = _weighted_or_assumed(df, mask, DUR_COLS, ASSUMED_HOLD_DURATION_YEARS)
    hold_conv = _weighted_or_assumed(df, mask, CONV_COLS, ASSUMED_HOLD_CONVEXITY)
    buy_dur, buy_conv = ASSUMED_BUY_DURATION_YEARS, ASSUMED_BUY_CONVEXITY
    horizon = TR_HORIZON_MONTHS

    rows = []
    overlay_stats = {}

    for bps in TR_PARALLEL_SHIFTS_BPS:
        row = {"shift_bps": bps}

        # Standard swap analysis
        standard = _compute_tr_components_parallel(df, mask, proceeds_mv, horizon, bps, hold_dur, hold_conv, buy_dur,
                                                   buy_conv)
        row.update({
            "TR_Hold_$": standard.tr_hold_dollar,
            "TR_Swap_$": standard.tr_swap_dollar,
            "Excess_$": standard.tr_excess_dollar,
            "TR_Hold_%": standard.tr_hold_pct,
            "TR_Swap_%": standard.tr_swap_pct,
            "Excess_%": standard.tr_excess_pct,
            "Income_Hold": standard.income_hold,
            "Price_Hold": standard.price_hold,
            "Roll_Hold": standard.roll_hold,
            "Income_Swap": standard.income_swap,
            "Price_Swap": standard.price_swap,
            "Roll_Swap": standard.roll_swap,
            "Realized_Today": standard.realized_pl_today,
        })

        # Funding overlay analysis
        if FUNDING_OVERLAY_ENABLED:
            funding_overlay = _compute_funding_overlay_components(
                df, mask, proceeds_mv, horizon, bps, hold_dur, hold_conv, buy_dur, buy_conv,
                funding_multiple=FUNDING_MULTIPLIER_OPTIONS[0]  # Use first multiplier option
            )
            row.update({
                "TR_Funded_$": funding_overlay.tr_funded_dollar,
                "TR_Funded_%": funding_overlay.tr_funded_pct,
                "Funded_Excess_$": funding_overlay.tr_funded_excess,
                "Funding_Cost": funding_overlay.funding_cost,
                "Funded_Income": funding_overlay.funded_income,
                "Funded_Price": funding_overlay.funded_price,
                "Unwind_Gain": funding_overlay.funded_unwind_gain,
            })

            if bps == 0:  # Store breakeven info for base case
                overlay_stats["funding_breakeven_rate"] = funding_overlay.funding_breakeven_rate
                overlay_stats["time_to_profit"] = funding_overlay.time_to_profit_months

        # Wait-then-buy analysis
        if WAIT_TO_BUY_ENABLED:
            for wait_months in WAIT_TO_BUY_MONTHS[1:]:  # Skip 0 (buy now)
                wait_overlay = _compute_wait_then_buy_components(
                    df, mask, proceeds_mv, horizon, bps, wait_months, buy_dur, buy_conv
                )
                row.update({
                    f"TR_Wait{wait_months}M_$": wait_overlay.tr_wait_dollar,
                    f"TR_Wait{wait_months}M_%": wait_overlay.tr_wait_pct,
                    f"Wait{wait_months}M_Advantage_$": wait_overlay.tr_wait_excess,
                    f"Cash_Income_{wait_months}M": wait_overlay.cash_income,
                    f"Delayed_Income_{wait_months}M": wait_overlay.delayed_income,
                })

        rows.append(row)

    table = pd.DataFrame(rows)

    # Calculate overall breakeven
    standard_table, standard_breakeven = compute_parallel_tr_table(df, mask)
    overlay_stats["standard_breakeven_bps"] = standard_breakeven

    return table, overlay_stats


# ---- KEY-RATE TR ----
def _get_bucket_col(df: pd.DataFrame, bucket: str) -> Optional[str]:
    for cand in KEY_RATE_KRD_COLUMN_MAP.get(bucket, []):
        if cand in df.columns:
            return cand
    return None


def _bond_bucket_weights_from_maturity(years_to_maturity: float) -> Dict[str, float]:
    centers = {"2Y": 2.0, "5Y": 5.0, "10Y": 10.0, "30Y": 30.0}
    y = float(years_to_maturity) if np.isfinite(years_to_maturity) and years_to_maturity >= 0 else 5.0
    dists = {k: abs(y - v) for k, v in centers.items()}
    inv = {k: 1.0 / (d + 1e-6) for k, d in dists.items()}
    s = sum(inv.values())
    return {k: (v / s) for k, v in inv.items()}


def _portfolio_bucket_dv01s(df: pd.DataFrame, mask: np.ndarray, fallback_dur_years: float) -> Dict[str, float]:
    mv_series = pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0)
    mv_total = float(mv_series.sum())
    out = {b: 0.0 for b in KEY_RATE_BUCKETS}
    if mv_total <= 0:
        return out

    any_cols = [_get_bucket_col(df, b) for b in KEY_RATE_BUCKETS]
    if any(c is not None for c in any_cols):
        for b in KEY_RATE_BUCKETS:
            col = _get_bucket_col(df, b)
            if col is None:
                continue
            vals = pd.to_numeric(df.loc[mask, col], errors="coerce").fillna(0.0)
            # Heuristic: <= 40 → interpret as KRD years; else DV01 dollars/bp
            if (vals.abs().max() <= 40.0):
                dv01_bond = mv_series.values * vals.values / 10000.0
            else:
                dv01_bond = vals.values
            out[b] = float(np.nansum(dv01_bond))
        return out

    dur_col = _get_first_existing(df, DUR_COLS)
    if dur_col is not None:
        dur_bond = pd.to_numeric(df.loc[mask, dur_col], errors="coerce").fillna(fallback_dur_years).values
    else:
        dur_bond = np.full(mask.sum(), fallback_dur_years, dtype=float)

    mv = mv_series.values
    dv01_total_bond = mv * dur_bond / 10000.0

    ytm_like = _years_to_maturity_like(df, mask, default_years=5.0).values
    for i in range(len(dv01_total_bond)):
        w = _bond_bucket_weights_from_maturity(ytm_like[i])
        for b in KEY_RATE_BUCKETS:
            out[b] += float(dv01_total_bond[i] * w[b])
    return out


def _price_change_keyrate(dv01_dict: Dict[str, float], shocks_bps: Dict[str, float]) -> float:
    """
    Price change in dollars for bucket DV01s and bucket shocks.

    IMPORTANT UNIT FIX:
      DV01 is $ per 1 bp. If the bucket moves 'bp' bps, ΔP($) = - DV01($/bp) * bp.
    """
    s = 0.0
    for b, dv01 in dv01_dict.items():
        bp = (shocks_bps.get(b, 0.0) or 0.0)  # in basis points
        s += - dv01 * bp
    return float(s)


def compute_keyrate_tr_table(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    proceeds_mv = float(pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0).sum())
    H = TR_HORIZON_MONTHS / 12.0

    inc_hold_ann = _income_from_sold(df, mask)
    income_hold = inc_hold_ann * H
    roll_hold = _pull_to_par_roll(df, mask, horizon_years=H)
    realized_pl_today = - float(pd.to_numeric(df.loc[mask, "loss"], errors="coerce").fillna(0.0).sum())
    income_swap = float(proceeds_mv * TARGET_BUY_BACK_YIELD * H)
    roll_swap = roll_hold if TR_INCLUDE_ROLL_ON_SWAP else 0.0  # NEW

    hold_dv01s = _portfolio_bucket_dv01s(df, mask, fallback_dur_years=ASSUMED_HOLD_DURATION_YEARS)

    buy_dv01_total = _dv01_from_duration(ASSUMED_BUY_DURATION_YEARS, proceeds_mv)
    w = BUY_KRD_WEIGHTS_DEFAULT.copy()
    s = sum(w.get(b, 0.0) for b in KEY_RATE_BUCKETS)
    w = {b: (w.get(b, 0.0) / s) if s > 0 else 1.0 / len(KEY_RATE_BUCKETS) for b in KEY_RATE_BUCKETS}
    buy_dv01s = {b: buy_dv01_total * w[b] for b in KEY_RATE_BUCKETS}

    rows = []
    for sc in KEY_RATE_SHOCKS:
        name = sc["name"]
        shifts = sc["shifts_bps"]

        price_hold = _price_change_keyrate(hold_dv01s, shifts)
        price_swap = _price_change_keyrate(buy_dv01s, shifts)

        tr_hold = income_hold + price_hold + roll_hold
        tr_swap = income_swap + price_swap + roll_swap + realized_pl_today
        tr_excess = tr_swap - tr_hold
        denom = proceeds_mv if proceeds_mv > 0 else 1.0

        rows.append({
            "scenario": name,
            "TR_Hold_$": tr_hold, "TR_Swap_$": tr_swap, "Excess_$": tr_excess,
            "TR_Hold_%": tr_hold / denom, "TR_Swap_%": tr_swap / denom, "Excess_%": tr_excess / denom,
            "Income_Hold": income_hold, "Price_Hold": price_hold, "Roll_Hold": roll_hold,
            "Income_Swap": income_swap, "Price_Swap": price_swap, "Roll_Swap": roll_swap,
            "Realized_Today": realized_pl_today,
            **{f"DV01_H_{b}": hold_dv01s[b] for b in KEY_RATE_BUCKETS},
            **{f"DV01_B_{b}": buy_dv01s[b] for b in KEY_RATE_BUCKETS},
        })
    return pd.DataFrame(rows)


# === EX-ROLL VERSIONS (roll = 0 for BOTH Hold and Swap) =======================

def compute_parallel_tr_table_exroll(df: pd.DataFrame, mask: np.ndarray) -> tuple[pd.DataFrame, float | None]:
    """
    Same columns as compute_parallel_tr_table, but Roll_Hold and Roll_Swap are forced to 0.
    Breakeven computed with roll_diff = 0.
    """
    proceeds_mv = float(pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0).sum())
    hold_dur = _weighted_or_assumed(df, mask, DUR_COLS, ASSUMED_HOLD_DURATION_YEARS)
    hold_conv = _weighted_or_assumed(df, mask, CONV_COLS, ASSUMED_HOLD_CONVEXITY)
    buy_dur, buy_conv = ASSUMED_BUY_DURATION_YEARS, ASSUMED_BUY_CONVEXITY
    horizon = TR_HORIZON_MONTHS
    H = horizon / 12.0

    inc_hold_ann = _income_from_sold(df, mask)
    income_hold = inc_hold_ann * H
    realized_pl_today = - float(pd.to_numeric(df.loc[mask, "loss"], errors="coerce").fillna(0.0).sum())
    income_swap = float(proceeds_mv * TARGET_BUY_BACK_YIELD * H)

    rows = []
    for bps in TR_PARALLEL_SHIFTS_BPS:
        price_hold = _price_change_dollar(proceeds_mv, hold_dur, hold_conv, bps)
        price_swap = _price_change_dollar(proceeds_mv, buy_dur, buy_conv, bps)
        roll_hold = 0.0
        roll_swap = 0.0

        tr_hold = income_hold + price_hold + roll_hold
        tr_swap = income_swap + price_swap + roll_swap + realized_pl_today
        tr_excess = tr_swap - tr_hold
        denom = proceeds_mv if proceeds_mv > 0 else 1.0

        rows.append({
            "shift_bps": bps,
            "TR_Hold_$": tr_hold, "TR_Swap_$": tr_swap, "Excess_$": tr_excess,
            "TR_Hold_%": tr_hold / denom, "TR_Swap_%": tr_swap / denom, "Excess_%": tr_excess / denom,
            "Income_Hold": income_hold, "Price_Hold": price_hold, "Roll_Hold": 0.0,
            "Income_Swap": income_swap, "Price_Swap": price_swap, "Roll_Swap": 0.0,
            "Realized_Today": realized_pl_today,
        })
    table = pd.DataFrame(rows)

    # Breakeven (no roll terms)
    dv01_hold = _dv01_from_duration(hold_dur, proceeds_mv)
    dv01_buy = _dv01_from_duration(buy_dur, proceeds_mv)
    dv01_diff = dv01_buy - dv01_hold
    inc_diff = (proceeds_mv * TARGET_BUY_BACK_YIELD - inc_hold_ann) * H

    breakeven_bps = None
    if abs(dv01_diff) > 1e-9:
        dy_star = (inc_diff + realized_pl_today) / dv01_diff
        breakeven_bps = float(dy_star * 10000.0)

    return table, breakeven_bps


def compute_keyrate_tr_table_exroll(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """
    Same as compute_keyrate_tr_table, but Roll_Hold = Roll_Swap = 0 for every row.
    """
    proceeds_mv = float(pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0).sum())
    H = TR_HORIZON_MONTHS / 12.0

    inc_hold_ann = _income_from_sold(df, mask)
    income_hold = inc_hold_ann * H
    realized_pl_today = - float(pd.to_numeric(df.loc[mask, "loss"], errors="coerce").fillna(0.0).sum())
    income_swap = float(proceeds_mv * TARGET_BUY_BACK_YIELD * H)

    hold_dv01s = _portfolio_bucket_dv01s(df, mask, fallback_dur_years=ASSUMED_HOLD_DURATION_YEARS)

    buy_dv01_total = _dv01_from_duration(ASSUMED_BUY_DURATION_YEARS, proceeds_mv)
    w = BUY_KRD_WEIGHTS_DEFAULT.copy()
    s = sum(w.get(b, 0.0) for b in KEY_RATE_BUCKETS)
    w = {b: (w.get(b, 0.0) / s) if s > 0 else 1.0 / len(KEY_RATE_BUCKETS) for b in KEY_RATE_BUCKETS}
    buy_dv01s = {b: buy_dv01_total * w[b] for b in KEY_RATE_BUCKETS}

    rows = []
    for sc in KEY_RATE_SHOCKS:
        name = sc["name"]
        shifts = sc["shifts_bps"]

        price_hold = _price_change_keyrate(hold_dv01s, shifts)
        price_swap = _price_change_keyrate(buy_dv01s, shifts)

        roll_hold = 0.0
        roll_swap = 0.0

        tr_hold = income_hold + price_hold + roll_hold
        tr_swap = income_swap + price_swap + roll_swap + realized_pl_today
        tr_excess = tr_swap - tr_hold
        denom = proceeds_mv if proceeds_mv > 0 else 1.0

        rows.append({
            "scenario": name,
            "TR_Hold_$": tr_hold, "TR_Swap_$": tr_swap, "Excess_$": tr_excess,
            "TR_Hold_%": tr_hold / denom, "TR_Swap_%": tr_swap / denom, "Excess_%": tr_excess / denom,
            "Income_Hold": income_hold, "Price_Hold": price_hold, "Roll_Hold": 0.0,
            "Income_Swap": income_swap, "Price_Swap": price_swap, "Roll_Swap": 0.0,
            "Realized_Today": realized_pl_today,
            **{f"DV01_H_{b}": hold_dv01s[b] for b in KEY_RATE_BUCKETS},
            **{f"DV01_B_{b}": buy_dv01s[b] for b in KEY_RATE_BUCKETS},
        })
    return pd.DataFrame(rows)