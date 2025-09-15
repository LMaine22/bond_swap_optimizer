# src/tr_engine.py - Simplified Total Return Analysis
from typing import List, Optional
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
    TR_INCLUDE_ROLL_ON_SWAP,
)

# Column aliases for bond data
DUR_COLS = ["Eff Dur", "mod_duration", "duration", "dur", "mduration"]
CONV_COLS = ["Eff Convex", "convexity", "conv"]
YTM_COLS = ["Proj Yield", "Proj Yield (TE)", "YTM", "YTM (TE)", "Market Yield", "Mrkt Yield (TE)", "Acctg Yield"]


def _get_first_existing(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    """Find first existing column from a list of candidates"""
    for c in cols:
        if c in df.columns:
            return c
    return None


def _mv_weights(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Calculate market value weights for selected bonds"""
    mv = pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0).values
    tot = mv.sum()
    return (mv / tot) if tot > 0 else np.zeros_like(mv)


def _weighted_or_assumed(df: pd.DataFrame, mask: np.ndarray, cols: List[str], assumed: float) -> float:
    """Get market-weighted average of a column, or use assumed value"""
    col = _get_first_existing(df, cols)
    if col is None:
        return float(assumed)

    s = pd.to_numeric(df.loc[mask, col], errors="coerce").astype(float)
    w = _mv_weights(df, mask)

    if w.sum() <= 0 or not np.isfinite(s).any():
        return float(assumed)

    return float(np.nansum(s.values * w))


def _income_from_sold(df: pd.DataFrame, mask: np.ndarray) -> float:
    """Calculate annual income from sold bonds"""
    if "income" in df.columns:
        return float(pd.to_numeric(df.loc[mask, "income"], errors="coerce").fillna(0.0).sum())

    # Fallback: use yield columns
    ycol = _get_first_existing(df, YTM_COLS)
    if ycol is not None and "par" in df.columns:
        y = pd.to_numeric(df.loc[mask, ycol], errors="coerce").fillna(0.0)
        par = pd.to_numeric(df.loc[mask, "par"], errors="coerce").fillna(0.0)
        return float((y * par).sum())

    # Last resort: use market value and target yield
    mv = pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0).sum()
    return float(mv * TARGET_BUY_BACK_YIELD)


def _price_change_dollar(market_value: float, duration_years: float, convexity: float, shift_bps: float) -> float:
    """Calculate price change in dollars for a rate shift"""
    dy = shift_bps / 10000.0  # Convert bps to decimal
    duration_effect = -duration_years * dy
    convexity_effect = 0.5 * convexity * dy * dy
    return float(market_value * (duration_effect + convexity_effect))


def _pull_to_par_roll(df: pd.DataFrame, mask: np.ndarray, horizon_years: float) -> float:
    """Calculate pull-to-par roll effect over the horizon"""
    if "par" not in df.columns or "market" not in df.columns:
        return 0.0

    par = pd.to_numeric(df.loc[mask, "par"], errors="coerce").fillna(0.0)
    mv = pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0)
    price_gap = par - mv

    # Simple assumption: bonds roll to par over their remaining life
    # For simplicity, assume 5-year average life if no maturity data
    roll_portion = min(horizon_years / 5.0, 1.0)  # Cap at 100%

    return float((price_gap * roll_portion).sum())


def _dv01_from_duration(duration_years: float, market_value: float) -> float:
    """Calculate DV01 (dollar value of 1 basis point) from duration"""
    return float(market_value * duration_years / 10000.0)


def compute_simple_tr_analysis(df: pd.DataFrame, mask: np.ndarray) -> str:
    """
    Compute simplified total return analysis for a bond swap
    Returns a formatted text summary
    """
    try:
        proceeds_mv = float(pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0).sum())
        if proceeds_mv <= 0:
            return "No bonds selected for analysis."

        # Get portfolio characteristics
        hold_dur = _weighted_or_assumed(df, mask, DUR_COLS, ASSUMED_HOLD_DURATION_YEARS)
        hold_conv = _weighted_or_assumed(df, mask, CONV_COLS, ASSUMED_HOLD_CONVEXITY)
        buy_dur = ASSUMED_BUY_DURATION_YEARS
        buy_conv = ASSUMED_BUY_CONVEXITY
        horizon_years = TR_HORIZON_MONTHS / 12.0

        # Calculate income components
        annual_income_hold = _income_from_sold(df, mask)
        income_hold = annual_income_hold * horizon_years
        annual_income_swap = proceeds_mv * TARGET_BUY_BACK_YIELD
        income_swap = annual_income_swap * horizon_years

        # Calculate realized gain/loss
        realized_pl = -float(pd.to_numeric(df.loc[mask, "loss"], errors="coerce").fillna(0.0).sum())

        # Roll effects
        roll_hold = _pull_to_par_roll(df, mask, horizon_years)
        roll_swap = roll_hold if TR_INCLUDE_ROLL_ON_SWAP else 0.0

        # Build analysis summary
        summary = f"""TOTAL RETURN ANALYSIS
{'=' * 50}
Horizon: {TR_HORIZON_MONTHS} months ({horizon_years:.1f} years)
Swap Size: ${proceeds_mv:,.0f}

PORTFOLIO CHARACTERISTICS:
  Held Bonds Duration: {hold_dur:.2f} years
  Buy Bonds Duration: {buy_dur:.2f} years

INCOME ANALYSIS:
  Current Annual Income: ${annual_income_hold:,.0f} ({annual_income_hold / proceeds_mv:.2%})
  Enhanced Annual Income: ${annual_income_swap:,.0f} ({annual_income_swap / proceeds_mv:.2%})
  Income Pickup: ${annual_income_swap - annual_income_hold:,.0f} ({(annual_income_swap - annual_income_hold) / proceeds_mv:.2%})

HORIZON INCOME:
  Hold Path: ${income_hold:,.0f}
  Swap Path: ${income_swap:,.0f}

REALIZED GAIN/LOSS:
  Immediate Impact: ${realized_pl:,.0f}

RATE SCENARIO ANALYSIS:
"""

        # Calculate scenarios
        scenarios = []
        for shift_bps in TR_PARALLEL_SHIFTS_BPS:
            price_hold = _price_change_dollar(proceeds_mv, hold_dur, hold_conv, shift_bps)
            price_swap = _price_change_dollar(proceeds_mv, buy_dur, buy_conv, shift_bps)

            tr_hold = income_hold + price_hold + roll_hold
            tr_swap = income_swap + price_swap + roll_swap + realized_pl
            tr_excess = tr_swap - tr_hold

            scenarios.append({
                'shift': shift_bps,
                'tr_hold': tr_hold,
                'tr_swap': tr_swap,
                'tr_excess': tr_excess,
                'tr_hold_pct': tr_hold / proceeds_mv,
                'tr_swap_pct': tr_swap / proceeds_mv,
                'tr_excess_pct': tr_excess / proceeds_mv
            })

        # Add scenario table
        summary += f"{'Shift (bps)':>10} {'Hold TR $':>12} {'Swap TR $':>12} {'Excess $':>12} {'Excess %':>10}\n"
        summary += f"{'-' * 10} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 10}\n"

        for sc in scenarios:
            summary += f"{sc['shift']:>10} ${sc['tr_hold']:>11,.0f} ${sc['tr_swap']:>11,.0f} ${sc['tr_excess']:>11,.0f} {sc['tr_excess_pct']:>8.1%}\n"

        # Calculate and show breakeven
        dv01_hold = _dv01_from_duration(hold_dur, proceeds_mv)
        dv01_swap = _dv01_from_duration(buy_dur, proceeds_mv)
        dv01_diff = dv01_swap - dv01_hold

        income_diff = income_swap - income_hold
        total_pickup = income_diff + realized_pl + (roll_swap - roll_hold)

        if abs(dv01_diff) > 1e-6:
            breakeven_bps = total_pickup / dv01_diff
            summary += f"\nBREAKEVEN ANALYSIS:\n"
            summary += f"  Hold DV01: ${dv01_hold:,.0f}/bp\n"
            summary += f"  Swap DV01: ${dv01_swap:,.0f}/bp\n"
            summary += f"  Net DV01: ${dv01_diff:,.0f}/bp\n"
            summary += f"  Breakeven Shift: {breakeven_bps:.0f} bps\n"
            summary += f"  (Swap outperforms if rates move more than {abs(breakeven_bps):.0f} bps {'down' if breakeven_bps > 0 else 'up'})\n"
        else:
            summary += f"\nBREAKEVEN: Cannot calculate (similar durations)\n"

        # Summary recommendation
        positive_scenarios = sum(1 for sc in scenarios if sc['tr_excess'] > 0)
        total_scenarios = len(scenarios)

        summary += f"\nSUMMARY:\n"
        summary += f"  Scenarios favoring swap: {positive_scenarios}/{total_scenarios}\n"
        summary += f"  Income enhancement: ${income_diff:,.0f} annually\n"
        summary += f"  Recovery period: {12 * abs(realized_pl) / max(income_diff, 1):.1f} months\n"

        if positive_scenarios >= total_scenarios // 2:
            summary += f"  ✓ Swap appears attractive in most rate scenarios\n"
        else:
            summary += f"  ⚠ Swap underperforms in most rate scenarios\n"

        return summary

    except Exception as e:
        return f"Error in TR analysis: {str(e)}"


def compute_parallel_tr_table(df: pd.DataFrame, mask: np.ndarray):
    """
    Legacy compatibility function for detailed TR table
    Returns a DataFrame and breakeven point
    """
    try:
        proceeds_mv = float(pd.to_numeric(df.loc[mask, "market"], errors="coerce").fillna(0.0).sum())
        hold_dur = _weighted_or_assumed(df, mask, DUR_COLS, ASSUMED_HOLD_DURATION_YEARS)
        hold_conv = _weighted_or_assumed(df, mask, CONV_COLS, ASSUMED_HOLD_CONVEXITY)
        buy_dur = ASSUMED_BUY_DURATION_YEARS
        buy_conv = ASSUMED_BUY_CONVEXITY
        horizon_years = TR_HORIZON_MONTHS / 12.0

        # Income and other components
        annual_income_hold = _income_from_sold(df, mask)
        income_hold = annual_income_hold * horizon_years
        income_swap = proceeds_mv * TARGET_BUY_BACK_YIELD * horizon_years
        realized_pl = -float(pd.to_numeric(df.loc[mask, "loss"], errors="coerce").fillna(0.0).sum())
        roll_hold = _pull_to_par_roll(df, mask, horizon_years)
        roll_swap = roll_hold if TR_INCLUDE_ROLL_ON_SWAP else 0.0

        # Build table
        rows = []
        for shift_bps in TR_PARALLEL_SHIFTS_BPS:
            price_hold = _price_change_dollar(proceeds_mv, hold_dur, hold_conv, shift_bps)
            price_swap = _price_change_dollar(proceeds_mv, buy_dur, buy_conv, shift_bps)

            tr_hold = income_hold + price_hold + roll_hold
            tr_swap = income_swap + price_swap + roll_swap + realized_pl
            tr_excess = tr_swap - tr_hold

            denom = proceeds_mv if proceeds_mv > 0 else 1.0

            rows.append({
                "shift_bps": shift_bps,
                "TR_Hold_$": tr_hold,
                "TR_Swap_$": tr_swap,
                "Excess_$": tr_excess,
                "TR_Hold_%": tr_hold / denom,
                "TR_Swap_%": tr_swap / denom,
                "Excess_%": tr_excess / denom,
                "Income_Hold": income_hold,
                "Price_Hold": price_hold,
                "Roll_Hold": roll_hold,
                "Income_Swap": income_swap,
                "Price_Swap": price_swap,
                "Roll_Swap": roll_swap,
                "Realized_Today": realized_pl,
            })

        table = pd.DataFrame(rows)

        # Calculate breakeven
        dv01_hold = _dv01_from_duration(hold_dur, proceeds_mv)
        dv01_swap = _dv01_from_duration(buy_dur, proceeds_mv)
        dv01_diff = dv01_swap - dv01_hold

        income_diff = income_swap - income_hold
        total_pickup = income_diff + realized_pl + (roll_swap - roll_hold)

        breakeven_bps = None
        if abs(dv01_diff) > 1e-6:
            breakeven_bps = float(total_pickup / dv01_diff)

        return table, breakeven_bps

    except Exception as e:
        print(f"Error in parallel TR calculation: {e}")
        return pd.DataFrame(), None


# Simplified versions of other TR functions for compatibility
def compute_keyrate_tr_table(df: pd.DataFrame, mask: np.ndarray):
    """Placeholder for key-rate analysis - simplified version"""
    return pd.DataFrame()


def compute_parallel_tr_table_exroll(df: pd.DataFrame, mask: np.ndarray):
    """Placeholder for ex-roll analysis"""
    return pd.DataFrame(), None


def compute_keyrate_tr_table_exroll(df: pd.DataFrame, mask: np.ndarray):
    """Placeholder for key-rate ex-roll analysis"""
    return pd.DataFrame()


def compute_parallel_tr_table_with_overlays(df: pd.DataFrame, mask: np.ndarray):
    """Simplified version without overlays"""
    table, breakeven = compute_parallel_tr_table(df, mask)
    overlay_stats = {"standard_breakeven_bps": breakeven}
    return table, overlay_stats