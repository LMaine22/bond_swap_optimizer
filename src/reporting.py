# src/reporting.py - Simplified Reporting
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import (
    MIN_SWAP_SIZE_DOLLARS,
    MAX_SWAP_SIZE_DOLLARS,
    ENFORCE_MIN_SWAP_LOSS,
    MIN_TOTAL_SWAP_LOSS_DOLLARS,
    MAX_TOTAL_SWAP_LOSS_DOLLARS,
    MIN_RECOVERY_PERIOD_MONTHS,
    MAX_RECOVERY_PERIOD_MONTHS,
    SOLD_WAVG_PROJ_YIELD_MAX,
    TARGET_BUY_BACK_YIELD,
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


def _fmt_money(x: float) -> str:
    try:
        return f"${float(x):,.0f}"
    except:
        return "$0"


def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x):.2%}"
    except:
        return "0.00%"


def _safe_num(x) -> float:
    try:
        return float(x)
    except:
        return 0.0


def _compute_metrics(df: pd.DataFrame, mask: np.ndarray) -> SwapMetrics:
    """Compute swap metrics from bond selection"""
    try:
        sel = df.loc[mask]
        if sel.empty:
            return SwapMetrics(0, 0, 0, 0, 0, 0, float("inf"), 0, float("inf"), 0)

        nsum = lambda c: _safe_num(pd.to_numeric(sel.get(c, 0), errors="coerce").sum())
        par = nsum("par")
        book = nsum("book")
        market = nsum("market")
        loss = nsum("loss")
        income = nsum("income")
        d_inc = nsum("delta_income")

        sold_wavg = (income / par) if par > 0 else float("inf")
        rec_mo = (12.0 * loss / d_inc) if d_inc > 0 else float("inf")

        # MV-weighted projected yield
        if "proj_yield" in sel.columns and market > 0:
            try:
                proj_y = float(
                    (pd.to_numeric(sel["proj_yield"], errors="coerce") *
                     pd.to_numeric(sel["market"], errors="coerce")).sum() / market
                )
            except:
                proj_y = (income / market) if market > 0 else 0.0
        else:
            proj_y = (income / market) if market > 0 else 0.0

        return SwapMetrics(par, book, market, loss, income, d_inc, sold_wavg, proj_y, rec_mo, len(sel))
    except Exception as e:
        print(f"Warning: Error computing metrics: {e}")
        return SwapMetrics(0, 0, 0, 0, 0, 0, float("inf"), 0, float("inf"), 0)


def generate_swap_summary(metrics: SwapMetrics) -> str:
    """Generate a clean, focused summary of the swap"""

    # Check constraint compliance
    constraints_met = []
    constraints_failed = []

    if MIN_SWAP_SIZE_DOLLARS <= metrics.market <= MAX_SWAP_SIZE_DOLLARS:
        constraints_met.append(f"✓ Size: {_fmt_money(metrics.market)}")
    else:
        constraints_failed.append(
            f"✗ Size: {_fmt_money(metrics.market)} (outside {_fmt_money(MIN_SWAP_SIZE_DOLLARS)}-{_fmt_money(MAX_SWAP_SIZE_DOLLARS)})")

    loss_ok = metrics.loss <= MAX_TOTAL_SWAP_LOSS_DOLLARS
    if ENFORCE_MIN_SWAP_LOSS:
        loss_ok = loss_ok and metrics.loss >= MIN_TOTAL_SWAP_LOSS_DOLLARS

    if loss_ok:
        constraints_met.append(f"✓ Loss: {_fmt_money(metrics.loss)}")
    else:
        if ENFORCE_MIN_SWAP_LOSS:
            constraints_failed.append(
                f"✗ Loss: {_fmt_money(metrics.loss)} (need {_fmt_money(MIN_TOTAL_SWAP_LOSS_DOLLARS)}-{_fmt_money(MAX_TOTAL_SWAP_LOSS_DOLLARS)})")
        else:
            constraints_failed.append(
                f"✗ Loss: {_fmt_money(metrics.loss)} (max {_fmt_money(MAX_TOTAL_SWAP_LOSS_DOLLARS)})")

    if MIN_RECOVERY_PERIOD_MONTHS <= metrics.recovery_months <= MAX_RECOVERY_PERIOD_MONTHS:
        constraints_met.append(f"✓ Recovery: {metrics.recovery_months:.2f} months")
    else:
        constraints_failed.append(
            f"✗ Recovery: {metrics.recovery_months:.2f} months (need {MIN_RECOVERY_PERIOD_MONTHS:.1f}-{MAX_RECOVERY_PERIOD_MONTHS:.1f})")

    if metrics.sold_wavg <= SOLD_WAVG_PROJ_YIELD_MAX:
        constraints_met.append(f"✓ Sold yield: {_fmt_pct(metrics.sold_wavg)}")
    else:
        constraints_failed.append(
            f"✗ Sold yield: {_fmt_pct(metrics.sold_wavg)} (max {_fmt_pct(SOLD_WAVG_PROJ_YIELD_MAX)})")

    # Build summary
    summary = f"""=== BOND SWAP SUMMARY ({SCENARIO_MODE.upper()} SCENARIO) ===

KEY METRICS:
  Income Enhancement: {_fmt_money(metrics.delta_income)}
  Recovery Period: {metrics.recovery_months:.2f} months
  Total Loss: {_fmt_money(metrics.loss)}
  Swap Size: {_fmt_money(metrics.market)}
  Positions: {metrics.count} bonds

FINANCIAL DETAILS:
  Total Par: {_fmt_money(metrics.par)}
  Book Value: {_fmt_money(metrics.book)}
  Market Value: {_fmt_money(metrics.market)}
  Net G/L: {_fmt_money(metrics.market - metrics.book)}

YIELD ANALYSIS:
  Current Income: {_fmt_money(metrics.income)}
  Enhanced Income: {_fmt_money(metrics.income + metrics.delta_income)}
  Sold Avg Yield: {_fmt_pct(metrics.sold_wavg)}
  Target Buy Yield: {_fmt_pct(TARGET_BUY_BACK_YIELD)}
  Yield Pickup: {_fmt_pct(TARGET_BUY_BACK_YIELD - metrics.sold_wavg)}

CONSTRAINT COMPLIANCE:"""

    for constraint in constraints_met:
        summary += f"\n  {constraint}"

    if constraints_failed:
        summary += "\n\nCONSTRAINT VIOLATIONS:"
        for constraint in constraints_failed:
            summary += f"\n  {constraint}"

    # Add scenario-specific notes
    if SCENARIO_MODE == "tax_loss":
        summary += f"""

TAX LOSS HARVESTING NOTES:
  • Recovery period of {metrics.recovery_months:.2f} months pushes gain recognition into next tax year
  • Loss of {_fmt_money(metrics.loss)} can offset gains for tax purposes
  • Minimum loss requirement: {"Met" if metrics.loss >= MIN_TOTAL_SWAP_LOSS_DOLLARS else "Not met"}
"""
    elif SCENARIO_MODE == "conservative":
        summary += f"""

CONSERVATIVE SWAP NOTES:
  • Minimizes capital loss while enhancing income
  • Quick recovery period for capital return
  • Lower risk profile suitable for conservative portfolios
"""

    return summary


def generate_bond_detail_table(df: pd.DataFrame, mask: np.ndarray) -> str:
    """Generate detailed table of selected bonds"""
    selected = df.loc[mask].copy()
    if selected.empty:
        return "No bonds selected."

    # Format key columns for display
    display_cols = []

    if "CUSIP" in selected.columns:
        display_cols.append("CUSIP")
    if "Description" in selected.columns:
        display_cols.append("Description")

    # Financial columns
    for col in ["par", "book", "market", "loss"]:
        if col in selected.columns:
            selected[f"{col}_fmt"] = selected[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "$0")
            display_cols.append(f"{col}_fmt")

    # Yield columns
    for col in ["proj_yield", "acctg_yield"]:
        if col in selected.columns:
            selected[f"{col}_fmt"] = selected[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            display_cols.append(f"{col}_fmt")

    # Income columns
    for col in ["income", "delta_income"]:
        if col in selected.columns:
            selected[f"{col}_fmt"] = selected[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "$0")
            display_cols.append(f"{col}_fmt")

    # Sort by delta income descending
    if "delta_income" in selected.columns:
        selected = selected.sort_values("delta_income", ascending=False)

    # Create table
    table = "SELECTED BONDS DETAIL:\n"
    table += "=" * 80 + "\n"

    # Add totals row first
    totals = {
        "par_fmt": f"${selected.get('par', 0).sum():,.0f}",
        "book_fmt": f"${selected.get('book', 0).sum():,.0f}",
        "market_fmt": f"${selected.get('market', 0).sum():,.0f}",
        "loss_fmt": f"${selected.get('loss', 0).sum():,.0f}",
        "income_fmt": f"${selected.get('income', 0).sum():,.0f}",
        "delta_income_fmt": f"${selected.get('delta_income', 0).sum():,.0f}",
    }

    table += "TOTALS:\n"
    table += f"  Par: {totals.get('par_fmt', 'N/A')}\n"
    table += f"  Market: {totals.get('market_fmt', 'N/A')}\n"
    table += f"  Loss: {totals.get('loss_fmt', 'N/A')}\n"
    table += f"  Delta Income: {totals.get('delta_income_fmt', 'N/A')}\n"
    table += "\n"

    # Add top contributors
    table += "TOP CONTRIBUTORS:\n"
    table += "-" * 40 + "\n"

    top_income = selected.nlargest(5, "delta_income") if "delta_income" in selected.columns else selected.head(5)
    for i, (_, row) in enumerate(top_income.iterrows(), 1):
        cusip = row.get("CUSIP", "Unknown")
        desc = row.get("Description", "")[:30] + "..." if len(str(row.get("Description", ""))) > 30 else row.get(
            "Description", "")
        delta_inc = f"${row.get('delta_income', 0):,.0f}" if pd.notna(row.get('delta_income')) else "$0"
        loss = f"${row.get('loss', 0):,.0f}" if pd.notna(row.get('loss')) else "$0"

        table += f"{i:2d}. {cusip} | {desc:<30} | Δ-Inc: {delta_inc:>10} | Loss: {loss:>10}\n"

    return table


def generate_report(
        best_solution: np.ndarray,
        best_score: Optional[float],
        bond_candidates: pd.DataFrame,
        full_portfolio: pd.DataFrame,
        output_dir: str,
        quiet: bool = False,
):
    """Generate comprehensive swap report"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Normalize mask
        mask = best_solution.astype(bool) if best_solution.dtype != bool else best_solution

        # Compute metrics
        metrics = _compute_metrics(bond_candidates, mask)
        selected = bond_candidates.loc[mask].copy()

        # Generate main summary
        summary_text = generate_swap_summary(metrics)
        summary_text += "\n\n" + generate_bond_detail_table(bond_candidates, mask)

        # Save summary
        with open(os.path.join(output_dir, "summary.txt"), "w") as f:
            f.write(summary_text)

        # Save selected bonds CSV
        if not selected.empty:
            # Clean up the selected bonds for CSV export
            export_cols = ["CUSIP", "par", "book", "market", "loss", "income", "delta_income"]
            if "Description" in selected.columns:
                export_cols.insert(1, "Description")
            if "proj_yield" in selected.columns:
                export_cols.append("proj_yield")
            if "acctg_yield" in selected.columns:
                export_cols.append("acctg_yield")

            export_data = selected[export_cols].copy()

            # Round financial columns
            for col in ["par", "book", "market", "loss", "income", "delta_income"]:
                if col in export_data.columns:
                    export_data[col] = export_data[col].round(2)

            # Convert yield columns to percentages
            # CORRECT - yields are already in decimal format
            for col in ["proj_yield", "acctg_yield"]:
                if col in export_data.columns:
                    export_data[col] = export_data[col].round(4)

            export_data.to_csv(os.path.join(output_dir, "selected_bonds.csv"), index=False)

        # Save machine-readable metrics
        metrics_dict = {
            "scenario": SCENARIO_MODE,
            "timestamp": datetime.now().isoformat(),
            "par": round(metrics.par, 2),
            "book": round(metrics.book, 2),
            "market": round(metrics.market, 2),
            "loss": round(metrics.loss, 2),
            "income": round(metrics.income, 2),
            "delta_income": round(metrics.delta_income, 2),
            "sold_wavg_pct": round(metrics.sold_wavg * 100, 4),
            "proj_y_pct": round(metrics.proj_y * 100, 4),
            "recovery_months": round(metrics.recovery_months, 2),
            "count": int(metrics.count),
            "yield_pickup_pct": round((TARGET_BUY_BACK_YIELD - metrics.sold_wavg) * 100, 4),
        }

        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f, indent=2)

        # Save candidate snapshot for reference
        bond_candidates.to_csv(os.path.join(output_dir, "candidates_snapshot.csv"), index=False)

        if not quiet:
            print(f"Report generated: {output_dir}")

    except Exception as e:
        print(f"ERROR generating report: {e}")
        # Create minimal error file
        with open(os.path.join(output_dir, "error.txt"), "w") as f:
            f.write(f"Report generation failed: {str(e)}")


def generate_simple_summary(run_dir: str, summary_data: List[Dict], scenario_mode: str):
    """Generate overall run summary"""
    try:
        summary_path = os.path.join(run_dir, "run_summary.txt")

        with open(summary_path, "w") as f:
            f.write(f"BOND SWAP OPTIMIZATION SUMMARY\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"Scenario: {scenario_mode.upper()}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Options analyzed: {len(summary_data)}\n\n")

            f.write(f"SCENARIO PARAMETERS:\n")
            f.write(f"  Size range: {_fmt_money(MIN_SWAP_SIZE_DOLLARS)} - {_fmt_money(MAX_SWAP_SIZE_DOLLARS)}\n")
            f.write(
                f"  Loss range: {_fmt_money(MIN_TOTAL_SWAP_LOSS_DOLLARS)} - {_fmt_money(MAX_TOTAL_SWAP_LOSS_DOLLARS)}\n")
            f.write(f"  Recovery range: {MIN_RECOVERY_PERIOD_MONTHS:.1f} - {MAX_RECOVERY_PERIOD_MONTHS:.1f} months\n")
            f.write(f"  Max sold yield: {_fmt_pct(SOLD_WAVG_PROJ_YIELD_MAX)}\n")
            if ENFORCE_MIN_SWAP_LOSS:
                f.write(f"  Minimum loss enforced: ${MIN_TOTAL_SWAP_LOSS_DOLLARS:,.0f}\n")
            f.write(f"\n")

            if summary_data:
                best = summary_data[0]
                f.write(f"BEST OPTION ({best['option_id']}):\n")
                f.write(f"  Income enhancement: {_fmt_money(best['delta_income'])}\n")
                f.write(f"  Recovery period: {best['recovery_months']:.2f} months\n")
                f.write(f"  Total loss: {_fmt_money(best['loss'])}\n")
                f.write(f"  Swap size: {_fmt_money(best['market_value'])}\n")
                f.write(f"  Bonds: {best['bond_count']}\n\n")

                f.write(f"ALL OPTIONS SUMMARY:\n")
                f.write(f"{'Rank':<4} {'Option':<8} {'Δ-Income':<12} {'Loss':<12} {'Recovery':<10} {'Size':<12}\n")
                f.write(f"{'-' * 4} {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 12}\n")

                for opt in summary_data:
                    f.write(f"{opt['rank']:<4} {opt['option_id']:<8} " +
                            f"{_fmt_money(opt['delta_income']):<12} " +
                            f"{_fmt_money(opt['loss']):<12} " +
                            f"{opt['recovery_months']:>8.2f}mo " +
                            f"{_fmt_money(opt['market_value']):<12}\n")

        print(f"Run summary saved: {summary_path}")

    except Exception as e:
        print(f"Warning: Could not generate run summary: {e}")