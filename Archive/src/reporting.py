# src/reporting.py
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from src.config import (
    MIN_SWAP_SIZE_DOLLARS,
    MAX_SWAP_SIZE_DOLLARS,
    ENFORCE_MIN_SWAP_LOSS,
    MIN_TOTAL_SWAP_LOSS_DOLLARS,
    MAX_TOTAL_SWAP_LOSS_DOLLARS,
    MAX_RECOVERY_PERIOD_MONTHS,
    SOLD_WAVG_PROJ_YIELD_MAX,
)

# Try to import overlay configs, but don't fail if they don't exist
try:
    from src.config import (
        SCENARIO_ANALYSIS_ENABLED,
        RISK_ATTRIBUTION_ENABLED,
        FUNDING_OVERLAY_ENABLED,
        WAIT_TO_BUY_ENABLED,
        FUNDING_BASE_RATE_BPS,
        FUNDING_SPREAD_BPS,
        CASH_PARKING_RATE_BPS,
        WAIT_TO_BUY_MONTHS,
    )
except ImportError:
    # Fallback values if new config items don't exist
    SCENARIO_ANALYSIS_ENABLED = True
    RISK_ATTRIBUTION_ENABLED = True
    FUNDING_OVERLAY_ENABLED = False
    WAIT_TO_BUY_ENABLED = False
    FUNDING_BASE_RATE_BPS = 500
    FUNDING_SPREAD_BPS = 50
    CASH_PARKING_RATE_BPS = 475
    WAIT_TO_BUY_MONTHS = [0, 3, 6, 12]


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
        return f"${float(x):,.2f}"
    except:
        return "$0.00"


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
                    (pd.to_numeric(sel["proj_yield"], errors="coerce") * pd.to_numeric(sel["market"],
                                                                                       errors="coerce")).sum()
                    / market
                )
            except:
                proj_y = (income / market) if market > 0 else 0.0
        else:
            proj_y = (income / market) if market > 0 else 0.0

        return SwapMetrics(par, book, market, loss, income, d_inc, sold_wavg, proj_y, rec_mo, len(sel))
    except Exception as e:
        print(f"Warning: Error computing metrics: {e}")
        return SwapMetrics(0, 0, 0, 0, 0, 0, float("inf"), 0, float("inf"), 0)


def generate_scenario_dashboard(tr_table: pd.DataFrame, kr_table: pd.DataFrame, output_dir: str) -> None:
    """Create simple scenario summary"""
    if not SCENARIO_ANALYSIS_ENABLED or tr_table.empty:
        return

    try:
        # Basic parallel shift analysis
        if 'Excess_%' in tr_table.columns:
            upside_scenarios = (tr_table['Excess_%'] > 0).sum()
            total_scenarios = len(tr_table)
            best_case = tr_table['Excess_%'].max()
            worst_case = tr_table['Excess_%'].min()

            dashboard = f"""
SCENARIO ANALYSIS DASHBOARD
============================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

PARALLEL SHIFT ANALYSIS:
{upside_scenarios}/{total_scenarios} parallel scenarios favor swap
Best case excess return: {best_case:.1%}
Worst case excess return: {worst_case:.1%}
Average excess return: {tr_table['Excess_%'].mean():.1%}
"""
        else:
            dashboard = "SCENARIO ANALYSIS: No excess return data available\n"

        # Key-rate analysis if available
        if not kr_table.empty and 'Excess_%' in kr_table.columns:
            kr_positive = (kr_table['Excess_%'] > 0).sum()
            dashboard += f"""
KEY-RATE SCENARIO ANALYSIS:
{kr_positive}/{len(kr_table)} scenarios favor swap
Best: {kr_table['Excess_%'].max():.1%}
Worst: {kr_table['Excess_%'].min():.1%}
"""

        # Save dashboard
        with open(f"{output_dir}/scenario_dashboard.txt", "w") as f:
            f.write(dashboard)

    except Exception as e:
        with open(f"{output_dir}/scenario_dashboard_error.txt", "w") as f:
            f.write(f"Error generating scenario dashboard: {str(e)}")


def generate_overlay_summary(tr_table: pd.DataFrame, overlay_stats: Dict, output_dir: str) -> None:
    """Generate simple overlay summary"""
    if not (FUNDING_OVERLAY_ENABLED or WAIT_TO_BUY_ENABLED) or tr_table.empty:
        return

    try:
        summary_lines = []
        summary_lines.append("OVERLAY ANALYSIS SUMMARY")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        summary_lines.append("")

        # Current rates
        funding_rate = (FUNDING_BASE_RATE_BPS + FUNDING_SPREAD_BPS) / 10000.0
        cash_rate = CASH_PARKING_RATE_BPS / 10000.0

        summary_lines.append("CURRENT ASSUMPTIONS:")
        summary_lines.append(f"Funding Rate: {funding_rate:.2%}")
        summary_lines.append(f"Cash Rate: {cash_rate:.2%}")
        summary_lines.append("")

        # Standard analysis
        zero_shift = tr_table[tr_table['shift_bps'] == 0] if 'shift_bps' in tr_table.columns else pd.DataFrame()
        if not zero_shift.empty:
            row = zero_shift.iloc[0]
            summary_lines.append("STANDARD SWAP (0bp shift):")
            if 'TR_Hold_%' in row:
                summary_lines.append(f"Hold Return: {row['TR_Hold_%']:.1%}")
            if 'TR_Swap_%' in row:
                summary_lines.append(f"Swap Return: {row['TR_Swap_%']:.1%}")
            if 'Excess_%' in row:
                summary_lines.append(f"Advantage: {row['Excess_%']:.1%}")
            summary_lines.append("")

        # Funding overlay
        if FUNDING_OVERLAY_ENABLED and 'funding_breakeven_rate' in overlay_stats:
            breakeven = overlay_stats['funding_breakeven_rate']
            summary_lines.append("FUNDING OVERLAY:")
            summary_lines.append(f"Breakeven Rate: {breakeven:.2%}")
            summary_lines.append(f"Current Rate: {funding_rate:.2%}")
            profitable = "Profitable" if breakeven > funding_rate else "Not Profitable"
            summary_lines.append(f"Status: {profitable}")
            summary_lines.append("")

        # Wait analysis
        if WAIT_TO_BUY_ENABLED:
            summary_lines.append("WAIT-THEN-BUY:")
            for wait_months in WAIT_TO_BUY_MONTHS[1:]:
                wait_col = f"Wait{wait_months}M_Advantage_$"
                if wait_col in tr_table.columns:
                    avg_advantage = tr_table[wait_col].mean()
                    summary_lines.append(f"Wait {wait_months}M: {_fmt_money(avg_advantage)} avg advantage")
            summary_lines.append("")

        # Save summary
        with open(f"{output_dir}/tr_overlay_summary.txt", "w") as f:
            f.write("\n".join(summary_lines))

    except Exception as e:
        with open(f"{output_dir}/overlay_summary_error.txt", "w") as f:
            f.write(f"Error generating overlay summary: {str(e)}")


def analyze_risk_drivers(df: pd.DataFrame, mask: np.ndarray, output_dir: str) -> None:
    """Simple risk analysis"""
    if not RISK_ATTRIBUTION_ENABLED:
        return

    try:
        selected = df.loc[mask].copy()
        if selected.empty:
            return

        # Simple stats
        total_market = selected['market'].sum() if 'market' in selected.columns else 0
        total_loss = selected['loss'].sum() if 'loss' in selected.columns else 0
        total_income = selected['delta_income'].sum() if 'delta_income' in selected.columns else 0
        count = len(selected)

        # Top contributors
        top_loss = []
        top_income = []

        if 'loss' in selected.columns:
            top_loss_df = selected.nlargest(5, 'loss')
            for _, row in top_loss_df.iterrows():
                cusip = row.get('CUSIP', 'Unknown')
                loss = row.get('loss', 0)
                market = row.get('market', 0)
                top_loss.append({"cusip": str(cusip), "loss": float(loss), "market": float(market)})

        if 'delta_income' in selected.columns:
            top_income_df = selected.nlargest(5, 'delta_income')
            for _, row in top_income_df.iterrows():
                cusip = row.get('CUSIP', 'Unknown')
                income = row.get('delta_income', 0)
                market = row.get('market', 0)
                top_income.append({"cusip": str(cusip), "delta_income": float(income), "market": float(market)})

        # Simple attribution dict
        attribution = {
            "summary": {
                "total_positions": count,
                "total_market": float(total_market),
                "total_loss": float(total_loss),
                "total_delta_income": float(total_income)
            },
            "top_loss_contributors": top_loss,
            "top_income_contributors": top_income
        }

        # Summary text
        summary_text = f"""
RISK ATTRIBUTION ANALYSIS
==========================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

PORTFOLIO SUMMARY:
Total positions: {count}
Total market value: {_fmt_money(total_market)}
Total loss: {_fmt_money(total_loss)}
Total delta income: {_fmt_money(total_income)}

TOP LOSS CONTRIBUTORS:
"""
        for item in top_loss[:5]:
            summary_text += f"  {item['cusip']}: {_fmt_money(item['loss'])}\n"

        summary_text += "\nTOP INCOME CONTRIBUTORS:\n"
        for item in top_income[:5]:
            summary_text += f"  {item['cusip']}: {_fmt_money(item['delta_income'])}\n"

        # Save files
        with open(f"{output_dir}/risk_attribution.json", "w") as f:
            json.dump(attribution, f, indent=2)

        with open(f"{output_dir}/risk_summary.txt", "w") as f:
            f.write(summary_text)

    except Exception as e:
        with open(f"{output_dir}/risk_attribution_error.txt", "w") as f:
            f.write(f"Error generating risk attribution: {str(e)}")


def generate_enhanced_summary(metrics: SwapMetrics, tr_table: pd.DataFrame, kr_table: pd.DataFrame,
                              overlay_stats: Dict = None) -> str:
    """Generate enhanced summary"""

    base_summary = f"""
=== Bond Swap Option Summary ===

Count:               {metrics.count}
Par total:           {_fmt_money(metrics.par)}
Book total:          {_fmt_money(metrics.book)}
Market total:        {_fmt_money(metrics.market)}
Net G/L (Mkt-Book):  {_fmt_money(metrics.market - metrics.book)}
Loss (abs):          {_fmt_money(metrics.loss)}
Income (sold):       {_fmt_money(metrics.income)}
Δ-Income (to buy):   {_fmt_money(metrics.delta_income)}
Sold WAVG Proj Yld:  {_fmt_pct(metrics.sold_wavg)}
MV-Weighted Proj Y:  {_fmt_pct(metrics.proj_y)}
Recovery (months):   {metrics.recovery_months:.2f}

--- Hard Caps (for reference) ---
Proceeds window:     {_fmt_money(MIN_SWAP_SIZE_DOLLARS)} .. {_fmt_money(MAX_SWAP_SIZE_DOLLARS)}
Total loss cap:      {_fmt_money(MIN_TOTAL_SWAP_LOSS_DOLLARS) if ENFORCE_MIN_SWAP_LOSS else "no min"} .. {_fmt_money(MAX_TOTAL_SWAP_LOSS_DOLLARS)} (positive=loss)
Recovery max:        ≤ {MAX_RECOVERY_PERIOD_MONTHS:.1f} months
Sold wavg proj yld:  ≤ {_fmt_pct(SOLD_WAVG_PROJ_YIELD_MAX)}
"""

    # Add scenario summary if available
    if SCENARIO_ANALYSIS_ENABLED and not tr_table.empty and 'Excess_%' in tr_table.columns:
        upside_scenarios = (tr_table['Excess_%'] > 0).sum()
        total_scenarios = len(tr_table)
        best_case = tr_table['Excess_%'].max()
        worst_case = tr_table['Excess_%'].min()

        scenario_summary = f"""

--- Scenario Analysis Summary ---
Favorable scenarios: {upside_scenarios}/{total_scenarios} ({upside_scenarios / total_scenarios:.0%})
Best case excess return: {best_case:.1%}
Worst case excess return: {worst_case:.1%}
Expected excess return: {tr_table['Excess_%'].mean():.1%}
"""
        base_summary += scenario_summary

    # Add overlay summary if available
    if overlay_stats and (FUNDING_OVERLAY_ENABLED or WAIT_TO_BUY_ENABLED):
        overlay_summary = "\n--- Overlay Analysis Summary ---\n"

        if FUNDING_OVERLAY_ENABLED and 'funding_breakeven_rate' in overlay_stats:
            current_funding_rate = (FUNDING_BASE_RATE_BPS + FUNDING_SPREAD_BPS) / 10000.0
            breakeven_rate = overlay_stats['funding_breakeven_rate']
            overlay_summary += f"Funding overlay breakeven: {breakeven_rate:.2%} (current: {current_funding_rate:.2%})\n"
            if breakeven_rate > current_funding_rate:
                overlay_summary += "✓ Funding overlay profitable at current rates\n"
            else:
                overlay_summary += "✗ Funding overlay unprofitable at current rates\n"

        base_summary += overlay_summary

    return base_summary


def generate_report(
        best_solution: np.ndarray,
        best_score: Optional[float],
    bond_candidates: pd.DataFrame,
    full_portfolio: pd.DataFrame,
        output_dir: str,
    quiet: bool = False,
):
    """Main report generation function - simplified and robust"""

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Normalize mask
        if best_solution.dtype != bool:
            mask = best_solution.astype(bool)
        else:
            mask = best_solution

        # Guard if shapes mismatch
        if len(mask) != len(bond_candidates):
            raise ValueError("best_solution length does not match bond_candidates shape")

        selected = bond_candidates.loc[mask].copy()
        metrics = _compute_metrics(bond_candidates, mask)

        # Files
        selected_path = os.path.join(output_dir, "selected_bonds.csv")
        summary_path = os.path.join(output_dir, "summary.txt")
        metrics_path = os.path.join(output_dir, "metrics.json")
        cands_path = os.path.join(output_dir, "candidates_snapshot.csv")

        # Save selected bonds
        try:
            formatted_selected = selected.copy()

            # Format numeric columns
            numeric_cols = ['par', 'book', 'market', 'loss', 'unreal_pct', 'acctg_yield', 'proj_yield',
                            'income', 'buyback_income', 'delta_income']

            for col in numeric_cols:
                if col in formatted_selected.columns:
                    formatted_selected[col] = pd.to_numeric(formatted_selected[col], errors='coerce').round(2)

            # Format yield columns to percentages
            yield_cols = ['acctg_yield', 'proj_yield']
            for col in yield_cols:
                if col in formatted_selected.columns:
                    formatted_selected[col] = (formatted_selected[col] * 100).round(2)

            formatted_selected.to_csv(selected_path, index=False)
        except Exception as e:
            print(f"Warning: Error saving selected bonds: {e}")
            selected.to_csv(selected_path, index=False)

        # Save candidate universe
        try:
            bond_candidates.to_csv(cands_path, index=False)
        except Exception as e:
            print(f"Warning: Error saving candidates: {e}")

        # Generate summary
        summary_text = generate_enhanced_summary(metrics, pd.DataFrame(), pd.DataFrame())

        # Add top contributors if available
        if not selected.empty:
            try:
                summary_text += "\n\nTop Contributors:\n"
                if 'loss' in selected.columns:
                    top_loss = selected.nlargest(3, 'loss')
                    summary_text += "Top Loss:\n"
                    for _, r in top_loss.iterrows():
                        ident = r.get("CUSIP", "Unknown")
                        summary_text += f"  - {ident}: {_fmt_money(r.get('loss', 0))}\n"

                if 'delta_income' in selected.columns:
                    top_income = selected.nlargest(3, 'delta_income')
                    summary_text += "Top Income:\n"
                    for _, r in top_income.iterrows():
                        ident = r.get("CUSIP", "Unknown")
                        summary_text += f"  - {ident}: {_fmt_money(r.get('delta_income', 0))}\n"
            except Exception as e:
                print(f"Warning: Error adding top contributors: {e}")

        with open(summary_path, "w") as f:
            f.write(summary_text)

        # Machine-readable metrics
        try:
            formatted_metrics = {
                "par": round(metrics.par, 2),
                "book": round(metrics.book, 2),
                "market": round(metrics.market, 2),
                "loss": round(metrics.loss, 2),
                "income": round(metrics.income, 2),
                "delta_income": round(metrics.delta_income, 2),
                "sold_wavg": round(metrics.sold_wavg * 100, 2),
                "proj_y": round(metrics.proj_y * 100, 2),
                "recovery_months": round(metrics.recovery_months, 2),
                "count": int(metrics.count)
            }
            with open(metrics_path, "w") as f:
                json.dump(formatted_metrics, f, indent=2)
        except Exception as e:
            print(f"Warning: Error saving metrics: {e}")

        # Generate enhanced analysis
        analyze_risk_drivers(bond_candidates, mask, output_dir)

        if not quiet:
            print(f"Selected written: {selected_path}")
            print(f"Summary written:  {summary_path}")
            print(f"Metrics written:  {metrics_path}")

    except Exception as e:
        print(f"ERROR in generate_report: {e}")
        # Create minimal files so the process doesn't completely fail
        with open(os.path.join(output_dir, "error.txt"), "w") as f:
            f.write(f"Report generation failed: {str(e)}")


def update_summary_with_scenarios(output_dir: str, tr_table: pd.DataFrame, kr_table: pd.DataFrame,
                                  overlay_stats: Dict = None) -> None:
    """Update summary with scenario analysis"""
    if not SCENARIO_ANALYSIS_ENABLED:
        return

    try:
        summary_path = os.path.join(output_dir, "summary.txt")
        if not os.path.exists(summary_path):
            return

        # Read existing summary
        with open(summary_path, "r") as f:
            existing_summary = f.read()

        # Add scenario analysis
        addition = ""
        if not tr_table.empty and 'Excess_%' in tr_table.columns:
            upside_scenarios = (tr_table['Excess_%'] > 0).sum()
            total_scenarios = len(tr_table)
            best_case = tr_table['Excess_%'].max()
            worst_case = tr_table['Excess_%'].min()

            addition = f"""

--- SCENARIO ANALYSIS ---
Favorable scenarios: {upside_scenarios}/{total_scenarios} ({upside_scenarios / total_scenarios:.0%})
Best case: {best_case:.1%}
Worst case: {worst_case:.1%}
Average: {tr_table['Excess_%'].mean():.1%}
"""

            if overlay_stats and FUNDING_OVERLAY_ENABLED and 'funding_breakeven_rate' in overlay_stats:
                current_funding = (FUNDING_BASE_RATE_BPS + FUNDING_SPREAD_BPS) / 10000.0
                breakeven = overlay_stats['funding_breakeven_rate']
                addition += f"""
Funding overlay breakeven: {breakeven:.2%} (current: {current_funding:.2%})
"""

        if addition:
            with open(summary_path, "w") as f:
                f.write(existing_summary + addition)

    except Exception as e:
        # Fail silently - this is just enhancement
        pass