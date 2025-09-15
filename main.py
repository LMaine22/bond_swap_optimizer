# main.py - Return All Unique Options Version
import os
import random
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    PORTFOLIO_FILE_PATH,
    SCENARIO_MODE,
    MIN_SWAP_SIZE_DOLLARS,
    MAX_SWAP_SIZE_DOLLARS,
    ENFORCE_MIN_SWAP_LOSS,
    MIN_TOTAL_SWAP_LOSS_DOLLARS,
    MAX_TOTAL_SWAP_LOSS_DOLLARS,
    MIN_RECOVERY_PERIOD_MONTHS,
    MAX_RECOVERY_PERIOD_MONTHS,
    SOLD_WAVG_PROJ_YIELD_MAX,
    TARGET_BUY_BACK_YIELD,
    ENABLE_TR_ANALYSIS,
    print_scenario_summary,
    switch_scenario,
    SCENARIO_PRESETS,
)
from src.data_handler import load_and_prepare_data, pre_filter_bonds
from src.genetic_algorithm import evolve_nsga
from src.pruning import prune_passengers, compute_metrics
from src.reporting import generate_report, generate_simple_summary
from src.tr_engine import compute_simple_tr_analysis


# Console formatting helpers
def _mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


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


def _checkmark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _jaccard_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Jaccard distance between two boolean masks"""
    set1 = set(np.where(mask1)[0])
    set2 = set(np.where(mask2)[0])

    if not set1 and not set2:
        return 0.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return 1.0 - (intersection / union) if union > 0 else 0.0


def _are_masks_identical(mask1: np.ndarray, mask2: np.ndarray) -> bool:
    """Check if two masks are exactly identical"""
    return np.array_equal(mask1.astype(bool), mask2.astype(bool))


def _remove_duplicates(options_list, min_jaccard_distance=0.05):
    """
    Remove duplicate options based on bond selection similarity
    min_jaccard_distance: minimum distance required (0.05 = 95% overlap triggers removal)
    """
    unique_options = []

    for option in options_list:
        is_duplicate = False
        current_mask = option["mask"].astype(bool)

        for existing in unique_options:
            existing_mask = existing["mask"].astype(bool)

            # Check for exact duplicates first
            if _are_masks_identical(current_mask, existing_mask):
                is_duplicate = True
                break

            # Check for near-duplicates using Jaccard distance
            distance = _jaccard_distance(current_mask, existing_mask)
            if distance < min_jaccard_distance:
                # Keep the one with better delta income
                if option["metrics"].delta_income > existing["metrics"].delta_income:
                    # Replace existing with current (better option)
                    unique_options.remove(existing)
                    break
                else:
                    # Current is worse, skip it
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_options.append(option)

    return unique_options


def _print_option_summary(opt_id: str, metrics, rank: int = None):
    """Print a clean summary of an option"""
    # Check constraints
    size_ok = MIN_SWAP_SIZE_DOLLARS <= metrics.market <= MAX_SWAP_SIZE_DOLLARS
    loss_ok = metrics.loss <= MAX_TOTAL_SWAP_LOSS_DOLLARS
    if ENFORCE_MIN_SWAP_LOSS:
        loss_ok = loss_ok and metrics.loss >= MIN_TOTAL_SWAP_LOSS_DOLLARS
    recovery_ok = (MIN_RECOVERY_PERIOD_MONTHS <= metrics.recovery_months <= MAX_RECOVERY_PERIOD_MONTHS)
    yield_ok = metrics.sold_wavg <= SOLD_WAVG_PROJ_YIELD_MAX

    rank_str = f" (Rank {rank})" if rank is not None else ""
    print(
        f"\n{opt_id}{rank_str}: Δ-Income {_fmt_money(metrics.delta_income)} | Recovery {metrics.recovery_months:.2f}mo")
    print(f"  Size: {_fmt_money(metrics.market)} {_checkmark(size_ok)}")
    print(f"  Loss: {_fmt_money(metrics.loss)} {_checkmark(loss_ok)}")
    print(f"  Recovery: {metrics.recovery_months:.2f} months {_checkmark(recovery_ok)}")
    print(f"  Sold Yield: {_fmt_pct(metrics.sold_wavg)} {_checkmark(yield_ok)}")
    print(f"  Bonds: {metrics.count}")


def _analyze_scenarios():
    """Analyze and compare different scenarios"""
    print("\n" + "=" * 60)
    print("SCENARIO COMPARISON ANALYSIS")
    print("=" * 60)

    current_mode = SCENARIO_MODE

    for scenario_name, config in SCENARIO_PRESETS.items():
        print(f"\n{scenario_name.upper()}:")
        print(f"  {config['description']}")
        print(f"  Loss Range: ${config['min_loss']:,.0f} - ${config['max_loss']:,.0f}")
        print(f"  Recovery: {config['min_recovery_months']:.1f} - {config['max_recovery_months']:.1f} months")
        if config['enforce_min_loss']:
            print(f"  ⚠️  Requires minimum ${config['min_loss']:,.0f} loss for tax purposes")
        if config['min_recovery_months'] > 0:
            print(f"  ⚠️  Must exceed {config['min_recovery_months']:.1f} months for tax timing")


def run_scenario_comparison(df: pd.DataFrame):
    """Run optimization for multiple scenarios and compare"""
    print("\n" + "=" * 60)
    print("RUNNING SCENARIO COMPARISON")
    print("=" * 60)

    results = {}
    original_mode = SCENARIO_MODE

    for scenario_name in ["conservative", "tax_loss"]:
        print(f"\n--- Running {scenario_name.upper()} scenario ---")
        switch_scenario(scenario_name)
        print_scenario_summary()

        # Run optimization
        candidates_df = pre_filter_bonds(df)
        if candidates_df.empty:
            print(f"No eligible bonds for {scenario_name} scenario")
            continue

        pareto_raw = evolve_nsga(candidates_df)
        if not pareto_raw:
            print(f"No feasible solutions for {scenario_name} scenario")
            continue

        # Get best option
        best_option = pareto_raw[0]
        pruned_mask, _ = prune_passengers(candidates_df, best_option["mask"])
        best_metrics = compute_metrics(candidates_df, pruned_mask)

        results[scenario_name] = {
            "metrics": best_metrics,
            "mask": pruned_mask,
            "candidates": len(candidates_df),
            "solutions_found": len(pareto_raw)
        }

        print(f"Best option: Δ-Income {_fmt_money(best_metrics.delta_income)}, " +
              f"Loss {_fmt_money(best_metrics.loss)}, " +
              f"Recovery {best_metrics.recovery_months:.2f}mo")

    # Restore original mode
    switch_scenario(original_mode)

    # Print comparison
    print("\n" + "=" * 60)
    print("SCENARIO COMPARISON RESULTS")
    print("=" * 60)

    if "conservative" in results and "tax_loss" in results:
        cons = results["conservative"]["metrics"]
        tax = results["tax_loss"]["metrics"]

        print(f"\nCONSERVATIVE vs TAX LOSS:")
        print(f"  Δ-Income:    {_fmt_money(cons.delta_income):>12} vs {_fmt_money(tax.delta_income):>12}")
        print(f"  Loss:        {_fmt_money(cons.loss):>12} vs {_fmt_money(tax.loss):>12}")
        print(f"  Recovery:    {cons.recovery_months:>9.1f}mo vs {tax.recovery_months:>9.1f}mo")
        print(f"  Size:        {_fmt_money(cons.market):>12} vs {_fmt_money(tax.market):>12}")

        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if tax.delta_income > cons.delta_income * 1.2:  # 20% more income
            print(f"  → Tax loss swap provides {_fmt_money(tax.delta_income - cons.delta_income)} more income")
        if cons.loss < tax.loss * 0.5:  # Half the loss
            print(f"  → Conservative swap reduces loss by {_fmt_money(tax.loss - cons.loss)}")

    return results


def main():
    """Main entry point - Return ALL unique options"""
    random.seed(42)
    np.random.seed(42)

    # Create output directory
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _mkdir(os.path.join("runs", f"swap_run_{stamp}"))

    print("\n" + "=" * 60)
    print("BOND SWAP OPTIMIZER - ALL OPTIONS VERSION")
    print("=" * 60)

    # Show current scenario
    print_scenario_summary()

    # Load data
    print("Loading portfolio data...")
    full_portfolio_df = load_and_prepare_data(PORTFOLIO_FILE_PATH)
    if full_portfolio_df is None or full_portfolio_df.empty:
        print("ERROR: Could not load portfolio data")
        return

    # Apply filters
    print("Filtering eligible bonds...")
    bond_candidates_df = pre_filter_bonds(full_portfolio_df)
    if bond_candidates_df.empty:
        print("ERROR: No eligible bonds after filtering")
        return

    print(f"Found {len(bond_candidates_df)} eligible bonds")
    print(f"Total eligible par: {_fmt_money(bond_candidates_df['par'].sum())}")

    # Run optimization
    print(f"\nRunning optimization for {SCENARIO_MODE.upper()} scenario...")
    pareto_raw = evolve_nsga(bond_candidates_df)
    if not pareto_raw:
        print("No feasible solutions found. Try relaxing constraints.")
        return

    print(f"Found {len(pareto_raw)} raw Pareto solutions")

    # Process ALL solutions with pruning
    print("Processing and pruning all solutions...")
    all_processed_options = []

    for i, option in enumerate(tqdm(pareto_raw, desc="Processing solutions")):
        raw_mask = option["mask"].astype(bool)

        try:
            pruned_mask, audit = prune_passengers(bond_candidates_df, raw_mask)
            metrics = compute_metrics(bond_candidates_df, pruned_mask)

            # Only keep if still feasible after pruning
            if (MIN_SWAP_SIZE_DOLLARS <= metrics.market <= MAX_SWAP_SIZE_DOLLARS and
                    metrics.loss <= MAX_TOTAL_SWAP_LOSS_DOLLARS and
                    metrics.recovery_months >= MIN_RECOVERY_PERIOD_MONTHS and
                    metrics.recovery_months <= MAX_RECOVERY_PERIOD_MONTHS and
                    metrics.sold_wavg <= SOLD_WAVG_PROJ_YIELD_MAX and
                    metrics.delta_income > 0):

                if ENFORCE_MIN_SWAP_LOSS and metrics.loss < MIN_TOTAL_SWAP_LOSS_DOLLARS:
                    continue

                all_processed_options.append({
                    "option_id": f"OPT_{i + 1:03d}",
                    "mask": pruned_mask,
                    "metrics": metrics,
                    "original_rank": i + 1,
                    "pruning_summary": audit.get("pruning_summary", "")
                })
        except Exception as e:
            print(f"Warning: Error processing option {i + 1}: {e}")
            continue

    print(f"Feasible solutions after pruning: {len(all_processed_options)}")

    # Remove duplicates and near-duplicates
    print("Removing duplicates...")
    unique_options = _remove_duplicates(all_processed_options, min_jaccard_distance=0.05)
    print(f"Unique solutions after deduplication: {len(unique_options)}")

    # Sort by delta income (primary) then recovery time (secondary)
    unique_options.sort(key=lambda x: (-x["metrics"].delta_income, x["metrics"].recovery_months))

    # Reassign clean option IDs
    for i, option in enumerate(unique_options):
        option["option_id"] = f"OPT_{i + 1:03d}"
        option["final_rank"] = i + 1

    # Display all results
    print(f"\n" + "=" * 60)
    print(f"ALL {len(unique_options)} UNIQUE OPTIONS - {SCENARIO_MODE.upper()} SCENARIO")
    print("=" * 60)

    # Show summary table first
    print("\nSUMMARY TABLE:")
    print(f"{'Rank':<4} {'Option':<8} {'Δ-Income':<12} {'Loss':<12} {'Recovery':<10} {'Size':<12} {'Bonds':<6}")
    print("-" * 70)

    for option in unique_options:
        m = option["metrics"]
        print(f"{option['final_rank']:<4} {option['option_id']:<8} " +
              f"{_fmt_money(m.delta_income):<12} " +
              f"{_fmt_money(m.loss):<12} " +
              f"{m.recovery_months:>8.2f}mo " +
              f"{_fmt_money(m.market):<12} " +
              f"{m.count:<6}")

    # Generate detailed reports for ALL options (up to reasonable limit)
    options_dir = _mkdir(os.path.join(run_dir, "options"))

    # Set a reasonable limit for detailed reports to avoid excessive file generation
    max_detailed_reports = min(50, len(unique_options))  # Cap at 50 detailed reports

    print(f"\nGenerating detailed reports for top {max_detailed_reports} options...")
    summary_data = []

    for i, option in enumerate(tqdm(unique_options[:max_detailed_reports], desc="Generating reports")):
        opt_id = option["option_id"]
        opt_dir = _mkdir(os.path.join(options_dir, opt_id))

        # Generate main report
        generate_report(
            best_solution=option["mask"].astype(int),
            best_score=None,
            bond_candidates=bond_candidates_df,
            full_portfolio=full_portfolio_df,
            output_dir=opt_dir,
            quiet=True,
        )

        # Generate simple TR analysis if enabled
        if ENABLE_TR_ANALYSIS:
            tr_summary = compute_simple_tr_analysis(bond_candidates_df, option["mask"])
            with open(os.path.join(opt_dir, "tr_analysis.txt"), "w") as f:
                f.write(tr_summary)

        # Add to summary
        m = option["metrics"]
        summary_data.append({
            "option_id": opt_id,
            "rank": option["final_rank"],
            "original_rank": option["original_rank"],
            "delta_income": round(m.delta_income, 0),
            "loss": round(m.loss, 0),
            "recovery_months": round(m.recovery_months, 2),
            "market_value": round(m.market, 0),
            "sold_yield": round(m.sold_wavg * 100, 2),
            "bond_count": m.count,
            "folder": os.path.relpath(opt_dir, run_dir),
            "pruning_summary": option["pruning_summary"]
        })

    # Add remaining options to summary (without detailed folders)
    for option in unique_options[max_detailed_reports:]:
        m = option["metrics"]
        summary_data.append({
            "option_id": option["option_id"],
            "rank": option["final_rank"],
            "original_rank": option["original_rank"],
            "delta_income": round(m.delta_income, 0),
            "loss": round(m.loss, 0),
            "recovery_months": round(m.recovery_months, 2),
            "market_value": round(m.market, 0),
            "sold_yield": round(m.sold_wavg * 100, 2),
            "bond_count": m.count,
            "folder": "",  # No detailed report folder
            "pruning_summary": option["pruning_summary"]
        })

    # Save comprehensive summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(run_dir, "all_options_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Generate overall summary
    generate_simple_summary(run_dir, summary_data, SCENARIO_MODE)

    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total unique options found: {len(unique_options)}")
    print(f"Detailed reports generated: {min(max_detailed_reports, len(unique_options))}")
    print(f"Results saved to: {run_dir}")
    print(f"Complete summary CSV: {summary_csv}")
    print(f"Detailed reports: {options_dir}")

    # Show constraints reminder
    print(f"\nCONSTRAINTS APPLIED ({SCENARIO_MODE.upper()}):")
    print(f"  Swap size: {_fmt_money(MIN_SWAP_SIZE_DOLLARS)} - {_fmt_money(MAX_SWAP_SIZE_DOLLARS)}")
    print(f"  Loss range: {_fmt_money(MIN_TOTAL_SWAP_LOSS_DOLLARS)} - {_fmt_money(MAX_TOTAL_SWAP_LOSS_DOLLARS)}")
    print(f"  Recovery: {MIN_RECOVERY_PERIOD_MONTHS:.1f} - {MAX_RECOVERY_PERIOD_MONTHS:.1f} months")
    print(f"  Max sold yield: {_fmt_pct(SOLD_WAVG_PROJ_YIELD_MAX)}")
    if ENFORCE_MIN_SWAP_LOSS:
        print(f"  ⚠️  Minimum loss enforced for tax purposes")
    if MIN_RECOVERY_PERIOD_MONTHS > 0:
        print(f"  ⚠️  Recovery must exceed {MIN_RECOVERY_PERIOD_MONTHS:.1f} months")

    # Show diversity statistics
    if len(unique_options) > 1:
        print(f"\nDIVERSITY STATISTICS:")
        jaccard_distances = []
        for i in range(min(10, len(unique_options))):
            for j in range(i + 1, min(10, len(unique_options))):
                dist = _jaccard_distance(unique_options[i]["mask"], unique_options[j]["mask"])
                jaccard_distances.append(dist)

        if jaccard_distances:
            print(f"  Average Jaccard distance (top 10): {np.mean(jaccard_distances):.3f}")
            print(f"  Min/Max distances: {np.min(jaccard_distances):.3f} / {np.max(jaccard_distances):.3f}")


def cli_main():
    """CLI entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Bond Swap Optimizer - All Options Version")
    parser.add_argument("--scenario", choices=list(SCENARIO_PRESETS.keys()),
                        help="Scenario type to run")
    parser.add_argument("--compare", action="store_true",
                        help="Compare conservative vs tax_loss scenarios")
    parser.add_argument("--analyze-scenarios", action="store_true",
                        help="Show scenario configurations without running optimization")

    args = parser.parse_args()

    if args.analyze_scenarios:
        _analyze_scenarios()
        return

    if args.scenario:
        if switch_scenario(args.scenario):
            print(f"Switched to {args.scenario} scenario")
        else:
            return

    if args.compare:
        # Load data once for comparison
        full_portfolio_df = load_and_prepare_data(PORTFOLIO_FILE_PATH)
        if full_portfolio_df is None or full_portfolio_df.empty:
            print("ERROR: Could not load portfolio data")
            return

        run_scenario_comparison(full_portfolio_df)
        return

    # Run normal optimization
    main()


if __name__ == "__main__":
    cli_main()