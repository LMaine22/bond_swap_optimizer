# main.py
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
    EXEC_SUMMARY_COUNT,
    REPORT_MAX_FOLDERS,
    MIN_SWAP_SIZE_DOLLARS,
    MAX_SWAP_SIZE_DOLLARS,
    ENFORCE_MIN_SWAP_LOSS,
    MIN_TOTAL_SWAP_LOSS_DOLLARS,
    MAX_TOTAL_SWAP_LOSS_DOLLARS,
    MAX_RECOVERY_PERIOD_MONTHS,
    SOLD_WAVG_PROJ_YIELD_MAX,
    TARGET_BUY_BACK_YIELD,
    TR_HORIZON_MONTHS,
    TR_PARALLEL_SHIFTS_BPS,
    MAX_PARETO_RANK_EXPORTED,  # still respected for discovery volume
    FUNDING_OVERLAY_ENABLED,  # NEW
    WAIT_TO_BUY_ENABLED,  # NEW
)
from src.data_handler import load_and_prepare_data, pre_filter_bonds
from src.reporting import (
    generate_report,
    generate_scenario_dashboard,  # NEW
    update_summary_with_scenarios,  # NEW
    generate_overlay_summary,  # NEW
)
from src.genetic_algorithm import evolve_nsga
from src.pruning import prune_passengers, compute_metrics
from src.tr_engine import (
    compute_parallel_tr_table,
    compute_keyrate_tr_table,
    compute_parallel_tr_table_exroll,
    compute_keyrate_tr_table_exroll,
    compute_parallel_tr_table_with_overlays,  # NEW
)
# NEW imports

# ----------------------- Console/format helpers -----------------------
VERBOSE_CONSOLE = True  # set False for one-liners only


def _mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True);
    return path


def _fmt_money(x: float) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"


def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x):.2%}"
    except Exception:
        return "n/a"


def _checkmark(ok: bool) -> str: return "✓" if ok else "✗"


def _range_ok(val: float, lo: float, hi: float) -> bool:
    try:
        v = float(val);
        return (v >= float(lo)) and (v <= float(hi))
    except Exception:
        return False


def _le_ok(val: float, cap: float) -> bool:
    try:
        return float(val) <= float(cap)
    except Exception:
        return False


def _jaccard_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool);
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum();
    union = np.logical_or(a, b).sum()
    return 1.0 - (inter / union if union > 0 else 0.0)


def _metrics_to_dict(m) -> dict:
    d = {
        "par": float(getattr(m, "par", 0.0)),
        "book": float(getattr(m, "book", 0.0)),
        "market": float(getattr(m, "market", 0.0)),
        "loss": float(getattr(m, "loss", 0.0)),
        "income": float(getattr(m, "income", 0.0)),
        "delta_income": float(getattr(m, "delta_income", 0.0)),
        "sold_wavg": float(getattr(m, "sold_wavg", np.inf)),
        "proj_y": float(getattr(m, "proj_y", 0.0)),
        "recovery_months": float(getattr(m, "recovery_months", np.inf)),
        "count": int(getattr(m, "count", 0)),
    }
    d["net_gl"] = d["market"] - d["book"]
    return d


def _print_option_summary(opt_id: str, mdict: dict, jaccard_dist: float | None = None, overlay_stats: dict = None):
    size_ok = _range_ok(mdict["market"], MIN_SWAP_SIZE_DOLLARS, MAX_SWAP_SIZE_DOLLARS)
    loss_ok = _le_ok(mdict["loss"], MAX_TOTAL_SWAP_LOSS_DOLLARS) and (not ENFORCE_MIN_SWAP_LOSS or mdict["loss"] >= MIN_TOTAL_SWAP_LOSS_DOLLARS)
    rec_ok = (np.isfinite(mdict["recovery_months"]) and mdict["recovery_months"] <= MAX_RECOVERY_PERIOD_MONTHS)
    sold_ok = _le_ok(mdict["sold_wavg"], SOLD_WAVG_PROJ_YIELD_MAX)

    pickup_vs_sold = TARGET_BUY_BACK_YIELD - mdict["sold_wavg"]
    pickup_vs_proj = TARGET_BUY_BACK_YIELD - mdict["proj_y"]

    header = (
        f"[{opt_id}] Δ-Income {_fmt_money(mdict['delta_income'])} | "
        f"Recovery {mdict['recovery_months']:.2f} mo | "
        f"Size {_fmt_money(mdict['market'])} | "
        f"Sold Wavg {_fmt_pct(mdict['sold_wavg'])} {_checkmark(sold_ok)}"
    )
    print("\n" + header)
    if jaccard_dist is not None:
        print(f"  Jaccard distance vs prior: {jaccard_dist:.2f}")

    print(
        f"  Proceeds (Market):         {_fmt_money(mdict['market'])}   "
        f"window: {_fmt_money(MIN_SWAP_SIZE_DOLLARS)} – {_fmt_money(MAX_SWAP_SIZE_DOLLARS)}  {_checkmark(size_ok)}\n"
        f"  Net G/L (Mkt-Book):        {_fmt_money(mdict['net_gl'])}\n"
        f"  Loss (abs):                {_fmt_money(mdict['loss'])}   cap: {_fmt_money(MIN_TOTAL_SWAP_LOSS_DOLLARS) if ENFORCE_MIN_SWAP_LOSS else 'no min'} .. {_fmt_money(MAX_TOTAL_SWAP_LOSS_DOLLARS)}  {_checkmark(loss_ok)}\n"
        f"  Recovery Period:           {mdict['recovery_months']:.2f} months   cap: ≤ {MAX_RECOVERY_PERIOD_MONTHS:.1f}  {_checkmark(rec_ok)}\n"
        f"  Sold Wavg Proj Yield:      {_fmt_pct(mdict['sold_wavg'])}   cap: ≤ {_fmt_pct(SOLD_WAVG_PROJ_YIELD_MAX)}  {_checkmark(sold_ok)}\n"
        f"  MV-Weighted Proj Yield:    {_fmt_pct(mdict['proj_y'])}\n"
        f"  Buyback (assumed):         {_fmt_pct(TARGET_BUY_BACK_YIELD)}\n"
        f"  Yield Pick-up vs Sold:     {_fmt_pct(pickup_vs_sold)}\n"
        f"  Yield Pick-up vs MV:       {_fmt_pct(pickup_vs_proj)}\n"
        f"  Total Par Sold:            {_fmt_money(mdict['par'])}\n"
        f"  Bonds in Set:              {mdict['count']}"
    )

    # NEW: Add overlay summary to console output
    if overlay_stats and (FUNDING_OVERLAY_ENABLED or WAIT_TO_BUY_ENABLED):
        print("  " + "─" * 50)
        print("  OVERLAY ANALYSIS:")

        if FUNDING_OVERLAY_ENABLED and 'funding_breakeven_rate' in overlay_stats:
            from src.config import FUNDING_BASE_RATE_BPS, FUNDING_SPREAD_BPS
            current_funding = (FUNDING_BASE_RATE_BPS + FUNDING_SPREAD_BPS) / 10000.0
            breakeven = overlay_stats['funding_breakeven_rate']
            profitable = "✓" if breakeven > current_funding else "✗"
            print(
                f"  Funding Overlay: {_fmt_pct(breakeven)} breakeven (current {_fmt_pct(current_funding)}) {profitable}")

        if WAIT_TO_BUY_ENABLED and 'best_wait_strategy' in overlay_stats:
            strategy = overlay_stats['best_wait_strategy']
            advantage = overlay_stats['best_wait_advantage']
            print(f"  Best Wait Strategy: {strategy} ({_fmt_money(advantage)} advantage)")

    print("  " + "─" * 74)


def _ensure_min_menu(exec_menu: list, pareto: list, min_k: int = 6) -> list:
    seen_masks = []
    out = list(exec_menu)
    for item in out:
        seen_masks.append(item["mask"])
    ordered = sorted(pareto, key=lambda r: (-r["metrics"].delta_income, r["metrics"].recovery_months))
    for cand in ordered:
        if len(out) >= min_k:
            break
        if all(not np.array_equal(cand["mask"], s) for s in seen_masks):
            out.append(cand);
            seen_masks.append(cand["mask"])
    return out


def _build_executive_menu_from_pareto(pareto: list, k: int, jaccard_min: float = 0.25) -> tuple[list, list[float]]:
    ordered = sorted(pareto, key=lambda r: (-r["metrics"].delta_income, r["metrics"].recovery_months))
    chosen, dists = [], []
    for cand in ordered:
        if not chosen:
            chosen.append(cand);
            dists.append(1.00)
            if len(chosen) >= k: break
            continue
        dmin = min(_jaccard_distance(cand["mask"], c["mask"]) for c in chosen)
        if dmin >= jaccard_min:
            chosen.append(cand);
            dists.append(dmin)
            if len(chosen) >= k: break
    i = 0
    while len(chosen) < k and i < len(ordered):
        cand = ordered[i]
        if all(not np.array_equal(cand["mask"], c["mask"]) for c in chosen):
            chosen.append(cand);
            dists.append(np.nan)
        i += 1
    return chosen, dists


def _build_micro_menus(source_list: list) -> dict:
    rows = []
    for item in source_list:
        m = item["metrics"]
        rows.append({"item": item, "delta_income": float(m.delta_income), "loss": float(m.loss),
                     "recovery": float(m.recovery_months)})
    df = pd.DataFrame(rows)
    micro = {}
    if df.empty: return micro
    low_loss = df.sort_values(["loss", "delta_income", "recovery"], ascending=[True, False, True]).head(3)
    micro["Low-Loss"] = [r["item"] for _, r in low_loss.iterrows()]
    fast_rec = df.sort_values(["recovery", "delta_income", "loss"], ascending=[True, False, True]).head(3)
    micro["Fast-Recovery"] = [r["item"] for _, r in fast_rec.iterrows()]
    max_dinc = df.sort_values(["delta_income", "recovery", "loss"], ascending=[False, True, True]).head(3)
    micro["Max Δ-Income"] = [r["item"] for _, r in max_dinc.iterrows()]
    return micro


def _row_ident(row: pd.Series) -> str:
    for c in ("security_id", "cusip", "ticker", "bond_id"):
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c])
    return f"idx_{row.name}"


def _extract_overlay_stats_for_console(tr_table: pd.DataFrame, overlay_stats: dict) -> dict:
    """Extract key overlay stats for console display"""
    console_stats = {}

    # Funding overlay stats
    if FUNDING_OVERLAY_ENABLED and 'funding_breakeven_rate' in overlay_stats:
        console_stats['funding_breakeven_rate'] = overlay_stats['funding_breakeven_rate']

    # Wait-then-buy stats
    if WAIT_TO_BUY_ENABLED and not tr_table.empty:
        wait_cols = [col for col in tr_table.columns if 'Wait' in col and 'Advantage' in col]
        if wait_cols:
            best_wait_strategy = None
            best_wait_advantage = float('-inf')
            for col in wait_cols:
                avg_advantage = tr_table[col].mean()
                if avg_advantage > best_wait_advantage:
                    best_wait_advantage = avg_advantage
                    best_wait_strategy = col.replace('_Advantage_$', '').replace('Wait', 'Wait ')

            if best_wait_strategy:
                console_stats['best_wait_strategy'] = best_wait_strategy
                console_stats['best_wait_advantage'] = best_wait_advantage

    return console_stats


# ------------------------------- Main --------------------------------
def main():
    random.seed(42);
    np.random.seed(42)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _mkdir(os.path.join("runs", f"swap_run_{stamp}"))
    options_dir = _mkdir(os.path.join(run_dir, "options"))

    print("\n--- Starting Bond Swap Optimizer (NSGA-II Pareto) ---")

    # NEW: Show enabled features
    enabled_features = ["NSGA-II Optimization", "Scenario Analysis"]
    if FUNDING_OVERLAY_ENABLED:
        enabled_features.append("Funding Overlay Analysis")
    if WAIT_TO_BUY_ENABLED:
        enabled_features.append("Wait-Then-Buy Analysis")
    print(f"Enabled features: {', '.join(enabled_features)}")

    full_portfolio_df = load_and_prepare_data(PORTFOLIO_FILE_PATH)
    if full_portfolio_df is None or full_portfolio_df.empty:
        print("ERROR: could not load portfolio. Exiting.");
        return

    bond_candidates_df = pre_filter_bonds(full_portfolio_df)
    if bond_candidates_df.empty:
        print("No eligible bonds after prefilter – nothing to do.");
        return

    print("Running NSGA-II optimization...")
    pareto_raw = evolve_nsga(bond_candidates_df)
    if not pareto_raw:
        print("No feasible solutions found under current hard caps. Try relaxing constraints.");
        return


    # --------- Build the full post-pruning list and assign sequential IDs ---------
    pareto_scored = []
    for cand in pareto_raw:
        rank = int(cand.get("rank", 0))
        if rank > MAX_PARETO_RANK_EXPORTED:
            continue
        raw_mask = cand["mask"].astype(bool)
        pruned_mask, audit = prune_passengers(bond_candidates_df, raw_mask)
        m = compute_metrics(bond_candidates_df, pruned_mask)
        pareto_scored.append({
            "mask": pruned_mask,
            "metrics": m,
            "rank": rank,
            "crowding_distance": float(cand.get("crowding_distance", np.nan)),
        })

    # Sort for stable presentation & ID assignment
    pareto_scored.sort(key=lambda r: (r["rank"], -float(getattr(r["metrics"], "delta_income", 0.0)),
                                      float(getattr(r["metrics"], "recovery_months", np.inf))))

    # Assign sequential option_ids to the full set
    for i, item in enumerate(pareto_scored, start=1):
        item["option_id"] = f"OPT_{i:03d}"

    print(f"Total Pareto candidates (exported ranks ≤ {MAX_PARETO_RANK_EXPORTED}): {len(pareto_scored)}")

    # ---------------- Executive menu (diverse slice) ----------------
    exec_menu, _jaccs = _build_executive_menu_from_pareto(
        pareto_scored, k=EXEC_SUMMARY_COUNT, jaccard_min=0.25
    )
    exec_menu = _ensure_min_menu(exec_menu, pareto_scored, min_k=min(6, EXEC_SUMMARY_COUNT))
    print(f"Executive picks: {len(exec_menu)} (target {EXEC_SUMMARY_COUNT})")

    # Prepare the render set and cap heavy reporting
    to_render = list(exec_menu)
    if len(to_render) > REPORT_MAX_FOLDERS:
        ordered = sorted(
            to_render,
            key=lambda r: (-float(getattr(r["metrics"], "delta_income", 0.0)),
                           float(getattr(r["metrics"], "recovery_months", np.inf)))
        )
        to_render = ordered[:REPORT_MAX_FOLDERS]
    print(f"Rendering {len(to_render)} full folders (cap {REPORT_MAX_FOLDERS})")

    # ---------------- Render loop ----------------
    rows_for_master_csv = []  # becomes pareto_all.csv (same columns as old options.csv)
    id_to_indices = {}  # option_id -> bond indices for re-render
    rendered_ids = set()  # which option_ids got folders
    prev_masks = []
    exec_menu_pruned = []

    # First, accumulate rows for ALL options (like old options.csv), folder path blank for now
    for item in pareto_scored:
        md = _metrics_to_dict(item["metrics"])
        rows_for_master_csv.append({
            "option_id": item["option_id"],
            "par": round(md["par"], 2),
            "book": round(md["book"], 2),
            "market": round(md["market"], 2),
            "net_gl": round(md["net_gl"], 2),
            "loss_abs": round(md["loss"], 2),
            "delta_income": round(md["delta_income"], 2),
            "sold_wavg_proj_y": round(md["sold_wavg"] * 100, 2),  # Convert to percentage
            "mv_weighted_proj_y": round(md["proj_y"] * 100, 2),  # Convert to percentage
            "recovery_months": round(md["recovery_months"], 2),
            "count": md["count"],
            "folder": "",  # filled for rendered ones below
        })
        id_to_indices[item["option_id"]] = [int(ix) for ix in np.where(item["mask"])[0]]

    # Now do heavy rendering for the curated set
    for item in tqdm(to_render, desc="Processing options", unit="option"):
        pruned_mask = item["mask"].astype(bool)
        final_m = item["metrics"]
        md = _metrics_to_dict(final_m)

        opt_id = item["option_id"]
        opt_dir = _mkdir(os.path.join(options_dir, opt_id))

        # Reports (PRUNED)
        generate_report(
            best_solution=pruned_mask.astype(int),
            best_score=np.nan,
            bond_candidates=bond_candidates_df,
            full_portfolio=full_portfolio_df,
            output_dir=opt_dir,
            quiet=True,
        )

        # ---------- Enhanced TR Analysis (Parallel shifts with overlays) ----------
        if FUNDING_OVERLAY_ENABLED or WAIT_TO_BUY_ENABLED:
            # Use enhanced TR analysis
            tr_table, overlay_stats = compute_parallel_tr_table_with_overlays(bond_candidates_df, pruned_mask)
            tr_table_rounded = tr_table.round(2)
            tr_table_rounded.to_csv(os.path.join(opt_dir, "tr_parallel.csv"), index=False)

            # Write enhanced TR summary
            with open(os.path.join(opt_dir, "tr_summary.txt"), "w") as f:
                f.write("=== Total Return (Hold vs Swap) – Parallel Shifts with Overlays ===\n")
                f.write(f"Horizon: {TR_HORIZON_MONTHS} months\n")
                f.write(f"Shifts (bps): {TR_PARALLEL_SHIFTS_BPS}\n")
                from src.config import TR_INCLUDE_ROLL_ON_SWAP
                f.write(f"Roll credited to swap path: {bool(TR_INCLUDE_ROLL_ON_SWAP)}\n")

                if 'standard_breakeven_bps' in overlay_stats:
                    breakeven_bps = overlay_stats['standard_breakeven_bps']
                    if breakeven_bps is not None and np.isfinite(breakeven_bps):
                        f.write(f"Standard breakeven parallel shift: {breakeven_bps:.1f} bps (Swap == Hold)\n")
                    else:
                        f.write("Standard breakeven parallel shift: n/a (ill-defined or DV01 diff ~ 0)\n")

                if FUNDING_OVERLAY_ENABLED:
                    f.write(f"\nFunding Overlay Analysis:\n")
                    from src.config import FUNDING_BASE_RATE_BPS, FUNDING_SPREAD_BPS
                    current_funding = (FUNDING_BASE_RATE_BPS + FUNDING_SPREAD_BPS) / 10000.0
                    f.write(f"Current funding rate: {current_funding:.2%}\n")
                    if 'funding_breakeven_rate' in overlay_stats:
                        f.write(f"Breakeven funding rate: {overlay_stats['funding_breakeven_rate']:.2%}\n")
                        profitable = overlay_stats['funding_breakeven_rate'] > current_funding
                        f.write(f"Funding overlay profitable: {'Yes' if profitable else 'No'}\n")

                if WAIT_TO_BUY_ENABLED:
                    f.write(f"\nWait-Then-Buy Analysis:\n")
                    from src.config import CASH_PARKING_RATE_BPS, WAIT_TO_BUY_MONTHS
                    cash_rate = CASH_PARKING_RATE_BPS / 10000.0
                    f.write(f"Cash parking rate: {cash_rate:.2%}\n")
                    f.write(f"Wait periods analyzed: {WAIT_TO_BUY_MONTHS[1:]} months\n")

            # Generate overlay summary
            generate_overlay_summary(tr_table_rounded, overlay_stats, opt_dir)

        else:
            # Use standard TR analysis
            tr_table, breakeven_bps = compute_parallel_tr_table(bond_candidates_df, pruned_mask)
            tr_table_rounded = tr_table.round(2)
            tr_table_rounded.to_csv(os.path.join(opt_dir, "tr_parallel.csv"), index=False)
            with open(os.path.join(opt_dir, "tr_summary.txt"), "w") as f:
                f.write("=== Total Return (Hold vs Swap) – Parallel Shifts ===\n")
                f.write(f"Horizon: {TR_HORIZON_MONTHS} months\n")
                f.write(f"Shifts (bps): {TR_PARALLEL_SHIFTS_BPS}\n")
                from src.config import TR_INCLUDE_ROLL_ON_SWAP
                f.write(f"Roll credited to swap path: {bool(TR_INCLUDE_ROLL_ON_SWAP)}\n")
                if breakeven_bps is not None and np.isfinite(breakeven_bps):
                    f.write(f"Breakeven parallel shift: {breakeven_bps:.1f} bps (Swap == Hold)\n")
                else:
                    f.write("Breakeven parallel shift: n/a (ill-defined or DV01 diff ~ 0)\n")
            overlay_stats = {}

        # ---------- TR (Key-rate / non-parallel) ----------
        kr_table = compute_keyrate_tr_table(bond_candidates_df, pruned_mask)
        kr_table_rounded = kr_table.round(2)
        kr_table_rounded.to_csv(os.path.join(opt_dir, "tr_keyrate.csv"), index=False)
        with open(os.path.join(opt_dir, "tr_keyrate_summary.txt"), "w") as f:
            from src.config import KEY_RATE_SHOCKS, KEY_RATE_BUCKETS, TR_INCLUDE_ROLL_ON_SWAP
            f.write("=== Total Return (Hold vs Swap) – Key-Rate Scenarios ===\n")
            f.write(f"Horizon: {TR_HORIZON_MONTHS} months\n")
            f.write(f"Roll credited to swap path: {bool(TR_INCLUDE_ROLL_ON_SWAP)}\n")
            f.write("Scenarios:\n")
            for sc in KEY_RATE_SHOCKS:
                f.write(f"  - {sc['name']}: {sc['shifts_bps']}\n")
            if not kr_table.empty:
                dv01_hold_total = float(sum(kr_table.filter(regex=r"^DV01_H_").iloc[0]))
                dv01_buy_total = float(sum(kr_table.filter(regex=r"^DV01_B_").iloc[0]))
                f.write("\nDV01 totals ($/bp):\n")
                f.write(f"  Hold total: {dv01_hold_total:,.2f}\n")
                f.write(f"  Buy  total: {dv01_buy_total:,.2f}\n")
                f.write("Per-bucket DV01 ($/bp):\n")
                for b in KEY_RATE_BUCKETS:
                    f.write(
                        f"  {b}: Hold {kr_table.iloc[0][f'DV01_H_{b}']:,.2f} | Buy {kr_table.iloc[0][f'DV01_B_{b}']:,.2f}\n")

        # ---------- TR (Parallel EX-ROLL) ----------
        tr_ex_table, breakeven_ex = compute_parallel_tr_table_exroll(bond_candidates_df, pruned_mask)

        # ---------- TR (Key-rate EX-ROLL) ----------
        kr_ex_table = compute_keyrate_tr_table_exroll(bond_candidates_df, pruned_mask)

        # NEW: Enhanced scenario analysis
        generate_scenario_dashboard(tr_table_rounded, kr_table_rounded, opt_dir)

        # NEW: Update summary with scenario insights
        update_summary_with_scenarios(opt_dir, tr_table_rounded, kr_table_rounded, overlay_stats)


        # Console output
        if prev_masks:
            jdist = min(_jaccard_distance(pruned_mask, pm) for pm in prev_masks)
        else:
            jdist = 1.00

        if VERBOSE_CONSOLE:
            # Extract overlay stats for console display
            console_overlay_stats = _extract_overlay_stats_for_console(tr_table_rounded, overlay_stats)
            _print_option_summary(opt_id, md, jaccard_dist=jdist, overlay_stats=console_overlay_stats)

        prev_masks.append(pruned_mask)
        exec_menu_pruned.append({"mask": pruned_mask, "metrics": final_m})
        rendered_ids.add(opt_id)

        # Update folder path in the master rows
        for r in rows_for_master_csv:
            if r["option_id"] == opt_id:
                r["folder"] = os.path.relpath(os.path.join(options_dir, opt_id), run_dir)
                break

    # ---- Write the single master CSV (same columns as old options.csv) ----
    pareto_df = pd.DataFrame(rows_for_master_csv).sort_values(
        ["delta_income", "recovery_months"], ascending=[False, True]
    )
    pareto_csv_path = os.path.join(run_dir, "pareto_all.csv")
    pareto_df.to_csv(pareto_csv_path, index=False)

    # Index for re-render by sequential option_id
    with open(os.path.join(run_dir, "pareto_index.json"), "w") as fp:
        json.dump({k: v for k, v in id_to_indices.items()}, fp)

    print(f"\nPareto exported: {pareto_csv_path}  [+ pareto_index.json]")
    print(f"Rendered folders: {len(rendered_ids)} under {os.path.join(run_dir, 'options')}")

    # NEW: Summary of enhanced features
    feature_summary = []

    if FUNDING_OVERLAY_ENABLED:
        overlay_count = sum(
            1 for rid in rendered_ids if os.path.exists(os.path.join(options_dir, rid, "tr_overlay_summary.txt")))
        feature_summary.append(f"Funding overlays: {overlay_count}")

    if WAIT_TO_BUY_ENABLED:
        feature_summary.append("Wait-then-buy analysis: enabled")

    # Count enhanced analysis files
    scenario_count = sum(
        1 for rid in rendered_ids if os.path.exists(os.path.join(options_dir, rid, "scenario_dashboard.txt")))
    risk_count = sum(
        1 for rid in rendered_ids if os.path.exists(os.path.join(options_dir, rid, "risk_attribution.json")))
    feature_summary.extend([f"Scenario dashboards: {scenario_count}", f"Risk attribution: {risk_count}"])

    print(f"Enhanced analysis: {', '.join(feature_summary)}")

    # ---- Optional terminal table (kept; it mirrors the CSV) ----
    print("\n--- Executive Options (table) ---")
    # Filter to only show rendered options
    rendered_df = pareto_df[pareto_df['option_id'].isin(rendered_ids)]
    for _, r in rendered_df.iterrows():
        size_ok = (MIN_SWAP_SIZE_DOLLARS <= r["market"] <= MAX_SWAP_SIZE_DOLLARS)
        sold_ok = (r["sold_wavg_proj_y"] <= SOLD_WAVG_PROJ_YIELD_MAX)
        flag = ""
        print(
            f"{r['option_id']:>10}  Δ{_fmt_money(r['delta_income']):>10}  "
            f"Rec {r['recovery_months']:.1f}m  Size {_fmt_money(r['market'])}  "
            f"Loss {_fmt_money(r['loss_abs']):>10}  Sold {_fmt_pct(r['sold_wavg_proj_y'] / 100.0)}{flag}"
        )

    print("\n--- Caps in force (hard) ---")
    print(f"Proceeds window:     {_fmt_money(MIN_SWAP_SIZE_DOLLARS)} .. {_fmt_money(MAX_SWAP_SIZE_DOLLARS)}")
    print(f"Total loss cap:      {_fmt_money(MIN_TOTAL_SWAP_LOSS_DOLLARS) if ENFORCE_MIN_SWAP_LOSS else 'no min'} .. {_fmt_money(MAX_TOTAL_SWAP_LOSS_DOLLARS)} (positive=loss)")
    print(f"Recovery max:        ≤ {MAX_RECOVERY_PERIOD_MONTHS:.1f} months")
    print(f"Sold wavg proj yld:  ≤ {_fmt_pct(SOLD_WAVG_PROJ_YIELD_MAX)}")

    # NEW: Show current market assumptions
    if FUNDING_OVERLAY_ENABLED or WAIT_TO_BUY_ENABLED:
        print("\n--- Market Assumptions ---")
        if FUNDING_OVERLAY_ENABLED:
            from src.config import FUNDING_BASE_RATE_BPS, FUNDING_SPREAD_BPS
            funding_rate = (FUNDING_BASE_RATE_BPS + FUNDING_SPREAD_BPS) / 10000.0
            print(f"Funding rate:        {_fmt_pct(funding_rate)} (Treasury + {FUNDING_SPREAD_BPS}bp)")
        if WAIT_TO_BUY_ENABLED:
            from src.config import CASH_PARKING_RATE_BPS, EXPECTED_RATE_CHANGES
            cash_rate = CASH_PARKING_RATE_BPS / 10000.0
            print(f"Cash parking rate:   {_fmt_pct(cash_rate)}")
            print(f"Expected cuts:       {EXPECTED_RATE_CHANGES}")

    print(f"\nFiles written under: {run_dir}\n")


# ------------------------------ Re-render -------------------------------------
def rerender_single(option_id: str, source_run_dir: str):
    """
    Re-render a single option folder later, using pareto_index.json saved in source_run_dir.
    Usage:
      python main.py --rerender OPT_001 --from-run runs/swap_run_YYYY-mm-dd_HH-MM-SS
    """
    idx_path = os.path.join(source_run_dir, "pareto_index.json")
    if not os.path.exists(idx_path):
        print(f"ERROR: {idx_path} not found.");
        return
    with open(idx_path, "r") as fp:
        index = json.load(fp)
    if option_id not in index:
        print(f"ERROR: option_id {option_id} not found in {idx_path}");
        return

    full_portfolio_df = load_and_prepare_data(PORTFOLIO_FILE_PATH)
    if full_portfolio_df is None or full_portfolio_df.empty:
        print("ERROR: could not load portfolio.");
        return
    bond_candidates_df = pre_filter_bonds(full_portfolio_df)
    if bond_candidates_df.empty:
        print("No eligible bonds after prefilter.");
        return

    indices = index[option_id]
    mask = np.zeros(len(bond_candidates_df), dtype=bool)
    mask[np.array(indices, dtype=int)] = True

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _mkdir(os.path.join("runs", f"rerender_{option_id}_{stamp}"))
    options_dir = _mkdir(os.path.join(run_dir, "options"))
    opt_dir = _mkdir(os.path.join(options_dir, option_id))

    pruned_mask, audit = prune_passengers(bond_candidates_df, mask)
    final_m = compute_metrics(bond_candidates_df, pruned_mask)
    generate_report(pruned_mask.astype(int), np.nan, bond_candidates_df, full_portfolio_df, output_dir=opt_dir,
                    quiet=True)

    # Enhanced TR analysis for re-render
    if FUNDING_OVERLAY_ENABLED or WAIT_TO_BUY_ENABLED:
        tr_table, overlay_stats = compute_parallel_tr_table_with_overlays(bond_candidates_df, pruned_mask)
        tr_table_rounded = tr_table.round(2)
        tr_table_rounded.to_csv(os.path.join(opt_dir, "tr_parallel.csv"), index=False)
        with open(os.path.join(opt_dir, "tr_summary.txt"), "w") as f:
            from src.config import TR_INCLUDE_ROLL_ON_SWAP
            f.write(f"Roll credited to swap path: {bool(TR_INCLUDE_ROLL_ON_SWAP)}\n")
            f.write(f"Horizon: {TR_HORIZON_MONTHS} months; Shifts: {TR_PARALLEL_SHIFTS_BPS}\n")
            if 'standard_breakeven_bps' in overlay_stats:
                breakeven_bps = overlay_stats['standard_breakeven_bps']
                f.write(f"Breakeven: {breakeven_bps if breakeven_bps is not None else 'n/a'} bps\n")
            if FUNDING_OVERLAY_ENABLED and 'funding_breakeven_rate' in overlay_stats:
                f.write(f"Funding breakeven: {overlay_stats['funding_breakeven_rate']:.2%}\n")
        generate_overlay_summary(tr_table_rounded, overlay_stats, opt_dir)
    else:
        tr_table, breakeven_bps = compute_parallel_tr_table(bond_candidates_df, pruned_mask)
        tr_table_rounded = tr_table.round(2)
        tr_table_rounded.to_csv(os.path.join(opt_dir, "tr_parallel.csv"), index=False)
        with open(os.path.join(opt_dir, "tr_summary.txt"), "w") as f:
            from src.config import TR_INCLUDE_ROLL_ON_SWAP
            f.write(f"Roll credited to swap path: {bool(TR_INCLUDE_ROLL_ON_SWAP)}\n")
            f.write(f"Horizon: {TR_HORIZON_MONTHS} months; Shifts: {TR_PARALLEL_SHIFTS_BPS}\n")
            f.write(f"Breakeven: {breakeven_bps if breakeven_bps is not None else 'n/a'} bps\n")
        overlay_stats = {}

    kr_table = compute_keyrate_tr_table(bond_candidates_df, pruned_mask)
    kr_table_rounded = kr_table.round(2)
    kr_table_rounded.to_csv(os.path.join(opt_dir, "tr_keyrate.csv"), index=False)

    # NEW: Enhanced analysis for re-render
    generate_scenario_dashboard(tr_table_rounded, kr_table_rounded, opt_dir)
    update_summary_with_scenarios(opt_dir, tr_table_rounded, kr_table_rounded, overlay_stats)


    print(f"Re-rendered {option_id} → {opt_dir}")


# --------------------------------- CLI ---------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerender", type=str, default=None, help="Option ID to re-render (e.g., OPT_001)")
    parser.add_argument("--from-run", type=str, default=None, help="Run directory containing pareto_index.json")
    args, _ = parser.parse_known_args()

    if args.rerender:
        if not args.from_run:
            print("ERROR: --from-run is required with --rerender")
        else:
            rerender_single(args.rerender, args.from_run)
    else:
        main()