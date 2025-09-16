# src/data_handler.py - Compatible with simplified system
import re
import warnings
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)
from src.config import (
    EXCEL_SHEET_NAME,
    ENFORCE_MIN_SELL_YIELD,
    MIN_SELL_ACCTG_YIELD,
    MAX_UNREALIZED_LOSS_PERCENT,
    MAX_INDIVIDUAL_LOSS_DOLLARS,
    TARGET_BUY_BACK_YIELD,
    SCENARIO_MODE,
)

REQUIRED = ["CUSIP", "Current Face", "Book Price", "Market Price"]
OPT_YIELDS = ["Acctg Yield", "Proj Yield (TE)"]

# Price parser for 32nds format (e.g., "98-24" -> 98.75)
_32_PATTERN = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*(\+)?\s*$")


def parse_price_to_decimal(val):
    """
    Convert price formats to decimal:
    - Plain decimals: 98.75, '101.125' → 98.75, 101.125
    - 32nds: '98-24' → 98.75 (98 + 24/32), '98-11+' → 98.34375 (98 + 11.5/32)
    """
    if pd.isna(val):
        return np.nan

    # Try decimal first
    try:
        return float(val)
    except (TypeError, ValueError):
        pass

    # Try 32nds format
    s = str(val).strip()
    m = _32_PATTERN.match(s)
    if not m:
        s2 = s.replace(" ", "")
        m = _32_PATTERN.match(s2)
        if not m:
            return np.nan

    whole = int(m.group(1))
    thirty_seconds = int(m.group(2))
    plus = m.group(3) is not None  # '+' means half-32nd
    frac = (thirty_seconds + (0.5 if plus else 0.0)) / 32.0
    return whole + frac


def _to_rate(series: pd.Series) -> pd.Series:
    """Convert percentage values to decimal rates"""
    s = pd.to_numeric(series, errors="coerce")
    if not isinstance(s, pd.Series):
        s = pd.Series(s, index=getattr(series, "index", None))
    # Convert percentage to decimal (e.g., 5.59 -> 0.0559)
    return s / 100.0


def load_and_prepare_data(file_path: str) -> pd.DataFrame | None:
    """
    Load bond portfolio data from parquet or Excel file and prepare for analysis
    """
    try:
        # Determine file format and load
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            print(f"Loaded {len(df)} bonds from parquet file")
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path, sheet_name=EXCEL_SHEET_NAME, engine="openpyxl")
            print(f"Loaded {len(df)} bonds from Excel file")
        else:
            print(f"ERROR: Unsupported file format: {file_path}")
            return None

    except Exception as e:
        print(f"ERROR: Failed to read {file_path}: {e}")
        return None

    # Validate required columns
    missing_cols = [col for col in REQUIRED if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return None

    # Build standardized DataFrame
    out = pd.DataFrame()
    out["CUSIP"] = df["CUSIP"].astype(str)
    out["par"] = pd.to_numeric(df["Current Face"], errors="coerce").fillna(0.0)

    # Parse bond prices (handle 32nds format)
    book_px_raw = df.get("Book Price")
    mkt_px_raw = df.get("Market Price")

    book_px = book_px_raw.apply(parse_price_to_decimal) if book_px_raw is not None else np.nan
    mkt_px = mkt_px_raw.apply(parse_price_to_decimal) if mkt_px_raw is not None else np.nan

    # Calculate dollar values from prices
    par_safe = out["par"].replace(0, np.nan)  # Avoid divide-by-zero
    out["book"] = out["par"] * book_px / 100.0
    out["market"] = out["par"] * mkt_px / 100.0

    # Calculate loss and unrealized percentage
    out["loss"] = out["book"] - out["market"]  # Positive = loss
    with np.errstate(divide="ignore", invalid="ignore"):
        out["unreal_pct"] = (out["market"] - out["book"]) / out["book"]

    # Handle yield columns
    if "Acctg Yield" in df.columns:
        out["acctg_yield"] = _to_rate(df["Acctg Yield"])
    else:
        out["acctg_yield"] = 0.0

    if "Proj Yield (TE)" in df.columns:
        out["proj_yield"] = _to_rate(df["Proj Yield (TE)"])
    else:
        out["proj_yield"] = np.nan

    # Calculate income streams
    # Current income: projected yield on par value
    out["income"] = out["proj_yield"] * out["par"]

    # Buyback income: target yield on market value (what we could earn if we buy)
    out["buyback_income"] = float(TARGET_BUY_BACK_YIELD) * out["market"]

    # Delta income: the enhancement from swapping
    out["delta_income"] = out["buyback_income"] - out["income"]

    # Preserve description for reporting
    if "Description" in df.columns:
        out["Description"] = df["Description"].astype(str)

    # Keep G/L percentage if available
    if "G/L %" in df.columns:
        out["gl_percent"] = pd.to_numeric(df["G/L %"], errors="coerce")

    # Preserve all original columns with raw__ prefix for reference
    for col in df.columns:
        out[f"raw__{col}"] = df[col]

    print(f"Data preparation complete. Total portfolio value: ${out['market'].sum():,.0f}")

    return out


def pre_filter_bonds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply pre-filtering rules to identify eligible bonds for swapping.
    This removes bonds that clearly don't meet basic criteria.
    """
    df = df.copy()
    df["Reason_Filtered"] = ""

    def add_reason(mask: pd.Series, text: str):
        """Add filtering reason to bonds that match the mask"""
        m = mask.fillna(False)
        has_reason = df["Reason_Filtered"].str.len().gt(0)
        # First reason
        df.loc[m & ~has_reason, "Reason_Filtered"] = text
        # Append if already has a reason
        df.loc[m & has_reason, "Reason_Filtered"] = df.loc[m & has_reason, "Reason_Filtered"] + "; " + text

    print(f"\nApplying pre-filters for {SCENARIO_MODE} scenario...")

    # 1. Unrealized loss percentage filter
    # Remove bonds with excessive unrealized losses
    mask_unrl = df["unreal_pct"] <= (MAX_UNREALIZED_LOSS_PERCENT / 100.0)
    add_reason(mask_unrl, f"unrealized_loss<={MAX_UNREALIZED_LOSS_PERCENT}%")

    # 2. Individual bond loss cap
    # Remove bonds with losses exceeding the per-bond limit
    mask_loss = df["loss"] > MAX_INDIVIDUAL_LOSS_DOLLARS
    add_reason(mask_loss, f"individual_loss>${MAX_INDIVIDUAL_LOSS_DOLLARS:,.0f}")

    # 3. Optional: minimum accounting yield filter
    if ENFORCE_MIN_SELL_YIELD:
        mask_ymin = df["acctg_yield"] < float(MIN_SELL_ACCTG_YIELD)
        add_reason(mask_ymin, f"acctg_yield<{MIN_SELL_ACCTG_YIELD:.2%}")

    # 4. Basic data quality filters
    # Remove bonds with missing critical data
    mask_missing_data = (
            pd.isna(df["market"]) |
            pd.isna(df["par"]) |
            (df["market"] <= 0) |
            (df["par"] <= 0)
    )
    add_reason(mask_missing_data, "missing_critical_data")

    # 5. Remove bonds with negative delta income (can't enhance income)
    #mask_negative_delta = df["delta_income"] <= 0
    #add_reason(mask_negative_delta, "negative_delta_income")

    # Generate filtered bonds report
    filtered = df[df["Reason_Filtered"] != ""].copy()
    if not filtered.empty:
        filter_cols = [
            c for c in ["CUSIP", "Description", "par", "book", "market", "loss",
                        "unreal_pct", "acctg_yield", "proj_yield", "delta_income", "Reason_Filtered"]
            if c in filtered.columns
        ]

        filtered_summary = filtered[filter_cols].copy()

        # Format for readability
        if "unreal_pct" in filtered_summary.columns:
            filtered_summary["unreal_pct"] = (filtered_summary["unreal_pct"] * 100).round(2)
        if "acctg_yield" in filtered_summary.columns:
            filtered_summary["acctg_yield"] = (filtered_summary["acctg_yield"] * 100).round(2)
        if "proj_yield" in filtered_summary.columns:
            filtered_summary["proj_yield"] = (filtered_summary["proj_yield"] * 100).round(2)

        filtered_summary.to_csv("filtered_out_bonds.csv", index=False)
        print(f"Excluded {len(filtered)} bonds (see filtered_out_bonds.csv for details)")

    # Keep eligible bonds
    kept = df[df["Reason_Filtered"] == ""].copy().reset_index(drop=True)

    if kept.empty:
        print("ERROR: No bonds passed pre-filtering!")
        return kept

    # Summary statistics
    elig_par = float(kept["par"].sum())
    elig_market = float(kept["market"].sum())
    total_delta_income = float(kept["delta_income"].sum())

    print(f"Eligible bonds: {len(kept)}")
    print(f"Eligible par value: ${elig_par:,.0f}")
    print(f"Eligible market value: ${elig_market:,.0f}")
    print(f"Total potential income enhancement: ${total_delta_income:,.0f}")

    # Scenario-specific eligibility check
    if SCENARIO_MODE == "tax_loss":
        loss_candidates = kept[kept["loss"] > 0]
        print(f"Bonds with losses (for tax harvesting): {len(loss_candidates)}")
        if len(loss_candidates) == 0:
            print("WARNING: No bonds with losses found for tax loss harvesting!")

    # Check if we have enough eligible bonds to meet minimum swap size
    max_possible_swap = elig_market
    from src.config import MIN_SWAP_SIZE_DOLLARS
    if max_possible_swap < MIN_SWAP_SIZE_DOLLARS:
        print(f"WARNING: Maximum possible swap size (${max_possible_swap:,.0f}) " +
              f"is below minimum requirement (${MIN_SWAP_SIZE_DOLLARS:,.0f})")

    return kept