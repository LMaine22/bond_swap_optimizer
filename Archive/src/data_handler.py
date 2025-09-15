import re
import pandas as pd
import numpy as np
from src.config import (
    EXCEL_SHEET_NAME,
    ENFORCE_MIN_SELL_YIELD,
    MIN_SELL_ACCTG_YIELD,
    MAX_UNREALIZED_LOSS_PERCENT,
    MAX_INDIVIDUAL_LOSS_DOLLARS,
    TARGET_BUY_BACK_YIELD,
)

REQUIRED = ["CUSIP", "Current Face", "Book Price", "Market Price"]
OPT_YIELDS = ["Acctg Yield", "Proj Yield (TE)"]

# -------------------------
# Price parser (converts 32nds to decimals)
# -------------------------
_32_PATTERN = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*(\+)?\s*$")

def parse_price_to_decimal(val):
    """
    Converts all price formats to decimal:
      - plain decimals: 98.75, '101.125' → 98.75, 101.125
      - 32nds strings: '98-24' → 98.75 (98 + 24/32)
                      '98-11+' → 98.34375 (98 + 11.5/32)
    Returns float or NaN if not parseable.
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
        # last resort: strip trailing symbols and try again
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
    s = pd.to_numeric(series, errors="coerce")
    if not isinstance(s, pd.Series):
        s = pd.Series(s, index=getattr(series, "index", None))
    # Convert percentage to decimal (e.g., 5.59 -> 0.0559)
    return s / 100.0

def load_and_prepare_data(file_path: str) -> pd.DataFrame | None:
    try:
        # Check file extension to determine format
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path, sheet_name=EXCEL_SHEET_NAME, engine="openpyxl")
        else:
            print(f"ERROR: Unsupported file format: {file_path}")
            return None
    except Exception as e:
        print(f"ERROR: failed to read {file_path}: {e}")
        return None

    for c in REQUIRED:
        if c not in df.columns:
            print(f"ERROR: missing column '{c}'. Found: {list(df.columns)}")
            return None

    out = pd.DataFrame()
    out["CUSIP"] = df["CUSIP"].astype(str)
    out["par"]   = pd.to_numeric(df["Current Face"], errors="coerce").fillna(0.0)

    # Parse prices: convert 32nds to decimals (e.g., '98-24' → 98.75, '98-11+' → 98.34375)
    book_px_raw = df.get("Book Price")
    mkt_px_raw  = df.get("Market Price")

    book_px = book_px_raw.apply(parse_price_to_decimal) if book_px_raw is not None else np.nan
    mkt_px  = mkt_px_raw.apply(parse_price_to_decimal)  if mkt_px_raw  is not None else np.nan

    # Compute values from prices
    par = out["par"].replace(0, np.nan)  # avoid divide-by-zero noise
    out["book"]   = out["par"] * book_px / 100.0
    out["market"] = out["par"] * mkt_px  / 100.0

    # If your workbook also carries explicit 'Book Value' / 'Market Value', you could
    # optionally backfill here. (Not required if parsing works.)

    # Loss & unrealized %
    out["loss"] = out["book"] - out["market"]   # positive = loss
    with np.errstate(divide="ignore", invalid="ignore"):
        out["unreal_pct"] = (out["market"] - out["book"]) / out["book"]

    # Accounting & Projected yields
    if "Acctg Yield" in df.columns:
        out["acctg_yield"] = _to_rate(df["Acctg Yield"])
    else:
        out["acctg_yield"] = 0.0

    if "Proj Yield (TE)" in df.columns:
        out["proj_yield"] = _to_rate(df["Proj Yield (TE)"])
    else:
        out["proj_yield"] = np.nan

    # Income basis: projected yield on PAR; buyback at 5% on MARKET
    out["income"]         = out["proj_yield"] * out["par"]
    out["buyback_income"] = float(TARGET_BUY_BACK_YIELD) * out["market"]
    out["delta_income"]   = out["buyback_income"] - out["income"]

    # Helpful display columns
    if "Description" in df.columns:
        out["Description"] = df["Description"].astype(str)
    if "G/L %" in df.columns:
        out["gl_percent"] = pd.to_numeric(df["G/L %"], errors="coerce")
    # Keep ALL original workbook columns so we can export them later
    for col in df.columns:
        out[f"raw__{col}"] = df[col]

    return out

def pre_filter_bonds(df: pd.DataFrame) -> pd.DataFrame:
    """Apply per-bond hard rules only (NO 5% yield screen on sells)."""
    df = df.copy()
    df["Reason_Filtered"] = ""

    def add_reason(mask: pd.Series, text: str):
        # Ensure boolean Series aligned and no NaNs
        m = mask.fillna(False)
        has = df["Reason_Filtered"].str.len().gt(0)
        # first reason
        df.loc[m & ~has, "Reason_Filtered"] = text
        # append if already has something
        df.loc[m & has, "Reason_Filtered"] = df.loc[m & has, "Reason_Filtered"] + ";" + text

    # 1) Unrealized loss % floor (e.g., keep bonds with unrealized >= -4%)
    #    -> filter out those at or below the -4% threshold
    mask_unrl = df["unreal_pct"] <= (MAX_UNREALIZED_LOSS_PERCENT / 100.0)
    add_reason(mask_unrl, f"unreal<={MAX_UNREALIZED_LOSS_PERCENT}%")

    # 2) Individual loss cap (book - market <= 40k); filter any above cap
    mask_loss = df["loss"] > MAX_INDIVIDUAL_LOSS_DOLLARS
    add_reason(mask_loss, f"loss>{MAX_INDIVIDUAL_LOSS_DOLLARS:,.0f}")

    # 3) Optional: per-bond MIN sell accounting yield (default OFF)
    if ENFORCE_MIN_SELL_YIELD:
        mask_ymin = df["acctg_yield"] < float(MIN_SELL_ACCTG_YIELD)
        add_reason(mask_ymin, f"acctg_yld<{MIN_SELL_ACCTG_YIELD:.2%}")

    # Emit filtered list
    filtered = df[df["Reason_Filtered"] != ""].copy()
    if not filtered.empty:
        cols = [c for c in ["CUSIP","Description","par","book","market","loss","unreal_pct","acctg_yield","proj_yield","Reason_Filtered"] if c in filtered.columns]
        filtered[cols].to_csv("filtered_out.csv", index=False)
        print("Generated report of excluded bonds at: 'filtered_out.csv'")

    kept = df[df["Reason_Filtered"] == ""].copy().reset_index(drop=True)
    # Helpful: show eligible par so you know if the 15–20MM floor is attainable
    elig_par = float(kept["par"].sum())
    print(f"Found {len(kept)} eligible bonds for the swap search. Eligible par: ${elig_par:,.0f}")
    return kept
