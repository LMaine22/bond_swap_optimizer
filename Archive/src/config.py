# src/config.py
# ---------------------- CONFIGURATION ----------------------

# ----- Helpers -----
def _pct(x: float) -> float:
    """Allow 5 => 0.05, 0.05 => 0.05."""
    try:
        x = float(x)
    except Exception:
        return 0.0
    return x / 100.0 if x > 1.0 else x

# ----- File Paths -----
PORTFOLIO_FILE_PATH = "data/Sample_v02.parquet"
EXCEL_SHEET_NAME    = "Sheet"

# ----- Individual Bond Rules (SELL candidates prefilter) -----
ENFORCE_MIN_SELL_YIELD   = False
MIN_SELL_ACCTG_YIELD_RAW = 0
MIN_SELL_ACCTG_YIELD     = _pct(MIN_SELL_ACCTG_YIELD_RAW)

MAX_UNREALIZED_LOSS_PERCENT = -9.0
MAX_INDIVIDUAL_LOSS_DOLLARS = 70_000

# ----- Swap (hard caps) -----
MIN_SWAP_SIZE_DOLLARS       = 7_500_000
MAX_SWAP_SIZE_DOLLARS       = 100_000_000
ENFORCE_MIN_SWAP_LOSS       = False  # Whether to enforce minimum loss constraint
MIN_TOTAL_SWAP_LOSS_DOLLARS = 100_000      # Minimum loss required (only used if ENFORCE_MIN_SWAP_LOSS=True)
MAX_TOTAL_SWAP_LOSS_DOLLARS = 1_000_000
MAX_RECOVERY_PERIOD_MONTHS  = 3.5

# Percent knobs (type 5 for 5%)
SOLD_WAVG_PROJ_YIELD_MAX_RAW = 3.9  # Effectively disables constraint for tax loss swaps
TARGET_BUY_BACK_YIELD_RAW    = 5

SOLD_WAVG_PROJ_YIELD_MAX = _pct(SOLD_WAVG_PROJ_YIELD_MAX_RAW)
TARGET_BUY_BACK_YIELD    = _pct(TARGET_BUY_BACK_YIELD_RAW)

# ----- Menu -----
MENU_NUM_OPTIONS = 10
VERBOSE_CONSOLE = True

# ----- NSGA-II (Pareto) -----
GA_SEED           = 49
GA_POP_SIZE       = 1000
GA_GENERATIONS    = 200
GA_MUTATION_RATE  = 0.25
GA_CROSSOVER_RATE = 0.6

# ----- Total Return (parallel) -----
TR_HORIZON_MONTHS = 12
TR_PARALLEL_SHIFTS_BPS = [-100, -50, 0, 50]

ASSUMED_HOLD_DURATION_YEARS = 5.0
ASSUMED_HOLD_CONVEXITY      = 0.0
ASSUMED_BUY_DURATION_YEARS  = 5.0
ASSUMED_BUY_CONVEXITY       = 0.0

# NEW: optional fairness toggle — give roll to swap path too
TR_INCLUDE_ROLL_ON_SWAP = True # set True to credit roll on the buy side as well

# ----- Total Return (key-rate / non-parallel) -----
KEY_RATE_BUCKETS = ["2Y", "5Y", "10Y", "30Y"]

# If you someday add explicit KRD/DV01 columns, list them here. If not present, we fall back automatically.
KEY_RATE_KRD_COLUMN_MAP = {
    "2Y":  ["krd_2y", "dv01_2y"],
    "5Y":  ["krd_5y", "dv01_5y"],
    "10Y": ["krd_10y","dv01_10y"],
    "30Y": ["krd_30y","dv01_30y"],
}

KEY_RATE_SHOCKS = [
    {"name": "Bull_Steepener",   "shifts_bps": {"2Y": -75, "5Y": -50, "10Y": -35, "30Y": -25}},
    {"name": "Bull_Flattener",   "shifts_bps": {"2Y": -25, "5Y": -40, "10Y": -55, "30Y": -75}},
    {"name": "Bear_Steepener",   "shifts_bps": {"2Y": +25, "5Y": +35, "10Y": +50, "30Y": +75}},
    {"name": "Bear_Flattener",   "shifts_bps": {"2Y": +75, "5Y": +50, "10Y": +35, "30Y": +25}},
    {"name": "Butterfly_WingsUp","shifts_bps": {"2Y": -40, "5Y": +10, "10Y": +10, "30Y": -40}},
    {"name": "Butterfly_BellyUp","shifts_bps": {"2Y": +10, "5Y": -40, "10Y": -40, "30Y": +10}},
]

# Default buy DV01 allocation (sums to 1). Adjust to tilt toward your intended buys.
BUY_KRD_WEIGHTS_DEFAULT = {
    "2Y": 0.20,
    "5Y": 0.40,
    "10Y": 0.30,
    "30Y": 0.10,
}
# === Pareto export & reporting knobs ===
# How many to show in the curated Executive menu (diverse slice of the frontier)
EXEC_SUMMARY_COUNT = 10

# Hard cap on how many *full* option folders (reports/TR/audits) to render per run
REPORT_MAX_FOLDERS = 36

# What to include in pareto_all.csv for every candidate:
#   "basic"             -> metrics only (fast; recommended)
#   "tr_parallel_lite"  -> add TR at { -50, 0, +50 } bps only
#   "full"              -> (not recommended) full TR for all; heavy
PARETO_EXPORT_LEVEL = "basic"

# Export how many non-dominated ranks (0 = only true frontier, 1 = include rank 1, etc.)
# If your GA doesn't provide ranks, we'll set 0 for all.
MAX_PARETO_RANK_EXPORTED = 1
# --- Proceeds coverage objective (for NSGA multi-objective spreading) ---
# If PROCEEDS_ANCHORS is None, we create evenly spaced anchors at the midpoints
# of PROCEEDS_ANCHOR_COUNT bins across the proceeds window [MIN..MAX].
PROCEEDS_ANCHOR_COUNT = 6
PROCEEDS_ANCHORS = None  # e.g., [0.10, 0.30, 0.50, 0.70, 0.90] in normalized [0,1] if you want custom

# ===================== NEW ENHANCEMENTS =====================

# ----- Stress Testing & Enhanced Scenarios -----
STRESS_SCENARIOS = [
    {"name": "Credit_Widening", "spread_shock_bps": 25, "rate_shock_bps": 0, "description": "Credit spreads widen 25bp"},
    {"name": "Fed_Panic", "spread_shock_bps": 50, "rate_shock_bps": -100, "description": "Emergency cuts + credit stress"},
    {"name": "Inflation_Spike", "spread_shock_bps": 10, "rate_shock_bps": 75, "description": "Rates spike on inflation concerns"},
    {"name": "Recession_Fears", "spread_shock_bps": 40, "rate_shock_bps": -75, "description": "Flight to quality + rate cuts"}
]

# Enhanced scenario reporting
SCENARIO_ANALYSIS_ENABLED = False
RISK_ATTRIBUTION_ENABLED = True


# Risk attribution buckets
RISK_ATTRIBUTION_BUCKETS = {
    "duration": [0, 3, 7, 15, 50],  # Short, Medium, Long, Very Long
    "market_value": 3,  # Tertiles: Small, Medium, Large positions
    "credit_quality": ["AAA", "AA", "A", "BBB", "Below_BBB"],  # If rating data available
    "sector": ["Treasury", "Agency", "Corporate", "Municipal", "Other"]  # If sector data available
}

# ===================== FUNDING OVERLAY FRAMEWORK =====================

# ----- Borrow-to-Buy Overlay Settings -----
FUNDING_OVERLAY_ENABLED = False

# Funding rate configuration
FUNDING_RATE_MODE = "static"  # "static" or "curve" (future: term structure based)
FUNDING_BASE_RATE_BPS = 500   # Current 1Y Treasury ~5.00%
FUNDING_SPREAD_BPS = 50       # Bank's funding spread over Treasury
FUNDING_TERM_MONTHS = 12      # Default funding term (matches TR horizon)

# Alternative funding rate calculation (if FUNDING_RATE_MODE = "curve")
FUNDING_CURVE_BASE = "SOFR"   # "SOFR", "Treasury", "LIBOR" (future enhancement)
FUNDING_CURVE_TENOR = "1Y"    # "3M", "6M", "1Y", "2Y"

# Overlay parameters
FUNDING_MULTIPLIER_OPTIONS = [1.0, 1.5, 2.0]  # How much to borrow relative to proceeds
FUNDING_UNWIND_MONTH = TR_HORIZON_MONTHS       # When to sell held bonds and repay funding

# Risk limits for funding overlays
MAX_FUNDING_DOLLARS = 50_000_000     # Maximum total funding amount
MAX_FUNDING_MULTIPLE = 3.0           # Maximum leverage multiple
FUNDING_MARGIN_HAIRCUT = 0.05        # 5% haircut on bond collateral value

# ----- Wait-Then-Buy Analysis Settings -----
WAIT_TO_BUY_ENABLED = True

# Time periods to analyze
WAIT_TO_BUY_MONTHS = [0, 3, 6, 12]   # 0 = buy now, others = wait N months then buy

# Cash parking rate while waiting (annualized)
CASH_PARKING_RATE_BPS = 475          # Current MMF/repo rates ~4.75%

# Expected rate environment assumptions for wait analysis
EXPECTED_RATE_CHANGES = {
    3: -25,   # Expect 25bp cut by month 3
    6: -50,   # Expect 50bp total cuts by month 6
    12: -75   # Expect 75bp total cuts by month 12
}

# Spread change assumptions (credit tightening/widening with rate cuts)
EXPECTED_SPREAD_CHANGES = {
    3: -5,    # 5bp credit tightening with first cut
    6: -10,   # 10bp tightening with sustained cuts
    12: -15   # 15bp tightening in full cutting cycle
}

# ----- Chain Swap / Multi-Step Settings -----
CHAIN_SWAP_ENABLED = True

# Trigger conditions for chain swaps
CHAIN_SWAP_TRIGGERS = {
    "rate_fall_profit_take": {
        "rate_threshold_bps": -75,
        "gain_threshold_pct": 0.05,  # 5% price appreciation
        "time_window_months": [6, 18],
        "action": "sell_and_redeploy",
        "redeploy_multiple": 2.0,    # Buy 2x original size
        "ladder_tenors": [3, 5, 7, 10]  # Ladder across these tenors
    },
    "curve_steepener_play": {
        "curve_change_bps": {"5Y": -50, "10Y": -25},  # 5s10s steepening
        "time_window_months": [3, 12],
        "action": "barbell_extension",
        "target_duration": 7.0
    }
}

# ----- Overlay Analysis Configuration -----

# Which overlays to include in TR analysis
OVERLAY_ANALYSIS_MODES = [
    "standard",        # Normal swap analysis (existing)
    "borrow_to_buy",   # Keep held bonds, borrow to buy new
    "wait_then_buy",   # Wait N months then execute purchase
    "funded_chain",    # Combination of funding + chain swap triggers
]

# Reporting preferences for overlays
OVERLAY_REPORTING = {
    "include_breakeven_rates": True,      # Show funding rate breakevens
    "include_time_to_profit": True,       # Show months to positive carry
    "include_risk_metrics": True,         # Show leverage/concentration risks
    "executive_summary_only": False,      # True = only summary, False = full tables
}

# Performance comparison baselines
OVERLAY_BENCHMARKS = {
    "cash": CASH_PARKING_RATE_BPS / 10000.0,  # Cash return for comparison
    "hold_portfolio": True,                     # Compare vs. holding existing bonds
    "market_index": None,                      # Future: compare vs. bond index
}

# ----- Advanced Funding Features (Future Enhancement) -----

# Graduated funding schedule (not yet implemented)
GRADUATED_FUNDING_ENABLED = False
GRADUATED_FUNDING_SCHEDULE = [
    {"month": 0, "multiplier": 0.33},         # Borrow 1/3 immediately
    {"trigger": "rate_fall_25bp", "add": 0.33},  # Borrow another 1/3 on 25bp cut
    {"trigger": "rate_fall_50bp", "add": 0.34},  # Borrow final 1/3 on 50bp cut
]

# Credit/margin monitoring (not yet implemented)
FUNDING_CREDIT_MONITORING = {
    "mark_to_market_frequency": "daily",
    "margin_call_threshold": 0.20,    # 20% decline triggers margin call
    "forced_liquidation_threshold": 0.30,  # 30% decline forces liquidation
}

# Rate environment sensitivity
FUNDING_RATE_SCENARIOS = [
    {"name": "Base", "funding_rate_change_bps": 0},
    {"name": "Funding_Stress", "funding_rate_change_bps": 100},  # Funding costs spike 100bp
    {"name": "Funding_Relief", "funding_rate_change_bps": -50},  # Funding costs fall 50bp
]