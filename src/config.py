# src/config.py
# ---------------------- CONFIGURATION WITH YIELD CLEANUP ----------------------

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
EXCEL_SHEET_NAME = "Sheet"

# ----- Scenario Selection -----
# Choose your scenario type:
# "tax_loss" - For tax loss harvesting (higher loss tolerance, recovery must exceed 3.5 months)
# "conservative" - For minimal loss swaps (lower loss tolerance, faster recovery preferred)
# "yield_cleanup" - For cleaning out low-yielding bonds (target 2-4% yields for removal)
# "custom" - Use custom parameters below

SCENARIO_MODE = "yield_cleanup"  # Change this to switch scenarios

# ----- Scenario Presets -----
SCENARIO_PRESETS = {
    "tax_loss": {
        "description": "Tax Loss Harvesting - Higher loss tolerance, recovery must exceed 3.5 months",
        "min_swap_size": 7_500_000,
        "max_swap_size": 100_000_000,
        "min_loss": 500_000,
        "max_loss": 1_000_000,
        "min_recovery_months": 3.5,  # Must exceed this for tax purposes
        "max_recovery_months": 24.0,  # Upper bound
        "target_buy_yield": 5.0,  # %
        "max_sold_wavg_yield": 7.0,  # % - More lenient to see more options
        "max_individual_bond_yield": 8.0,  # % - Maximum yield of any single bond to sell
        "enforce_min_loss": True,
    },
    "conservative": {
        "description": "Conservative Swap - Minimize losses, faster recovery",
        "min_swap_size": 7_500_000,
        "max_swap_size": 100_000_000,
        "min_loss": 0,  # No minimum loss required
        "max_loss": 500_000,  # Lower max loss
        "min_recovery_months": 0.0,
        "max_recovery_months": 12.0,  # Prefer faster recovery
        "target_buy_yield": 5.0,  # %
        "max_sold_wavg_yield": 6.0,  # % - Expanded to see more options
        "max_individual_bond_yield": 7.0,  # % - Maximum yield of any single bond to sell
        "enforce_min_loss": False,
    },
    "yield_cleanup": {
        "description": "Yield Cleanup - Target low-yielding bonds for removal (2-4% range)",
        "min_swap_size": 2_500_000,
        "max_swap_size": 100_000_000,
        "min_loss": 0,  # No minimum loss required
        "max_loss": 750_000,  # Moderate loss tolerance
        "min_recovery_months": 0.0,
        "max_recovery_months": 18.0,  # Flexible recovery
        "target_buy_yield": 5.5,  # % - Higher target to ensure good pickup
        "max_sold_wavg_yield": 2.5,  # % - Focus on low-yielding portfolios
        "min_sold_wavg_yield": 1.5,  # % - NEW: Minimum to ensure we're cleaning up low yields
        "max_individual_bond_yield": 5.0,  # % - Don't sell anything too good
        "enforce_min_loss": False,
    },
    "custom": {
        "description": "Custom: $7.5M swap with 3.5 month recovery target",
        "min_swap_size": 7_500_000,
        "max_swap_size": 100_000_000,
        "min_loss": 0,  # No minimum loss required
        "max_loss": 500_000,  # Adjust based on client tolerance
        "min_recovery_months": 0.0,  # No minimum recovery
        "max_recovery_months": 3.5,  # Target 3.5 month recovery
        "target_buy_yield": 5.0,  # %
        "max_sold_wavg_yield": 6.0,  # % - Expanded range
        "min_sold_wavg_yield": 0.0,  # % - No minimum constraint
        "max_individual_bond_yield": 7.0,  # % - Maximum yield of any single bond to sell
        "enforce_min_loss": False,  # Not a tax loss swap
    }
}

# ----- Load Current Scenario -----
if SCENARIO_MODE not in SCENARIO_PRESETS:
    print(f"Warning: Unknown scenario '{SCENARIO_MODE}', defaulting to 'conservative'")
    SCENARIO_MODE = "conservative"

_current_scenario = SCENARIO_PRESETS[SCENARIO_MODE]

# ----- Main Swap Parameters (loaded from scenario) -----
MIN_SWAP_SIZE_DOLLARS = _current_scenario["min_swap_size"]
MAX_SWAP_SIZE_DOLLARS = _current_scenario["max_swap_size"]
ENFORCE_MIN_SWAP_LOSS = _current_scenario["enforce_min_loss"]
MIN_TOTAL_SWAP_LOSS_DOLLARS = _current_scenario["min_loss"]
MAX_TOTAL_SWAP_LOSS_DOLLARS = _current_scenario["max_loss"]
MIN_RECOVERY_PERIOD_MONTHS = _current_scenario["min_recovery_months"]
MAX_RECOVERY_PERIOD_MONTHS = _current_scenario["max_recovery_months"]

# Convert percentages
TARGET_BUY_BACK_YIELD = _pct(_current_scenario["target_buy_yield"])
SOLD_WAVG_PROJ_YIELD_MAX = _pct(_current_scenario["max_sold_wavg_yield"])
SOLD_WAVG_PROJ_YIELD_MIN = _pct(_current_scenario.get("min_sold_wavg_yield", 0.0))  # NEW
MAX_INDIVIDUAL_BOND_YIELD = _pct(_current_scenario["max_individual_bond_yield"])

# ----- Individual Bond Pre-filtering -----
ENFORCE_MIN_SELL_YIELD = False
MIN_SELL_ACCTG_YIELD_RAW = 0
MIN_SELL_ACCTG_YIELD = _pct(MIN_SELL_ACCTG_YIELD_RAW)

MAX_UNREALIZED_LOSS_PERCENT = -9.0  # Filter out bonds with > 9% unrealized loss
MAX_INDIVIDUAL_LOSS_DOLLARS = 200_000  # Max loss per individual bond

# ----- Algorithm Parameters -----
GA_SEED = 49
GA_POP_SIZE = 500  # Reduced for faster execution
GA_GENERATIONS = 100  # Reduced for faster execution
GA_MUTATION_RATE = 0.25
GA_CROSSOVER_RATE = 0.6

# ----- Reporting -----
EXEC_SUMMARY_COUNT = 8  # Number of diverse options to show
REPORT_MAX_FOLDERS = 12  # Max detailed reports to generate
MAX_PARETO_RANK_EXPORTED = 1

# Proceeds coverage objective (for diversity)
PROCEEDS_ANCHOR_COUNT = 6
PROCEEDS_ANCHORS = None  # Auto-generate

# ----- Simple Total Return Analysis -----
ENABLE_TR_ANALYSIS = True  # Set to False to skip TR analysis entirely
TR_HORIZON_MONTHS = 12
TR_PARALLEL_SHIFTS_BPS = [-50, 0, 50]  # Simplified set

# Simple assumptions for TR analysis
ASSUMED_HOLD_DURATION_YEARS = 5.0
ASSUMED_HOLD_CONVEXITY = 0.0
ASSUMED_BUY_DURATION_YEARS = 5.0
ASSUMED_BUY_CONVEXITY = 0.0
TR_INCLUDE_ROLL_ON_SWAP = True

# ----- Advanced Features (Disabled by Default) -----
# These can be enabled if needed, but keeping simple for now
FUNDING_OVERLAY_ENABLED = False
WAIT_TO_BUY_ENABLED = False
SCENARIO_ANALYSIS_ENABLED = False
RISK_ATTRIBUTION_ENABLED = False

# Key-rate analysis (simplified)
KEY_RATE_BUCKETS = ["2Y", "5Y", "10Y", "30Y"]
KEY_RATE_KRD_COLUMN_MAP = {
    "2Y": ["krd_2y", "dv01_2y"],
    "5Y": ["krd_5y", "dv01_5y"],
    "10Y": ["krd_10y", "dv01_10y"],
    "30Y": ["krd_30y", "dv01_30y"],
}

KEY_RATE_SHOCKS = [
    {"name": "Steepener", "shifts_bps": {"2Y": -50, "5Y": -25, "10Y": 0, "30Y": 25}},
    {"name": "Flattener", "shifts_bps": {"2Y": 25, "5Y": 10, "10Y": -10, "30Y": -25}},
]

BUY_KRD_WEIGHTS_DEFAULT = {
    "2Y": 0.20,
    "5Y": 0.40,
    "10Y": 0.30,
    "30Y": 0.10,
}


# ----- Helper Functions -----
def print_scenario_summary():
    """Print current scenario configuration"""
    print(f"\n=== SCENARIO: {SCENARIO_MODE.upper()} ===")
    print(_current_scenario["description"])
    print(f"Swap Size: ${MIN_SWAP_SIZE_DOLLARS:,.0f} - ${MAX_SWAP_SIZE_DOLLARS:,.0f}")
    print(f"Loss Range: ${MIN_TOTAL_SWAP_LOSS_DOLLARS:,.0f} - ${MAX_TOTAL_SWAP_LOSS_DOLLARS:,.0f}")
    if ENFORCE_MIN_SWAP_LOSS:
        print(f"Minimum Loss Required: ${MIN_TOTAL_SWAP_LOSS_DOLLARS:,.0f} (for tax purposes)")
    print(f"Recovery Period: {MIN_RECOVERY_PERIOD_MONTHS:.1f} - {MAX_RECOVERY_PERIOD_MONTHS:.1f} months")
    if MIN_RECOVERY_PERIOD_MONTHS > 0:
        print(f"  → Must exceed {MIN_RECOVERY_PERIOD_MONTHS} months for tax loss harvesting")
    print(f"Target Buy Yield: {TARGET_BUY_BACK_YIELD:.2%}")
    print(f"Sold Yield Range: {SOLD_WAVG_PROJ_YIELD_MIN:.2%} - {SOLD_WAVG_PROJ_YIELD_MAX:.2%}")
    if SOLD_WAVG_PROJ_YIELD_MIN > 0:
        print(f"  → Targets low-yielding bonds for cleanup")
    print()


def switch_scenario(new_mode: str):
    """Switch to a different scenario mode"""
    global SCENARIO_MODE, MIN_SWAP_SIZE_DOLLARS, MAX_SWAP_SIZE_DOLLARS
    global ENFORCE_MIN_SWAP_LOSS, MIN_TOTAL_SWAP_LOSS_DOLLARS, MAX_TOTAL_SWAP_LOSS_DOLLARS
    global MIN_RECOVERY_PERIOD_MONTHS, MAX_RECOVERY_PERIOD_MONTHS
    global TARGET_BUY_BACK_YIELD, SOLD_WAVG_PROJ_YIELD_MAX, SOLD_WAVG_PROJ_YIELD_MIN
    global MAX_INDIVIDUAL_BOND_YIELD, _current_scenario

    if new_mode not in SCENARIO_PRESETS:
        print(f"Error: Unknown scenario '{new_mode}'. Available: {list(SCENARIO_PRESETS.keys())}")
        return False

    SCENARIO_MODE = new_mode
    _current_scenario = SCENARIO_PRESETS[SCENARIO_MODE]

    # Update all parameters
    MIN_SWAP_SIZE_DOLLARS = _current_scenario["min_swap_size"]
    MAX_SWAP_SIZE_DOLLARS = _current_scenario["max_swap_size"]
    ENFORCE_MIN_SWAP_LOSS = _current_scenario["enforce_min_loss"]
    MIN_TOTAL_SWAP_LOSS_DOLLARS = _current_scenario["min_loss"]
    MAX_TOTAL_SWAP_LOSS_DOLLARS = _current_scenario["max_loss"]
    MIN_RECOVERY_PERIOD_MONTHS = _current_scenario["min_recovery_months"]
    MAX_RECOVERY_PERIOD_MONTHS = _current_scenario["max_recovery_months"]
    TARGET_BUY_BACK_YIELD = _pct(_current_scenario["target_buy_yield"])
    SOLD_WAVG_PROJ_YIELD_MAX = _pct(_current_scenario["max_sold_wavg_yield"])
    SOLD_WAVG_PROJ_YIELD_MIN = _pct(_current_scenario.get("min_sold_wavg_yield", 0.0))
    MAX_INDIVIDUAL_BOND_YIELD = _pct(_current_scenario["max_individual_bond_yield"])

    print(f"Switched to scenario: {new_mode}")
    return True