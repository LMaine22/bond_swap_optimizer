#!/usr/bin/env python3
"""
Quick analysis script: Calculate metrics if we sold ALL bonds in the portfolio.
No GA, no filtering, just a straight calculation of what happens if we sell everything.

Usage:
    python analyze_all_bonds.py
    
This gives you a quick overview of what would happen if you sold your entire portfolio
without any optimization or filtering. Useful for understanding the overall portfolio
characteristics and why the GA needs to find smaller, feasible subsets.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import (
    TARGET_BUY_BACK_YIELD,
    MIN_SWAP_SIZE_DOLLARS,
    MAX_SWAP_SIZE_DOLLARS,
    MAX_TOTAL_SWAP_LOSS_DOLLARS,
    MAX_RECOVERY_PERIOD_MONTHS,
    SOLD_WAVG_PROJ_YIELD_MAX,
)
from src.data_handler import load_and_prepare_data
from src.pruning import compute_metrics

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

def _checkmark(ok: bool) -> str: 
    return "✓" if ok else "✗"

def analyze_all_bonds():
    """Calculate metrics for selling ALL bonds in the portfolio."""
    
    print("=== ALL BONDS ANALYSIS ===")
    print("Calculating metrics if we sold EVERY bond in the portfolio...\n")
    
    # Load data
    from src.config import PORTFOLIO_FILE_PATH
    full_portfolio_df = load_and_prepare_data(PORTFOLIO_FILE_PATH)
    
    if full_portfolio_df is None or full_portfolio_df.empty:
        print("ERROR: Could not load portfolio data.")
        return
    
    print(f"Loaded {len(full_portfolio_df)} bonds from portfolio")
    print(f"Total par value: {_fmt_money(full_portfolio_df['par'].sum())}")
    print()
    
    # Create a mask that selects ALL bonds (all True)
    all_bonds_mask = np.ones(len(full_portfolio_df), dtype=bool)
    
    # Calculate metrics for selling everything
    try:
        # Calculate manually with correct logic
        total_par = full_portfolio_df['par'].sum()
        total_market = full_portfolio_df['market'].sum()
        total_book = full_portfolio_df['book'].sum()
        total_loss = total_book - total_market
        
        # Use the raw yield data (in percentage format) and convert to decimal
        raw_yield_mean = full_portfolio_df['raw__Proj Yield (TE)'].mean()
        current_yield = raw_yield_mean / 100.0  # Convert from percentage to decimal
        current_income = total_par * current_yield
        
        # Correct delta income calculation: buyback on PAR, not market
        buyback_income = total_par * TARGET_BUY_BACK_YIELD
        correct_delta_income = buyback_income - current_income
        
        # Recovery period calculation
        if correct_delta_income > 0:
            recovery_months = (12.0 * total_loss) / correct_delta_income
        else:
            recovery_months = float('inf')
        
        # Convert to dictionary for easier access
        md = {
            "market": total_market,
            "book": total_book, 
            "loss": total_loss,
            "delta_income": correct_delta_income,
            "recovery_months": recovery_months,
            "sold_wavg": current_yield,
            "proj_y": current_yield,  # Same as sold_wavg for all bonds
            "par": total_par,
            "count": len(full_portfolio_df)
        }
        
        # Check constraints
        size_ok = MIN_SWAP_SIZE_DOLLARS <= md["market"] <= MAX_SWAP_SIZE_DOLLARS
        loss_ok = md["loss"] <= MAX_TOTAL_SWAP_LOSS_DOLLARS
        rec_ok = (np.isfinite(md["recovery_months"]) and 
                 md["recovery_months"] <= MAX_RECOVERY_PERIOD_MONTHS)
        sold_ok = md["sold_wavg"] <= SOLD_WAVG_PROJ_YIELD_MAX
        
        # Display results
        print("=== RESULTS: SELL ALL BONDS ===")
        print(f"Total Proceeds (Market):     {_fmt_money(md['market'])}")
        print(f"Total Book Value:            {_fmt_money(md['book'])}")
        print(f"Total Gain/Loss:             {_fmt_money(md['loss'])}")
        print(f"Delta Income (Annual):       {_fmt_money(md['delta_income'])}")
        if np.isfinite(md['recovery_months']):
            print(f"Recovery Period:             {md['recovery_months']:.2f} months")
        else:
            print(f"Recovery Period:             N/A (negative delta income)")
            
            # Show what buyback yield would be needed for positive recovery
            # To get 1.49% yield pick-up (like in screenshot)
            target_pickup = 0.0149
            required_buyback_yield = md['sold_wavg'] + target_pickup
            required_buyback_income = md['par'] * required_buyback_yield
            required_delta_income = required_buyback_income - (md['par'] * md['sold_wavg'])
            recovery_years = md['loss'] / required_delta_income if required_delta_income > 0 else float('inf')
            
            print(f"  (To get 1.49% pick-up: need {_fmt_pct(required_buyback_yield)} buyback yield)")
            print(f"  (This would give {recovery_years:.2f} year recovery period)")
        print(f"Sold Wavg Proj Yield:        {_fmt_pct(md['sold_wavg'])}")
        print(f"MV-Weighted Proj Yield:      {_fmt_pct(md['proj_y'])}")
        print(f"Buyback (assumed):           {_fmt_pct(TARGET_BUY_BACK_YIELD)}")
        print(f"Yield Pick-up:               {_fmt_pct(TARGET_BUY_BACK_YIELD - md['sold_wavg'])}")
        print(f"Total Par Sold:              {_fmt_money(md['par'])}")
        print(f"Bonds in Set:                {md['count']}")
        print()
        
        # Constraint checks
        print("=== CONSTRAINT CHECKS ===")
        print(f"Size OK:                     {_checkmark(size_ok)} "
              f"({_fmt_money(md['market'])} in range {_fmt_money(MIN_SWAP_SIZE_DOLLARS)} - {_fmt_money(MAX_SWAP_SIZE_DOLLARS)})")
        print(f"Loss OK:                     {_checkmark(loss_ok)} "
              f"({_fmt_money(md['loss'])} <= {_fmt_money(MAX_TOTAL_SWAP_LOSS_DOLLARS)})")
        if np.isfinite(md['recovery_months']):
            print(f"Recovery OK:                 {_checkmark(rec_ok)} "
                  f"({md['recovery_months']:.2f} <= {MAX_RECOVERY_PERIOD_MONTHS:.1f} months)")
        else:
            print(f"Recovery OK:                 {_checkmark(rec_ok)} "
                  f"(N/A - negative delta income)")
        print(f"Sold Yield OK:               {_checkmark(sold_ok)} "
              f"({_fmt_pct(md['sold_wavg'])} <= {_fmt_pct(SOLD_WAVG_PROJ_YIELD_MAX)})")
        print()
        
        # Overall feasibility
        all_ok = size_ok and loss_ok and rec_ok and sold_ok
        print(f"OVERALL FEASIBLE:            {_checkmark(all_ok)}")
        
        if not all_ok:
            print("\n❌ Selling all bonds would violate constraints!")
            print("   This is why the GA needs to find smaller, feasible subsets.")
        else:
            print("\n✅ Selling all bonds would be feasible!")
            print("   But the GA might find better smaller subsets.")
            
    except Exception as e:
        print(f"ERROR calculating metrics: {e}")
        return
    
    # Additional insights
    print("\n=== INSIGHTS ===")
    if md["loss"] < 0:
        print(f"• This would be a GAIN of {_fmt_money(-md['loss'])}")
    else:
        print(f"• This would be a LOSS of {_fmt_money(md['loss'])}")
        
    if md["delta_income"] > 0:
        print(f"• Annual income would INCREASE by {_fmt_money(md['delta_income'])}")
    else:
        print(f"• Annual income would DECREASE by {_fmt_money(-md['delta_income'])}")
        
    if np.isfinite(md["recovery_months"]) and md["recovery_months"] > 0:
        if md["recovery_months"] < 12:
            print(f"• Loss would be recovered in {md['recovery_months']:.1f} months")
        else:
            print(f"• Loss would be recovered in {md['recovery_months']/12:.1f} years")
    elif md["recovery_months"] < 0:
        print("• This is a gain - no recovery period needed!")
    else:
        print("• Recovery period is undefined (no delta income)")
        
    print(f"• Average bond size: {_fmt_money(md['market'] / md['count'])}")
    print(f"• Portfolio yield: {_fmt_pct(md['sold_wavg'])}")
    print(f"• Target buyback yield: {_fmt_pct(TARGET_BUY_BACK_YIELD)}")
    print(f"• Potential yield pickup: {_fmt_pct(TARGET_BUY_BACK_YIELD - md['sold_wavg'])}")

if __name__ == "__main__":
    analyze_all_bonds()
