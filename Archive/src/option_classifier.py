"""
Option Classification Module

Classifies bond swap options into categories based on their strongest characteristics:
- low_loss: Options with lowest losses (closest to 0 or gains)
- high_income: Options with highest delta income
- fast_recovery: Options with shortest recovery periods
- low_yield: Options with lowest sold weighted average accounting yield
- multi_strength: Options that excel in multiple areas
- low_strength: Options that don't excel in any particular area
"""

import os
import shutil
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def classify_options(rows: List[Dict], run_dir: str) -> List[Dict]:
    """
    Classify options into categories based on their strongest characteristics.
    
    Args:
        rows: List of option dictionaries with metrics
        run_dir: Base directory for the run
        
    Returns:
        Updated rows with new folder paths and categories
    """
    if not rows:
        return rows
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(rows)
    
    # Calculate normalized scores (0-1) for each metric
    # Lower is better for loss and recovery, higher is better for income and yield
    scores = {}
    
    # Loss score: lower loss = higher score (inverted)
    loss_scores = 1.0 - (df['net_gl'] - df['net_gl'].min()) / (df['net_gl'].max() - df['net_gl'].min())
    scores['low_loss'] = loss_scores.fillna(0.0)
    
    # Income score: higher income = higher score
    income_scores = (df['delta_income'] - df['delta_income'].min()) / (df['delta_income'].max() - df['delta_income'].min())
    scores['high_income'] = income_scores.fillna(0.0)
    
    # Recovery score: lower recovery = higher score (inverted)
    recovery_scores = 1.0 - (df['recovery_months'] - df['recovery_months'].min()) / (df['recovery_months'].max() - df['recovery_months'].min())
    scores['fast_recovery'] = recovery_scores.fillna(0.0)
    
    # Yield score: lower yield = higher score (inverted)
    # Calculate yield from delta_income and par (approximation)
    # This is a simplified calculation - in practice we'd need the actual yield data
    if 'delta_income' in df.columns and 'par' in df.columns:
        # Approximate yield as delta_income / par * 100
        approx_yield = (df['delta_income'] / df['par'] * 100).fillna(0.0)
        yield_scores = 1.0 - (approx_yield - approx_yield.min()) / (approx_yield.max() - approx_yield.min())
        scores['low_yield'] = yield_scores.fillna(0.5)
    else:
        yield_scores = pd.Series([0.5] * len(df))  # Fallback
        scores['low_yield'] = yield_scores
    
    # Find each option's strongest category
    categories = []
    for i, row in df.iterrows():
        option_scores = {cat: scores[cat].iloc[i] for cat in scores.keys()}
        
        # Check for multi-strength (excel in 2+ categories)
        strong_categories = [cat for cat, score in option_scores.items() if score >= 0.7]
        if len(strong_categories) >= 2:
            category = 'multi_strength'
        # Check for low-strength (no category above threshold)
        elif max(option_scores.values()) < 0.6:
            category = 'low_strength'
        else:
            # Assign to strongest category
            category = max(option_scores.items(), key=lambda x: x[1])[0]
        
        categories.append(category)
    
    # Create category subdirectories and move files
    options_dir = os.path.join(run_dir, "options")
    category_counts = {}
    
    updated_rows = []
    for i, (row, category) in enumerate(zip(rows, categories)):
        # Count options in each category
        category_counts[category] = category_counts.get(category, 0) + 1
        opt_num = category_counts[category]
        
        # Create new folder path
        old_folder = row['folder']
        new_folder = os.path.join(options_dir, category, f"opt_{opt_num:02d}")
        
        # Create category directory if it doesn't exist
        category_dir = os.path.join(options_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Move files from old location to new location
        if os.path.exists(old_folder):
            if os.path.exists(new_folder):
                shutil.rmtree(new_folder)
            shutil.move(old_folder, new_folder)
        
        # Update row with new folder path
        updated_row = row.copy()
        updated_row['folder'] = new_folder
        updated_row['category'] = category
        updated_rows.append(updated_row)
    
    return updated_rows


def create_category_summary(rows: List[Dict], run_dir: str) -> None:
    """
    Create a summary file showing how options were classified.
    
    Args:
        rows: List of classified option dictionaries
        run_dir: Base directory for the run
    """
    if not rows:
        return
    
    summary_path = os.path.join(run_dir, "options", "classification_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("BOND SWAP OPTIONS CLASSIFICATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        # Group by category
        categories = {}
        for row in rows:
            cat = row.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(row)
        
        # Write summary for each category
        for category, options in categories.items():
            f.write(f"{category.upper().replace('_', ' ')} ({len(options)} options):\n")
            f.write("-" * 30 + "\n")
            
            for opt in options:
                f.write(f"  {opt['folder']}\n")
                f.write(f"    Loss: ${opt['net_gl']:,.2f}\n")
                f.write(f"    Delta Income: ${opt['delta_income']:,.2f}\n")
                f.write(f"    Recovery: {opt['recovery_months']:.2f} months\n")
                f.write(f"    Positions: {opt['positions']}\n")
                f.write("\n")
            
            f.write("\n")
        
        f.write("Classification Logic:\n")
        f.write("- low_loss: Lowest losses (closest to 0 or gains)\n")
        f.write("- high_income: Highest delta income\n")
        f.write("- fast_recovery: Shortest recovery periods\n")
        f.write("- low_yield: Lowest sold weighted average accounting yield\n")
        f.write("- multi_strength: Excel in 2+ categories (score >= 0.7)\n")
        f.write("- low_strength: No category above threshold (score < 0.6)\n")
