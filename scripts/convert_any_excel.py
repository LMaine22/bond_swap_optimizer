#!/usr/bin/env python3
"""
Convert any Excel file to Parquet format.
This script will try to read the first sheet of any Excel file.
"""

import pandas as pd
import os
import sys

def convert_excel_to_parquet(excel_path, output_path=None):
    """Convert Excel file to Parquet format."""
    
    if not os.path.exists(excel_path):
        print(f"❌ Error: File not found: {excel_path}")
        return False
    
    if output_path is None:
        output_path = excel_path.replace('.xlsx', '.parquet').replace('.xls', '.parquet')
    
    print(f"🔄 Converting: {excel_path} → {output_path}")
    
    try:
        # Try to read the first sheet
        print("📖 Reading Excel file...")
        df = pd.read_excel(excel_path, sheet_name=0)  # First sheet
        
        print(f"✅ Excel file loaded successfully")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Convert to Parquet
        print("💾 Converting to Parquet...")
        df.to_parquet(output_path, index=False, compression='snappy')
        
        print(f"✅ Conversion successful!")
        print(f"   Parquet file created: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_any_excel.py <excel_file> [output_file]")
        print("Example: python convert_any_excel.py data/Sample_v02.xlsx")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_excel_to_parquet(excel_file, output_file)
