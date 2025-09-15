#!/usr/bin/env python3
"""
Convert Excel data to Parquet format for faster loading.
This script converts the Excel file to Parquet and validates the conversion.
"""

import pandas as pd
import os
from pathlib import Path

def convert_excel_to_parquet():
    """Convert Excel file to Parquet format."""
    
    # File paths
    excel_file = "../data/Sample_v02.xlsx"
    parquet_file = "../data/Sample_v02.parquet"
    
    print("🔄 Converting Excel to Parquet...")
    print(f"Input:  {excel_file}")
    print(f"Output: {parquet_file}")
    
    # Check if Excel file exists
    if not os.path.exists(excel_file):
        print(f"❌ Error: Excel file not found: {excel_file}")
        return False
    
    try:
        # Read Excel file
        print("📖 Reading Excel file...")
        df = pd.read_excel(excel_file, sheet_name='Sheet')
        
        print(f"✅ Excel file loaded successfully")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Convert to Parquet
        print("💾 Converting to Parquet...")
        df.to_parquet(parquet_file, index=False, compression='snappy')
        
        # Verify conversion
        print("🔍 Verifying conversion...")
        df_parquet = pd.read_parquet(parquet_file)
        
        # Compare data
        if df.equals(df_parquet):
            print("✅ Data integrity verified - files are identical")
        else:
            print("⚠️  Warning: Data differs between Excel and Parquet")
            print(f"   Excel shape: {df.shape}")
            print(f"   Parquet shape: {df_parquet.shape}")
        
        # File size comparison
        excel_size = os.path.getsize(excel_file) / 1024**2
        parquet_size = os.path.getsize(parquet_file) / 1024**2
        compression_ratio = excel_size / parquet_size
        
        print(f"\n📊 File size comparison:")
        print(f"   Excel:   {excel_size:.1f} MB")
        print(f"   Parquet: {parquet_size:.1f} MB")
        print(f"   Compression: {compression_ratio:.1f}x smaller")
        
        print(f"\n🎉 Conversion successful!")
        print(f"   Parquet file created: {parquet_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False

def benchmark_loading():
    """Benchmark loading speed of Excel vs Parquet."""
    import time
    
    excel_file = "../data/Sample_v02.xlsx"
    parquet_file = "../data/Sample_v02.parquet"
    
    if not os.path.exists(parquet_file):
        print("❌ Parquet file not found. Run conversion first.")
        return
    
    print("\n⏱️  Benchmarking loading speed...")
    
    # Benchmark Excel
    print("📊 Loading Excel...")
    start_time = time.time()
    df_excel = pd.read_excel(excel_file, sheet_name='Sheet')
    excel_time = time.time() - start_time
    print(f"   Excel load time: {excel_time:.3f} seconds")
    
    # Benchmark Parquet
    print("📊 Loading Parquet...")
    start_time = time.time()
    df_parquet = pd.read_parquet(parquet_file)
    parquet_time = time.time() - start_time
    print(f"   Parquet load time: {parquet_time:.3f} seconds")
    
    # Calculate speedup
    speedup = excel_time / parquet_time
    print(f"\n🚀 Speedup: {speedup:.1f}x faster with Parquet!")
    
    # Verify data is identical
    if df_excel.equals(df_parquet):
        print("✅ Data integrity confirmed")
    else:
        print("⚠️  Data differs between formats")

if __name__ == "__main__":
    print("=" * 60)
    print("📊 EXCEL TO PARQUET CONVERTER")
    print("=" * 60)
    
    # Convert Excel to Parquet
    success = convert_excel_to_parquet()
    
    if success:
        # Benchmark the conversion
        benchmark_loading()
        
        print("\n" + "=" * 60)
        print("✅ CONVERSION COMPLETE!")
        print("=" * 60)
        print("Next steps:")
        print("1. Update config.py to use Parquet file")
        print("2. Update data_handler.py to read Parquet")
        print("3. Test the optimized system")
    else:
        print("\n❌ Conversion failed. Check the errors above.")
