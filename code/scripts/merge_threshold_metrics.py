#!/usr/bin/env python
"""
Merge individual threshold CSV files into metrics_by_threshold.csv and threshold_stats.csv

Usage:
    python merge_threshold_metrics.py                    # Merge both head and body
    python merge_threshold_metrics.py --anatomy head     # Merge head only
    python merge_threshold_metrics.py --anatomy body     # Merge body only
"""

import os
import glob
import argparse
import pandas as pd


def merge_csv_files(input_dir, anatomy):
    """Merge individual threshold CSV files into metrics_by_threshold.csv and threshold_stats.csv"""
    csv_files = sorted(glob.glob(os.path.join(input_dir, anatomy, 'metrics_0.*.csv')))
    
    if not csv_files:
        print(f"No metrics_0.*.csv files found in {input_dir}/{anatomy}/")
        return
    
    print(f"Found {len(csv_files)} CSV files for {anatomy}")
    
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total records: {len(df)}")
    
    # Save merged metrics
    metrics_path = os.path.join(input_dir, anatomy, 'metrics_by_threshold.csv')
    df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")
    
    # Calculate and save stats
    stats = df.groupby('threshold')[['ssim', 'rmse']].agg(['mean', 'std'])
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()
    stats_path = os.path.join(input_dir, anatomy, 'threshold_stats.csv')
    stats.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")
    
    # Best thresholds
    best_ssim_thr = stats.loc[stats['ssim_mean'].idxmax(), 'threshold']
    best_rmse_thr = stats.loc[stats['rmse_mean'].idxmin(), 'threshold']
    print(f"\nOptimal thresholds for {anatomy}:")
    print(f"  Best SSIM: {best_ssim_thr:.2f} (SSIM={stats['ssim_mean'].max():.4f})")
    print(f"  Best RMSE: {best_rmse_thr:.2f} (RMSE={stats['rmse_mean'].min():.2f})")


def main():
    parser = argparse.ArgumentParser(description='Merge threshold CSV files')
    parser.add_argument('--anatomy', type=str, choices=['head', 'body'], 
                        help='Anatomy to process (default: both)')
    parser.add_argument('--input_dir', type=str, 
                        default='results_threshold_sweep',
                        help='Input directory containing threshold results')
    args = parser.parse_args()
    
    if args.anatomy:
        merge_csv_files(args.input_dir, args.anatomy)
    else:
        for anatomy in ['head', 'body']:
            print(f"\n{'='*50}")
            print(f"Processing {anatomy.upper()}")
            print('='*50)
            merge_csv_files(args.input_dir, anatomy)


if __name__ == '__main__':
    main()
