"""
통계적 유의성 검정 스크립트
- Paired t-test: 대응표본 t-검정
- Wilcoxon signed-rank test: 비모수적 대응표본 검정
- Cohen's d: 효과 크기 (effect size)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

_PAPER_RESULTS_DIR = Path(__file__).resolve().parent


def perform_paired_tests(df: pd.DataFrame, 
                         anatomy: str, 
                         metric: str, 
                         proposed_model: str = 'Proposed') -> pd.DataFrame:
    """
    Proposed 모델 vs 다른 모델들에 대해 paired statistical tests 수행
    
    Args:
        df: 전체 데이터프레임 (columns: folder, anatomy, model, ssim, fsim, rmse)
        anatomy: 'body' or 'head'
        metric: 'ssim' or 'rmse'
        proposed_model: 비교 기준 모델명
    
    Returns:
        검정 결과 DataFrame
    """
    anatomy_df = df[df['anatomy'] == anatomy]
    models = [m for m in anatomy_df['model'].unique() if m != proposed_model]
    
    # Pivot: 각 샘플(folder)에 대해 모델별 metric 값
    pivot = anatomy_df.pivot(index='folder', columns='model', values=metric)
    
    results = []
    for model in models:
        proposed = pivot[proposed_model].dropna()
        other = pivot[model].dropna()
        
        # 공통 샘플만 사용 (paired test이므로)
        common = proposed.index.intersection(other.index)
        p = proposed.loc[common].values
        o = other.loc[common].values
        
        # 1. Paired t-test
        # H0: mean(proposed - other) = 0
        # H1: mean(proposed - other) ≠ 0
        t_stat, t_pval = stats.ttest_rel(p, o)
        
        # 2. Wilcoxon signed-rank test (비모수적 검정)
        # 정규성 가정이 불확실할 때 robust한 대안
        w_stat, w_pval = stats.wilcoxon(p, o)
        
        # 3. Cohen's d (효과 크기)
        # d = mean(diff) / std(diff)
        diff = p - o
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
        results.append({
            'anatomy': anatomy,
            'metric': metric,
            'comparison': f'Proposed vs {model}',
            'n_samples': len(common),
            'proposed_mean': np.mean(p),
            'proposed_std': np.std(p, ddof=1),
            'other_mean': np.mean(o),
            'other_std': np.std(o, ddof=1),
            'diff_mean': np.mean(diff),
            'diff_std': np.std(diff, ddof=1),
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_pvalue': w_pval,
            'cohens_d': cohens_d,
            'significant_005': t_pval < 0.05,
            'significant_001': t_pval < 0.01,
            'significant_0001': t_pval < 0.001
        })
    
    return pd.DataFrame(results)


def interpret_cohens_d(d: float) -> str:
    """Cohen's d 해석"""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    elif d < 1.2:
        return "large"
    else:
        return "very large"


def main():
    # Load data
    df = pd.read_csv(str(_PAPER_RESULTS_DIR / 'all_metrics_results.csv'))
    
    print("=" * 80)
    print("통계적 유의성 검정")
    print("Methods: Paired t-test, Wilcoxon signed-rank test, Cohen's d")
    print("=" * 80)
    
    # Run tests for all combinations
    all_results = []
    for anatomy in ['body', 'head']:
        for metric in ['ssim', 'fsim', 'rmse']:
            result = perform_paired_tests(df, anatomy, metric)
            all_results.append(result)
    
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Display results
    for anatomy in ['body', 'head']:
        for metric in ['ssim', 'fsim', 'rmse']:
            print(f"\n### {anatomy.upper()} - {metric.upper()} ###")
            subset = results_df[
                (results_df['anatomy'] == anatomy) & 
                (results_df['metric'] == metric)
            ]
            for _, row in subset.iterrows():
                # Significance stars
                if row['t_pvalue'] < 0.001:
                    sig = "***"
                elif row['t_pvalue'] < 0.01:
                    sig = "**"
                elif row['t_pvalue'] < 0.05:
                    sig = "*"
                else:
                    sig = ""
                
                effect = interpret_cohens_d(row['cohens_d'])
                print(f"{row['comparison']:25s} | "
                      f"Δ={row['diff_mean']:+.4f} | "
                      f"t={row['t_statistic']:8.2f} | "
                      f"p={row['t_pvalue']:.2e} {sig:3s} | "
                      f"d={row['cohens_d']:+.3f} ({effect})")
    
    # Save results to Excel with proper number formatting
    output_path = str(_PAPER_RESULTS_DIR / 'statistical_significance.xlsx')
    
    # Use ExcelWriter for more control over formatting
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, index=False, sheet_name='Results')
        
        # Get the worksheet
        worksheet = writer.sheets['Results']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 30)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"\n결과 저장: {output_path}")
    
    return results_df


if __name__ == '__main__':
    main()
