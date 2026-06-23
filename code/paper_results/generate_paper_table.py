#!/usr/bin/env python3
"""
Generate paper-ready table with Mean±Std format.
Format: Part | Metric | FBP | UNet | UFormer | Restormer | NAFNet | Ours
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# 데이터 로드
# =============================================================================
CSV_PATH = Path(__file__).parent / "all_metrics_results.csv"
df = pd.read_csv(CSV_PATH)

output_dir = Path(__file__).parent
metrics = ['ssim', 'fsim', 'rmse']
# 컬럼 순서 (이미지와 동일)
model_order = ['FBP', 'UNet', 'Uformer', 'Restormer', 'NAFNet', 'Proposed']
col_names = ['FBP', 'UNet', 'UFormer', 'Restormer', 'NAFNet', 'Ours']

# =============================================================================
# Mean ± Std Table 생성
# =============================================================================
print("=" * 100)
print("Paper Table - Mean ± Std")
print("=" * 100)

table_data = []
for anatomy in ['body', 'head']:
    df_anat = df[df['anatomy'] == anatomy]
    
    for metric in metrics:
        row = {
            'Part': anatomy.capitalize(),
            'Metric': f'{metric.upper()} {"↑" if metric != "rmse" else "↓"}'
        }
        
        means = df_anat.groupby('model')[metric].mean()
        stds = df_anat.groupby('model')[metric].std()
        
        # Find best model for bolding
        if metric == 'rmse':
            best_model = means.idxmin()
        else:
            best_model = means.idxmax()
        
        for model, col in zip(model_order, col_names):
            mean = means[model]
            std = stds[model]
            
            # 모든 메트릭 소수점 2자리
            val = f"{mean:.2f}±{std:.2f}"
            
            row[col] = val
        
        table_data.append(row)

result_df = pd.DataFrame(table_data)
print(result_df.to_string(index=False))

# =============================================================================
# Excel 저장
# =============================================================================
excel_path = output_dir / 'paper_table_final.xlsx'
result_df.to_excel(excel_path, index=False, sheet_name='Mean_Std')
print(f"\n✅ Saved: {excel_path.name}")

# =============================================================================
# CSV 저장
# =============================================================================
csv_path = output_dir / 'paper_table_final.csv'
result_df.to_csv(csv_path, index=False)
print(f"✅ Saved: {csv_path.name}")

# =============================================================================
# LaTeX Table
# =============================================================================
print("\n" + "=" * 100)
print("LaTeX Table")
print("=" * 100)

print(r"""
\begin{table}[htbp]
\centering
\caption{Quantitative comparison of reconstruction methods}
\label{tab:results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll|cccccc}
\toprule
Part & Metric & FBP & UNet & UFormer & Restormer & NAFNet & \textbf{Ours} \\
\midrule""")

for i, row in result_df.iterrows():
    vals = [row['FBP'], row['UNet'], row['UFormer'], row['Restormer'], row['NAFNet'], row['Ours']]
    line = f"{row['Part']} & {row['Metric']} & " + " & ".join(vals) + r" \\"
    
    # Add midrule between Body and Head
    if row['Part'] == 'Body' and row['Metric'] == 'RMSE ↓':
        line += "\n\\midrule"
    
    print(line)

print(r"""\bottomrule
\end{tabular}%
}
\end{table}
""")

print("\n완료!")
