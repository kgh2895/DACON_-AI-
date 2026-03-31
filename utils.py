"""
유틸리티: EDA, 데이터 검증, 시각화
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from config import CFG
from dataset import load_views


# ── EDA ──────────────────────────────────────────────────────────────────────

def eda(train_csv: Path, dev_csv: Path):
    train_df = pd.read_csv(train_csv)
    dev_df   = pd.read_csv(dev_csv)

    label_col = 'label' if 'label' in train_df.columns else 'Label'

    print('=' * 50)
    print(f'Train: {len(train_df)}개 | Dev: {len(dev_df)}개')
    print()

    for name, df in [('Train', train_df), ('Dev', dev_df)]:
        counts = df[label_col].value_counts()
        print(f'[{name}] 클래스 분포')
        for cls, cnt in counts.items():
            pct = cnt / len(df) * 100
            print(f'  {cls:10s}: {cnt:4d}개 ({pct:.1f}%)')
        print()

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, df) in zip(axes, [('Train', train_df), ('Dev', dev_df)]):
        counts = df[label_col].value_counts()
        ax.bar(counts.index, counts.values, color=['steelblue', 'tomato'])
        ax.set_title(f'{name} Label Distribution')
        ax.set_ylabel('Count')
        for i, (cls, cnt) in enumerate(counts.items()):
            ax.text(i, cnt + 1, str(cnt), ha='center')
    plt.tight_layout()
    plt.savefig(CFG.ROOT / 'eda_label_dist.png', dpi=120)
    print('EDA 차트 저장: eda_label_dist.png')


def visualize_samples(train_dir: Path, train_csv: Path, n: int = 4):
    """stable / unstable 샘플 각 n개씩 시각화"""
    df        = pd.read_csv(train_csv)
    label_col = 'label' if 'label' in df.columns else 'Label'

    stable_df   = df[df[label_col].str.lower() == 'stable'].sample(n, random_state=42)
    unstable_df = df[df[label_col].str.lower() == 'unstable'].sample(n, random_state=42)
    samples     = pd.concat([stable_df, unstable_df])

    fig, axes = plt.subplots(len(samples), 2, figsize=(8, len(samples) * 3))
    for i, (_, row) in enumerate(samples.iterrows()):
        sid    = row['id']
        label  = row[label_col]
        views  = load_views(train_dir / sid, num_views=2)
        color  = 'green' if label.lower() == 'stable' else 'red'

        for j, img in enumerate(views):
            ax = axes[i][j]
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(f'{sid} [{label}]', color=color, fontsize=9)

    plt.suptitle('Sample Visualization (Left=View0, Right=View1)', fontsize=12)
    plt.tight_layout()
    plt.savefig(CFG.ROOT / 'sample_visualization.png', dpi=120)
    print('샘플 시각화 저장: sample_visualization.png')


# ── 데이터 검증 ───────────────────────────────────────────────────────────────

def validate_data(data_dir: Path, csv_path: Path, split: str = 'train'):
    """모든 샘플의 이미지 파일이 존재하는지 확인"""
    df        = pd.read_csv(csv_path)
    label_col = 'label' if 'label' in df.columns else 'Label'
    missing   = []

    for _, row in df.iterrows():
        sample_dir = data_dir / row['id']
        if not sample_dir.exists():
            missing.append(row['ID'])
            continue
        files = []
        for ext in CFG.IMG_EXTENSIONS:
            files.extend(list(sample_dir.glob(f'*{ext}')))
        if len(files) < CFG.NUM_VIEWS:
            missing.append(f"{row['ID']} (이미지 {len(files)}개, 필요 {CFG.NUM_VIEWS}개)")

    print(f'[{split}] 검증 결과: {len(df) - len(missing)}/{len(df)}개 정상')
    if missing:
        print(f'문제 샘플 ({len(missing)}개):')
        for m in missing[:20]:
            print(f'  - {m}')
    else:
        print('모든 샘플 정상.')


# ── 제출 파일 검증 ─────────────────────────────────────────────────────────────

def validate_submission(sub_path: Path, sample_sub_path: Path):
    sub    = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_sub_path)

    print('=== 제출 파일 검증 ===')
    print(f'행 수: {len(sub)} (기대: {len(sample)})')

    # 컬럼 확인
    if list(sub.columns) != list(sample.columns):
        print(f'⚠ 컬럼 불일치: {list(sub.columns)} vs {list(sample.columns)}')
    else:
        print('컬럼 일치: OK')

    # ID 순서 확인
    if list(sub['ID']) != list(sample['ID']):
        print('⚠ ID 순서 불일치')
    else:
        print('ID 순서: OK')

    # 확률값 범위 확인
    prob_cols = [c for c in sub.columns if c != 'ID']
    for col in prob_cols:
        out_of_range = ((sub[col] < 0) | (sub[col] > 1)).sum()
        if out_of_range:
            print(f'⚠ {col}: 범위 벗어난 값 {out_of_range}개')
        else:
            print(f'{col}: 범위 OK  (min={sub[col].min():.4f}, max={sub[col].max():.4f})')

    # 합산 확인
    if len(prob_cols) == 2:
        row_sum = sub[prob_cols].sum(axis=1)
        if not np.allclose(row_sum, 1.0, atol=1e-5):
            print(f'⚠ 행별 합산 != 1 (최대 오차: {(row_sum - 1).abs().max():.6f})')
        else:
            print('행별 합산 = 1.0: OK')

    print('===================')


if __name__ == '__main__':
    eda(CFG.TRAIN_CSV, CFG.DEV_CSV)
    validate_data(CFG.TRAIN_DIR, CFG.TRAIN_CSV, 'train')
    validate_data(CFG.DEV_DIR,   CFG.DEV_CSV,   'dev')
    visualize_samples(CFG.TRAIN_DIR, CFG.TRAIN_CSV)
