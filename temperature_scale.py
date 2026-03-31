"""
Temperature Scaling (dev/OOF 기반 T 최적화 → test submission 적용)

실행:
    python temperature_scale.py                          # dev 기반 최적 T 탐색 후 적용
    python temperature_scale.py --sub submissions/A.csv  # 특정 submission에 적용
    python temperature_scale.py --T 0.465                # T 고정 적용
    python temperature_scale.py --oof                    # OOF 기반 T 탐색 (GPU 불필요)
    python temperature_scale.py --oof --ckpt checkpoints/20260318_2216  # 특정 체크포인트 OOF
"""

import argparse
import datetime
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import autocast
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from config import CFG
from dataset import StructureDataset, get_val_transform
from model import build_model


@torch.no_grad()
def _predict_single_fold(model, loader, device) -> np.ndarray:
    """단일 fold dev 예측."""
    model.eval()
    preds = []
    for views, _ in loader:
        views = views.to(device)
        video_feats = torch.zeros(views.size(0), CFG.VIDEO_FEAT_DIM, device=device)
        with autocast('cuda', dtype=torch.bfloat16, enabled=CFG.AMP):
            logits = model(views, video_feats)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        preds.append(probs)
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_dev(ckpt_dir: Path, device, folds=None) -> np.ndarray:
    """dev set 예측 (5-fold 앙상블, TTA 없음). folds로 특정 fold만 선택 가능."""
    dev_df = pd.read_csv(CFG.DEV_CSV)
    transform = get_val_transform(CFG.IMAGE_SIZE)
    ds = StructureDataset(dev_df, CFG.DEV_DIR, transform=transform, is_test=True)
    loader = DataLoader(ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    fold_preds = []
    for fold in range(1, CFG.N_FOLDS + 1):
        if folds and fold not in folds:
            print(f'  Fold {fold}: 제외 (--folds)')
            continue

        ckpt = ckpt_dir / f'fold{fold}_phase2.pth'
        if not ckpt.exists():
            ckpt = ckpt_dir / f'fold{fold}_phase1.pth'
        if not ckpt.exists():
            print(f'  Fold {fold}: 체크포인트 없음, 스킵')
            continue

        model = build_model(pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        preds = _predict_single_fold(model, loader, device)
        fold_preds.append(preds)
        del model
        torch.cuda.empty_cache()
        print(f'  Fold {fold}: 예측 완료')

    if not fold_preds:
        raise RuntimeError('사용 가능한 체크포인트 없음')

    # geometric mean ensemble
    log_preds = [np.log(p + 1e-8) for p in fold_preds]
    ensemble = np.exp(np.mean(log_preds, axis=0))
    ensemble /= ensemble.sum(axis=1, keepdims=True)
    return ensemble


@torch.no_grad()
def predict_dev_per_fold(ckpt_dir: Path, device, folds=None) -> dict:
    """각 fold 개별 dev 예측 반환. {fold_num: preds_array}"""
    dev_df = pd.read_csv(CFG.DEV_CSV)
    transform = get_val_transform(CFG.IMAGE_SIZE)
    ds = StructureDataset(dev_df, CFG.DEV_DIR, transform=transform, is_test=True)
    loader = DataLoader(ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    results = {}
    for fold in range(1, CFG.N_FOLDS + 1):
        if folds and fold not in folds:
            continue

        ckpt = ckpt_dir / f'fold{fold}_phase2.pth'
        if not ckpt.exists():
            ckpt = ckpt_dir / f'fold{fold}_phase1.pth'
        if not ckpt.exists():
            print(f'  Fold {fold}: 체크포인트 없음, 스킵')
            continue

        model = build_model(pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        preds = _predict_single_fold(model, loader, device)
        results[fold] = preds
        del model
        torch.cuda.empty_cache()

    return results


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """log-odds 스케일에서 temperature 적용."""
    log_odds = np.log(probs[:, 1] + 1e-10) - np.log(probs[:, 0] + 1e-10)
    p_unstable = 1.0 / (1.0 + np.exp(-log_odds / T))
    return np.stack([1.0 - p_unstable, p_unstable], axis=1)


def find_optimal_T(probs: np.ndarray, labels: np.ndarray,
                   bounds=(0.05, 3.0)) -> float:
    """LogLoss를 최소화하는 T 탐색."""
    def objective(T):
        scaled = apply_temperature(probs, T)
        return log_loss(labels, scaled)

    result = minimize_scalar(objective, bounds=bounds, method='bounded')
    return result.x


def bootstrap_T(probs: np.ndarray, labels: np.ndarray,
                n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap으로 최적 T의 신뢰구간 추정.

    Returns:
        dict with keys: T_median, T_mean, T_std, ci_lower, ci_upper,
                        T_samples (full bootstrap distribution)
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    T_samples = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            T_i = find_optimal_T(probs[idx], labels[idx])
            T_samples.append(T_i)
        except Exception:
            continue

    T_samples = np.array(T_samples)
    return {
        'T_median': np.median(T_samples),
        'T_mean':   np.mean(T_samples),
        'T_std':    np.std(T_samples),
        'ci_lower': np.percentile(T_samples, 2.5),
        'ci_upper': np.percentile(T_samples, 97.5),
        'T_samples': T_samples,
    }


def run_oof_tscale(ckpt_dir: Path):
    """OOF 기반 T-scaling (GPU 불필요). Bootstrap CI 포함."""
    oof_pred_path = ckpt_dir / 'oof_preds.npy'
    oof_label_path = ckpt_dir / 'oof_labels.npy'

    if not oof_pred_path.exists() or not oof_label_path.exists():
        print(f'OOF 파일 없음: {ckpt_dir}')
        return None

    oof_preds = np.load(oof_pred_path)
    oof_labels = np.load(oof_label_path)

    # 유효한 행만 필터 (zeros = 미완성 fold)
    valid = oof_preds.sum(axis=1) > 0
    n_valid = valid.sum()
    n_total = len(oof_labels)
    print(f'OOF: {n_valid}/{n_total}개 유효')

    if n_valid < 50:
        print(f'  유효 샘플 부족 ({n_valid}개), 신뢰도 낮음')
        if n_valid == 0:
            return None

    probs = oof_preds[valid]
    labels = oof_labels[valid]

    # normalize (합이 1이 아닐 수 있음)
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.where(row_sums > 0, row_sums, 1.0)

    # 라벨 분포
    n0, n1 = (labels == 0).sum(), (labels == 1).sum()
    print(f'  라벨 분포: stable={n0}, unstable={n1}')

    # T=1.0 baseline
    ll_base = log_loss(labels, probs)
    print(f'  OOF LogLoss (T=1.0): {ll_base:.6f}')

    # 점추정
    T_opt = find_optimal_T(probs, labels)
    scaled = apply_temperature(probs, T_opt)
    ll_scaled = log_loss(labels, scaled)
    print(f'  점추정 T: {T_opt:.4f} | OOF LogLoss: {ll_base:.6f} → {ll_scaled:.6f}')

    # Bootstrap CI
    print(f'\n  Bootstrap 2000회 진행 중...')
    boot = bootstrap_T(probs, labels)
    print(f'  ──────────────────────────────────────')
    print(f'  T median:  {boot["T_median"]:.4f}')
    print(f'  T mean:    {boot["T_mean"]:.4f} ± {boot["T_std"]:.4f}')
    print(f'  95% CI:    [{boot["ci_lower"]:.4f}, {boot["ci_upper"]:.4f}]')

    # 제출 시 권장 T 범위 (CI 기반, 보수적)
    # 과도한 sharpening 방지: CI 하한과 median 중 큰 값
    T_safe = max(boot['ci_lower'], 0.212)
    T_rec = boot['T_median']
    print(f'\n  ▶ 권장 제출 T: {T_rec:.3f} (범위: {T_safe:.3f} ~ {boot["ci_upper"]:.3f})')

    # Grid search: 주요 T 값별 OOF LogLoss
    print(f'\n  T별 OOF LogLoss:')
    grid_ts = np.arange(0.15, 0.50, 0.01)
    best_grid_t, best_grid_ll = None, float('inf')
    for t in grid_ts:
        sc = apply_temperature(probs, t)
        ll = log_loss(labels, sc)
        if ll < best_grid_ll:
            best_grid_t, best_grid_ll = t, ll
        marker = ' ◀ best' if abs(t - T_opt) < 0.005 else ''
        if 0.20 <= t <= 0.35 or abs(t - T_opt) < 0.005:
            print(f'    T={t:.2f}: {ll:.6f}{marker}')

    return {'T_opt': T_opt, 'boot': boot, 'n_valid': n_valid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=str, default=None, help='적용할 submission CSV')
    parser.add_argument('--T', type=float, default=None, help='T 고정값 (없으면 dev로 탐색)')
    parser.add_argument('--ckpt', type=str, default=None, help='dev 예측용 체크포인트 dir')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='사용할 fold 번호 (예: --folds 1 2 3)')
    parser.add_argument('--per-fold', action='store_true',
                        help='각 fold 개별 dev LogLoss 출력')
    parser.add_argument('--oof', action='store_true',
                        help='OOF 기반 T 탐색 (GPU 불필요, Bootstrap CI 포함)')
    args = parser.parse_args()

    # ckpt_dir 결정
    ckpt_dir = Path(args.ckpt) if args.ckpt else Path('checkpoints/20260309_1110')

    # ── OOF 모드: GPU 불필요, 저장된 OOF로 T 탐색 ──
    if args.oof:
        print(f'OOF 기반 T-scaling ({ckpt_dir})')
        result = run_oof_tscale(ckpt_dir)
        if result and args.sub:
            T_use = result['boot']['T_median']
            print(f'\n  submission에 T={T_use:.3f} 적용 중...')
            ts = datetime.datetime.now().strftime('%m%d_%H%M')
            sub_path = Path(args.sub)
            sub = pd.read_csv(sub_path)
            probs = sub[['stable_prob', 'unstable_prob']].values
            scaled = apply_temperature(probs, T_use)
            sub['stable_prob'] = scaled[:, 0]
            sub['unstable_prob'] = scaled[:, 1]
            out = CFG.SUBMIT_DIR / f'submission_T{T_use:.3f}_{ts}_{sub_path.stem[-9:]}.csv'
            sub.to_csv(out, index=False)
            print(f'  저장: {out.name}')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── per-fold 모드: 각 fold 개별 dev LogLoss 출력 후 종료 ──
    if args.per_fold:
        print(f'Per-fold dev 분석 중... ({ckpt_dir})')
        dev_df = pd.read_csv(CFG.DEV_CSV)
        dev_labels = dev_df['label'].map({'stable': 0, 'unstable': 1}).values

        per_fold_preds = predict_dev_per_fold(ckpt_dir, device, folds=args.folds)
        print(f'\n{"="*55}')
        print(f'  Per-Fold Dev LogLoss  |  {ckpt_dir.name}')
        print(f'{"="*55}')
        for fold_num in sorted(per_fold_preds.keys()):
            preds = per_fold_preds[fold_num]
            ll = log_loss(dev_labels, preds)
            T_opt_f = find_optimal_T(preds, dev_labels)
            scaled = apply_temperature(preds, T_opt_f)
            ll_scaled = log_loss(dev_labels, scaled)
            print(f'  Fold {fold_num}: DevLL={ll:.4f} → T={T_opt_f:.3f} DevLL={ll_scaled:.4f}')

        # 앙상블 결과도 출력
        if len(per_fold_preds) > 1:
            all_preds = list(per_fold_preds.values())
            log_preds = [np.log(p + 1e-8) for p in all_preds]
            ensemble = np.exp(np.mean(log_preds, axis=0))
            ensemble /= ensemble.sum(axis=1, keepdims=True)
            ll_ens = log_loss(dev_labels, ensemble)
            T_opt_ens = find_optimal_T(ensemble, dev_labels)
            scaled_ens = apply_temperature(ensemble, T_opt_ens)
            ll_ens_scaled = log_loss(dev_labels, scaled_ens)
            folds_str = ','.join(str(f) for f in sorted(per_fold_preds.keys()))
            print(f'  ────────────────────────────────────────')
            print(f'  Ensemble({folds_str}): DevLL={ll_ens:.4f} → T={T_opt_ens:.3f} DevLL={ll_ens_scaled:.4f}')
        print()
        return

    # 최적 T 탐색
    if args.T is not None:
        T_opt = args.T
        print(f'T 고정: {T_opt}')
    else:
        print(f'Dev 예측 중... ({ckpt_dir})')
        dev_preds = predict_dev(ckpt_dir, device, folds=args.folds)

        dev_df = pd.read_csv(CFG.DEV_CSV)
        dev_labels = dev_df['label'].map({'stable': 0, 'unstable': 1}).values

        before_ll = log_loss(dev_labels, dev_preds)
        print(f'Dev LogLoss (T=1.0): {before_ll:.4f}')

        T_opt = find_optimal_T(dev_preds, dev_labels)
        scaled_dev = apply_temperature(dev_preds, T_opt)
        after_ll = log_loss(dev_labels, scaled_dev)
        print(f'최적 T: {T_opt:.4f} | Dev LogLoss: {before_ll:.4f} → {after_ll:.4f}')

    # submission에 T 적용
    sub_paths = []
    if args.sub:
        sub_paths = [Path(args.sub)]
    else:
        # 최신 숫자 timestamp submission들에 적용 (stacked 제외)
        sub_paths = sorted(
            CFG.SUBMIT_DIR.glob('submission_[0-9]*.csv'),
            key=lambda p: p.stat().st_mtime
        )[-2:]  # 가장 최신 2개 (ConvNeXt + EfficientNetV2-S)

    ts = datetime.datetime.now().strftime('%m%d_%H%M')
    for sub_path in sub_paths:
        sub = pd.read_csv(sub_path)
        probs = sub[['stable_prob', 'unstable_prob']].values
        scaled = apply_temperature(probs, T_opt)
        sub['stable_prob'] = scaled[:, 0]
        sub['unstable_prob'] = scaled[:, 1]
        out = CFG.SUBMIT_DIR / f'submission_T{T_opt:.3f}_{ts}_{sub_path.stem[-9:]}.csv'
        sub.to_csv(out, index=False)
        print(f'저장: {out.name}')

    print('\n완료!')


if __name__ == '__main__':
    main()
