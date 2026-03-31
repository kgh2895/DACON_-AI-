"""
추론 파이프라인

1. 모든 fold의 Phase 2 체크포인트 로드 (없으면 Phase 1 fallback)
2. TTA(Test-Time Augmentation): 각 fold × TTA_TIMES 예측 평균
3. 앙상블 결과로 sample_submission 형식에 맞춰 CSV 저장

제출 형식 (확인됨):
    id, unstable_prob, stable_prob
    TEST_0001, 0.15, 0.85
    ...
"""

import argparse
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from config import CFG
from dataset import StructureDataset, get_val_transform, get_tta_transform
from model import build_model


# ── TTA 추론 ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_tta(model, test_df, device, tta_times: int = CFG.TTA_TIMES) -> np.ndarray:
    """
    TTA_TIMES번 랜덤 aug를 적용한 예측의 평균을 반환한다.
    첫 1회는 aug 없는 clean 예측, 이후는 TTA.
    Returns: (N, NUM_CLASSES) 확률 배열
    """
    model.eval()
    all_rounds = []

    for t in range(tta_times):
        transform = (
            get_val_transform(CFG.IMAGE_SIZE)      # 0번째: clean
            if t == 0
            else get_tta_transform(CFG.IMAGE_SIZE)  # 이후: TTA
        )
        ds = StructureDataset(
            test_df, CFG.TEST_DIR, transform=transform, is_test=True
        )
        loader = DataLoader(
            ds,
            batch_size=CFG.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=True,
        )

        round_probs = []
        for views, _ in loader:
            views       = views.to(device)
            video_feats = torch.zeros(views.size(0), CFG.VIDEO_FEAT_DIM, device=device)
            with autocast('cuda', dtype=torch.bfloat16, enabled=CFG.AMP):
                logits = model(views, video_feats=video_feats)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            round_probs.append(probs)

        all_rounds.append(np.concatenate(round_probs, axis=0))

    # TTA 평균 (log-space averaging이 더 안정적)
    log_probs = [np.log(p + 1e-8) for p in all_rounds]
    avg_log   = np.mean(log_probs, axis=0)
    avg_probs = np.exp(avg_log)
    avg_probs = avg_probs / avg_probs.sum(axis=1, keepdims=True)   # renormalize
    return avg_probs


# ── ID 순서 보존 추론 ─────────────────────────────────────────────────────────

def get_sample_ids(test_df) -> list[str]:
    """test_df에서 ID 컬럼 추출 (컬럼명 유연하게 처리)"""
    for col in ['ID', 'id', 'Id']:
        if col in test_df.columns:
            return test_df[col].tolist()
    raise KeyError('ID 컬럼을 찾을 수 없습니다. sample_submission.csv를 확인하세요.')


# ── 메인 추론 ─────────────────────────────────────────────────────────────────

def resolve_ckpt_dir():
    """
    .current_run 마커 → 해당 run 폴더 사용.
    없으면 checkpoints/ 하위 폴더 중 가장 최신 것 사용.
    """
    base   = CFG.CKPT_BASE_DIR
    marker = base / '.current_run'

    if marker.exists():
        run_id  = marker.read_text().strip()
        run_dir = base / run_id
        if run_dir.exists():
            return run_dir

    # 마커 없으면 타임스탬프 이름(YYYYMMDD_HHMM) 폴더 중 최신 것
    candidates = sorted(
        [d for d in base.iterdir() if d.is_dir() and len(d.name) == 13],
        reverse=True
    )
    if candidates:
        return candidates[0]

    return base  # fallback: 기존 flat 구조



def resolve_ckpt_dirs() -> list:
    """메인 dir + CFG.EXTRA_CKPT_DIRS 합쳐서 반환 (헤테로 앙상블용)."""
    dirs = [resolve_ckpt_dir()]
    from pathlib import Path as _Path
    for extra in CFG.EXTRA_CKPT_DIRS:
        d = _Path(extra)
        if d.exists():
            dirs.append(d)
    return dirs

def run_inference(args=None):
    device  = torch.device(CFG.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args is not None and args.ckpt:
        from pathlib import Path as _Path
        ckpt_dirs = [_Path(args.ckpt)]
    else:
        ckpt_dirs = resolve_ckpt_dirs()
    CFG.CKPT_DIR = ckpt_dirs[0]
    print(f'Checkpoint dirs: {ckpt_dirs}')

    # sample_submission으로 test ID 목록 파악
    sample_sub = pd.read_csv(CFG.SAMPLE_SUB)
    test_ids   = get_sample_ids(sample_sub)
    test_df    = pd.DataFrame({'id': test_ids})

    fold_preds = []

    for ckpt_dir in ckpt_dirs:
      for fold in range(1, CFG.N_FOLDS + 1):
        if args is not None and args.folds and fold not in args.folds:
            print(f'  Fold {fold}: 제외 (--folds)')
            continue

        # Phase 2 체크포인트 우선, 없으면 Phase 1 fallback
        ckpt_p2 = ckpt_dir / f'fold{fold}_phase2.pth'
        ckpt_p1 = ckpt_dir / f'fold{fold}_phase1.pth'

        phase1_only = args is not None and args.phase1_only
        if not phase1_only and ckpt_p2.exists():
            ckpt_path = ckpt_p2
            tag       = 'phase2'
        elif ckpt_p1.exists():
            ckpt_path = ckpt_p1
            tag       = 'phase1'
        else:
            print(f'  Fold {fold}: 체크포인트 없음, 스킵')
            continue

        print(f'\n── Fold {fold} ({tag}) TTA×{CFG.TTA_TIMES} 추론 ──')
        model = build_model(pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        preds = predict_tta(model, test_df, device)
        fold_preds.append(preds)
        print(f'  예측 완료: {preds.shape}')
        del model; torch.cuda.empty_cache()

    if not fold_preds:
        raise RuntimeError('사용 가능한 체크포인트가 없습니다. train.py를 먼저 실행하세요.')

    # Fold 앙상블 (log-space geometric mean)
    log_preds  = [np.log(p + 1e-8) for p in fold_preds]
    ensemble   = np.exp(np.mean(log_preds, axis=0))
    ensemble   = ensemble / ensemble.sum(axis=1, keepdims=True)

    # ── 제출 파일 생성 ────────────────────────────────────────────────────────
    # 실제 제출 컬럼: id, unstable_prob, stable_prob
    # LABEL2IDX: stable=0, unstable=1
    PROB_COL_MAP = {
        'unstable_prob': 1,   # ensemble[:, 1]
        'stable_prob':   0,   # ensemble[:, 0]
        'unstable':      1,
        'stable':        0,
    }

    sub_cols = sample_sub.columns.tolist()
    print(f'\n제출 컬럼: {sub_cols}')

    submission = pd.DataFrame({'id': test_ids})

    for col in sub_cols:
        if col == 'id':
            continue
        if col in PROB_COL_MAP:
            submission[col] = ensemble[:, PROB_COL_MAP[col]]
        else:
            print(f'  ⚠ 알 수 없는 컬럼 "{col}" → 0.5로 채움')
            submission[col] = 0.5

    # 저장
    ts         = datetime.datetime.now().strftime('%m%d_%H%M')
    out_path   = CFG.SUBMIT_DIR / f'submission_{ts}.csv'
    submission.to_csv(out_path, index=False)
    print(f'\n제출 파일 저장 완료: {out_path}')
    print(submission.head(10))
    return submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase1-only', action='store_true', help='Phase1 체크포인트만 사용')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='사용할 fold 번호 (예: --folds 1 2 3)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='체크포인트 dir (예: checkpoints/20260312_0131)')
    args = parser.parse_args()
    run_inference(args)
