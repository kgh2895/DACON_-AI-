"""
학습 파이프라인 v6

개선사항 (v5 대비):
  - SAM (Sharpness-Aware Minimization): 평탄한 minimum → 일반화 개선
  - EMA (Exponential Moving Average): 가중치 지수이동평균 → 노이즈 감소
  - OOF 예측 저장: Phase1 held-out 예측으로 T 최적화 (dev 편향 해소)
"""

import copy
import datetime
import gc
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from torch.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from config import CFG
from dataset import (
    StructureDataset, get_train_transform, get_val_transform,
    make_combined_df, mixup_data, cutmix_data, load_video_cache
)
from pathlib import Path
from model import build_model


# ── EMA ──────────────────────────────────────────────────────────────────────

class ModelEMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.module.state_dict()


# ── SAM helpers ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _sam_perturb(model, rho: float):
    """Compute SAM perturbation from current gradients and apply it."""
    grad_norm = torch.norm(
        torch.stack([p.grad.norm(p=2) for p in model.parameters() if p.grad is not None]),
        p=2,
    )
    for p in model.parameters():
        if p.grad is None:
            continue
        e_w = rho * p.grad / (grad_norm + 1e-12)
        p.add_(e_w)
        p._sam_e_w = e_w


@torch.no_grad()
def _sam_restore(model):
    """Remove SAM perturbation."""
    for p in model.parameters():
        if hasattr(p, '_sam_e_w'):
            p.sub_(p._sam_e_w)
            del p._sam_e_w


def load_soft_labels(path: Path) -> dict:
    """soft_labels.json 로드. 없으면 빈 dict 반환."""
    if not path.exists():
        return {}
    import json
    with open(path) as f:
        data = json.load(f)
    print(f'Soft labels 로드: {len(data)}개 샘플')
    return data


# ── Run 폴더 관리 ────────────────────────────────────────────────────────────

def resolve_run_dir():
    """
    이전에 미완료된 실행이 있으면 재개, 없으면 새 타임스탬프 폴더 생성.
    checkpoints/.current_run 파일에 현재 run ID를 기록.
    """
    base   = CFG.CKPT_BASE_DIR
    marker = base / '.current_run'

    if marker.exists():
        run_id  = marker.read_text().strip()
        run_dir = base / run_id
        if run_dir.exists():
            all_p2_done = all(
                (run_dir / f'fold{i}_phase2.pth').exists()
                for i in range(1, CFG.N_FOLDS + 1)
            )
            has_resume = any(
                (run_dir / f'fold{i}_phase2_resume.pth').exists()
                for i in range(1, CFG.N_FOLDS + 1)
            )
            if not all_p2_done or has_resume:
                print(f'[Resume] 이전 실행 재개 → {run_id}')
                return run_dir

    run_id  = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(run_id)
    print(f'[New Run] 새 실행 시작 → {run_id}')
    return run_dir


# ── 재현성 ────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── 샘플러 ────────────────────────────────────────────────────────────────────

def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce)
        return ((1 - p_t) ** self.gamma * ce).mean()


def mixup_ce_loss(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


def aux_loss_fn(motion_pred, video_feats):
    """
    영상 피처의 첫 번째 차원(motion_score_norm)을 보조 회귀 타깃으로 사용.
    video_feats가 0인 샘플(test/pseudo)은 마스킹.
    """
    motion_target = video_feats[:, 0]   # motion_score_norm
    mask = motion_target > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=motion_pred.device)
    return F.mse_loss(motion_pred[mask].squeeze(-1), motion_target[mask])


# ── Train Epoch ───────────────────────────────────────────────────────────────

def _forward_batch(model, views, labels, video_feats, soft_labels, criterion, device, warmup=False):
    """단일 배치 forward → (cls_loss, aux_loss) 반환. Mixup/CutMix 포함."""
    # Per-sample video feature dropout
    feat_mask = (torch.rand(video_feats.size(0), 1, device=device) > CFG.VIDEO_FEAT_DROP).float()
    video_feats = video_feats * feat_mask

    # Mixup / CutMix (warmup 중 비활성화)
    r = np.random.rand()
    if not warmup and r < CFG.MIXUP_PROB:
        views, la, lb, video_feats, lam = mixup_data(views, labels, video_feats)
        use_mix = True
    elif not warmup and r < CFG.MIXUP_PROB + CFG.CUTMIX_PROB:
        views, la, lb, video_feats, lam = cutmix_data(views, labels, video_feats)
        use_mix = True
    else:
        use_mix = False

    with autocast('cuda', dtype=torch.bfloat16, enabled=CFG.AMP):
        logits, motion_pred = model(views, video_feats=video_feats, training=True)

        if use_mix:
            cls_loss = mixup_ce_loss(criterion, logits, la, lb, lam)
        else:
            hard_loss = criterion(logits, labels)
            valid = soft_labels[:, 0] >= 0
            if valid.any() and CFG.SOFT_LABEL_WEIGHT > 0:
                sl = soft_labels[valid]
                sl = sl / sl.sum(dim=1, keepdim=True)
                kl = F.kl_div(
                    F.log_softmax(logits[valid], dim=1),
                    sl, reduction='batchmean'
                )
                cls_loss = (1 - CFG.SOFT_LABEL_WEIGHT) * hard_loss + CFG.SOFT_LABEL_WEIGHT * kl
            else:
                cls_loss = hard_loss

        aux_loss = aux_loss_fn(motion_pred, video_feats)

    return cls_loss, aux_loss


def train_epoch(model, loader, optimizer, criterion, scaler, device, ema=None, warmup=False):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    use_sam = CFG.USE_SAM

    _t_start = time.time()
    _t_data  = 0.0
    _t_prev  = time.time()

    for step, batch in enumerate(loader):
        _t_data += time.time() - _t_prev
        views, labels, video_feats, soft_labels = batch
        views, labels = views.to(device), labels.to(device)
        video_feats  = video_feats.to(device)
        soft_labels  = soft_labels.to(device)

        cls_loss, aux_loss = _forward_batch(
            model, views, labels, video_feats, soft_labels, criterion, device, warmup=warmup
        )
        loss = (cls_loss + CFG.AUX_WEIGHT * aux_loss) / CFG.GRAD_ACCUM
        scaler.scale(loss).backward()

        is_step = (step + 1) % CFG.GRAD_ACCUM == 0 or (step + 1) == len(loader)

        if is_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if use_sam:
                # ── SAM: perturb → second forward → restore → step ──
                _sam_perturb(model, CFG.SAM_RHO)
                optimizer.zero_grad()

                # Second forward: scaler와 분리 (unscale_ 이후 scaler.scale 금지)
                with autocast('cuda', dtype=torch.bfloat16, enabled=CFG.AMP):
                    logits2, _ = model(views, video_feats=video_feats, training=True)
                    loss2 = criterion(logits2, labels)
                loss2.backward()

                _sam_restore(model)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()

            if ema is not None:
                ema.update(model)

            optimizer.zero_grad()

        total_loss += (cls_loss.item() + CFG.AUX_WEIGHT * aux_loss.item()) * len(labels)
        _t_prev = time.time()

    _t_total = time.time() - _t_start
    print(f'  [Timing] data={_t_data:.0f}s  gpu={_t_total - _t_data:.0f}s  total={_t_total:.0f}s', flush=True)

    return total_loss / len(loader.dataset)


# ── Val Epoch ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for views, labels, _vf, _sl in loader:
        views = views.to(device)
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(views, training=False)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    probs  = np.concatenate(all_probs,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return log_loss(labels, probs), probs


# ── Single Fold Training ──────────────────────────────────────────────────────

def _train_single_fold(fold, trn_idx, val_idx, train_df, video_cache, device,
                        soft_labels, no_warmup, no_focal):
    """단일 fold 학습. Returns (best_loss, oof_probs, val_labels)"""
    print(f'\n── Fold {fold+1}/{CFG.N_FOLDS} ──')
    trn_df = train_df.iloc[trn_idx]
    val_df = train_df.iloc[val_idx]

    trn_ds = StructureDataset(trn_df, CFG.TRAIN_DIR, get_train_transform(CFG.IMAGE_SIZE),
                              video_cache=video_cache, soft_labels=soft_labels)
    val_ds = StructureDataset(val_df, CFG.TRAIN_DIR, get_val_transform(CFG.IMAGE_SIZE),
                              video_cache=video_cache)

    label_idx = [StructureDataset.LABEL2IDX[l.lower()] for l in trn_df['label'].tolist()]
    sampler   = make_weighted_sampler(label_idx)

    trn_loader = DataLoader(trn_ds, batch_size=CFG.BATCH_SIZE, sampler=sampler,
                            num_workers=CFG.NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                            num_workers=2, pin_memory=True,
                            persistent_workers=False)

    WARMUP_EPOCHS = 0 if no_warmup else 5
    model     = build_model(pretrained=True).to(device)
    ema       = ModelEMA(model, decay=CFG.EMA_DECAY) if CFG.USE_EMA else None
    focal_criterion = FocalLoss(gamma=2.0)
    ce_criterion    = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    if no_warmup:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.PHASE1_COSINE_TMAX, eta_min=CFG.MIN_LR
        )
    else:
        warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.PHASE1_COSINE_TMAX - WARMUP_EPOCHS, eta_min=CFG.MIN_LR
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS]
        )
    scaler    = GradScaler('cuda', enabled=CFG.AMP)

    best_loss    = float('inf')
    patience_cnt = 0
    start_epoch  = 1
    ckpt_path   = CFG.CKPT_DIR / f'fold{fold+1}_phase1.pth'
    resume_path = CFG.CKPT_DIR / f'fold{fold+1}_phase1_resume.pth'

    # ── 중단된 fold 재개 ────────────────────────────────────────────────
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        best_loss    = state['best_loss']
        patience_cnt = state.get('patience_cnt', 0)
        start_epoch  = state['epoch'] + 1
        if ema is not None and 'ema' in state:
            ema.module.load_state_dict(state['ema'])
        print(f'  [Resume] Fold {fold+1} epoch {start_epoch}부터 재개 (best={best_loss:.4f})')

    for epoch in range(start_epoch, CFG.PHASE1_EPOCHS + 1):
        is_warmup = (not no_warmup) and (epoch <= WARMUP_EPOCHS)
        criterion = ce_criterion if (is_warmup or no_focal) else focal_criterion
        t0       = time.time()
        trn_loss = train_epoch(model, trn_loader, optimizer, criterion, scaler, device, ema=ema, warmup=is_warmup)

        # Validation: EMA 모델로 평가 (있으면)
        eval_model = ema.module if ema is not None else model
        val_loss, val_probs = val_epoch(eval_model, val_loader, device)
        if epoch <= CFG.PHASE1_COSINE_TMAX:
            scheduler.step()
        # epoch > PHASE1_COSINE_TMAX: LR = MIN_LR 유지

        mark = ' ★' if val_loss < best_loss else ''
        wu_tag = ' [WU]' if is_warmup else ''
        print(f'  Epoch {epoch:02d}/{CFG.PHASE1_EPOCHS} | '
              f'Train: {trn_loss:.4f} | Val: {val_loss:.4f} | '
              f'{time.time()-t0:.1f}s{mark}{wu_tag}', flush=True)

        if val_loss < best_loss:
            best_loss    = val_loss
            patience_cnt = 0
            # EMA 가중치 저장 (있으면), 없으면 일반 모델
            save_state = ema.state_dict() if ema is not None else model.state_dict()
            torch.save(save_state, ckpt_path)
        elif not is_warmup:
            patience_cnt += 1
            if patience_cnt >= CFG.PHASE1_PATIENCE:
                print(f'  Early stopping at epoch {epoch} (patience={CFG.PHASE1_PATIENCE})', flush=True)
                break

        # 조기 collapse 감지
        if epoch >= CFG.FOLD_COLLAPSE_CHECK_EPOCH and best_loss >= CFG.FOLD_COLLAPSE_THRESHOLD:
            print(f'  *** FOLD COLLAPSE 감지 (epoch {epoch}, best_val={best_loss:.4f} >= {CFG.FOLD_COLLAPSE_THRESHOLD}) ***')
            break

        # resume 체크포인트 저장 (매 epoch) — GC로 메모리 확보 후 저장
        gc.collect()
        torch.cuda.empty_cache()
        resume_state = {
            'epoch':        epoch,
            'model':        model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'scheduler':    scheduler.state_dict(),
            'scaler':       scaler.state_dict(),
            'best_loss':    best_loss,
            'patience_cnt': patience_cnt,
        }
        if ema is not None:
            resume_state['ema'] = ema.state_dict()
        torch.save(resume_state, resume_path)
        del resume_state
        gc.collect()

    # ── OOF 예측 수집 (best 체크포인트로) ────────────────────────────────
    if ckpt_path.exists():
        best_model = build_model(pretrained=False).to(device)
        best_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        _, oof_probs = val_epoch(best_model, val_loader, device)
        del best_model
    else:
        # collapse로 best 체크포인트가 없는 경우
        oof_probs = np.full((len(val_idx), CFG.NUM_CLASSES), 1.0 / CFG.NUM_CLASSES)

    val_labels = [StructureDataset.LABEL2IDX[l.lower()] for l in val_df['label'].tolist()]

    print(f'  Best Val LogLoss: {best_loss:.4f} → {ckpt_path}', flush=True)
    resume_path.unlink(missing_ok=True)
    del model, trn_loader, val_loader
    if ema is not None:
        del ema
    gc.collect()
    torch.cuda.empty_cache()

    return best_loss, oof_probs, val_labels


# ── Phase 1: K-Fold ───────────────────────────────────────────────────────────

def train_phase1(train_df: pd.DataFrame, video_cache: dict, device: torch.device, soft_labels: dict = None, only_folds: list = None, no_warmup: bool = False, no_focal: bool = False):
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

    print(f'\n{"="*60}')
    print(f'  Phase 1 — {CFG.N_FOLDS}-Fold CV  |  Epochs: {CFG.PHASE1_EPOCHS}')
    print(f'  Backbone: {CFG.BACKBONE}  |  ImgSize: {CFG.IMAGE_SIZE}')
    print(f'  SAM: {CFG.USE_SAM} (rho={CFG.SAM_RHO})  EMA: {CFG.USE_EMA} (decay={CFG.EMA_DECAY})')
    print(f'  Mixup: {CFG.MIXUP_PROB}  CutMix: {CFG.CUTMIX_PROB}  AuxW: {CFG.AUX_WEIGHT}')
    if no_warmup or no_focal:
        flags = []
        if no_warmup: flags.append('no-warmup')
        if no_focal:  flags.append('no-focal (CE only)')
        print(f'  Flags: {", ".join(flags)}')
    else:
        print(f'  Warmup: 5 epochs (CE + no Mixup/CutMix + LR 0.01→1.0)')
    if only_folds:
        print(f'  대상 Fold: {only_folds}')
    print(f'{"="*60}')

    # OOF predictions 수집
    oof_preds  = np.zeros((len(train_df), CFG.NUM_CLASSES))
    oof_labels = np.zeros(len(train_df), dtype=int)

    for fold, (trn_idx, val_idx) in enumerate(
        skf.split(train_df, train_df['label'])
    ):
        if only_folds and (fold + 1) not in only_folds:
            print(f'\n── Fold {fold+1}/{CFG.N_FOLDS} ── [스킵]')
            continue

        for retry in range(CFG.FOLD_MAX_RETRIES + 1):
            if retry > 0:
                print(f'\n  *** FOLD {fold+1} RETRY {retry}/{CFG.FOLD_MAX_RETRIES} ***')
                (CFG.CKPT_DIR / f'fold{fold+1}_phase1.pth').unlink(missing_ok=True)
                (CFG.CKPT_DIR / f'fold{fold+1}_phase1_resume.pth').unlink(missing_ok=True)
                seed_everything(CFG.SEED + (retry * 1000) + fold)

            best_loss, fold_oof_probs, fold_val_labels = _train_single_fold(
                fold, trn_idx, val_idx, train_df, video_cache, device,
                soft_labels, no_warmup, no_focal
            )

            if best_loss <= CFG.FOLD_QUALITY_THRESHOLD:
                print(f'  Fold {fold+1} PASS (best_val={best_loss:.4f})')
                break
            elif retry < CFG.FOLD_MAX_RETRIES:
                print(f'  Fold {fold+1} FAIL (best_val={best_loss:.4f}) → 재시도')
            else:
                print(f'  Fold {fold+1} MAX RETRIES 도달. best_val={best_loss:.4f}')

        seed_everything(CFG.SEED)

        oof_preds[val_idx] = fold_oof_probs
        oof_labels[val_idx] = fold_val_labels

    # ── OOF 저장 및 T 최적화 ────────────────────────────────────────────────
    oof_path = CFG.CKPT_DIR / 'oof_preds.npy'
    np.save(oof_path, oof_preds)
    np.save(CFG.CKPT_DIR / 'oof_labels.npy', oof_labels)

    oof_ll = log_loss(oof_labels, oof_preds)
    print(f'\n  OOF LogLoss (T=1.0): {oof_ll:.4f}')

    # OOF 기반 최적 T 탐색
    from temperature_scale import apply_temperature
    from scipy.optimize import minimize_scalar
    def obj(T):
        scaled = apply_temperature(oof_preds, T)
        return log_loss(oof_labels, scaled)
    result = minimize_scalar(obj, bounds=(0.15, 2.0), method='bounded')
    T_opt = result.x
    oof_ll_opt = result.fun
    print(f'  OOF 최적 T: {T_opt:.4f} | OOF LogLoss: {oof_ll:.4f} → {oof_ll_opt:.4f}')
    print(f'  OOF predictions 저장: {oof_path}')



# ── Pseudo Labeling ───────────────────────────────────────────────────────────

@torch.no_grad()
def generate_pseudo_labels(video_cache: dict, device: torch.device) -> pd.DataFrame:
    """Phase 1 체크포인트로 test set pseudo labels 생성."""
    sample_sub = pd.read_csv(CFG.SAMPLE_SUB)
    for col in ['id', 'ID', 'Id']:
        if col in sample_sub.columns:
            test_ids = sample_sub[col].tolist()
            break
    test_df = pd.DataFrame({'id': test_ids})

    fold_preds = []
    for fold in range(1, CFG.N_FOLDS + 1):
        ckpt_path = CFG.CKPT_DIR / f'fold{fold}_phase1.pth'
        if not ckpt_path.exists():
            continue
        model = build_model(pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        ds     = StructureDataset(test_df, CFG.TEST_DIR, get_val_transform(CFG.IMAGE_SIZE), is_test=True)
        loader = DataLoader(ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                            num_workers=CFG.NUM_WORKERS, pin_memory=True)
        preds = []
        for views, _ in loader:
            views       = views.to(device)
            video_feats = torch.zeros(views.size(0), CFG.VIDEO_FEAT_DIM, device=device)
            with autocast('cuda', dtype=torch.bfloat16, enabled=CFG.AMP):
                logits = model(views, video_feats=video_feats)
            preds.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        fold_preds.append(np.concatenate(preds, axis=0))
        del model; torch.cuda.empty_cache()

    if not fold_preds:
        return pd.DataFrame(columns=['id', 'label', 'source'])

    ensemble  = np.exp(np.mean([np.log(p + 1e-8) for p in fold_preds], axis=0))
    ensemble  = ensemble / ensemble.sum(axis=1, keepdims=True)
    max_probs = ensemble.max(axis=1)
    pred_lbl  = ensemble.argmax(axis=1)
    high_conf = max_probs >= CFG.PSEUDO_CONF

    idx2label = {0: 'stable', 1: 'unstable'}
    records = [
        {'id': test_ids[i], 'label': idx2label[pred_lbl[i]], 'source': 'pseudo'}
        for i in range(len(test_ids)) if high_conf[i]
    ]
    pseudo_df = pd.DataFrame(records)
    print(f'\n[Pseudo Labeling] {len(pseudo_df)}/{len(test_ids)}개 선택 (conf≥{CFG.PSEUDO_CONF})')
    stable_cnt   = (pseudo_df['label'] == 'stable').sum() if len(pseudo_df) else 0
    unstable_cnt = (pseudo_df['label'] == 'unstable').sum() if len(pseudo_df) else 0
    print(f'  stable={stable_cnt}, unstable={unstable_cnt}')
    return pseudo_df

# ── Phase 2: Domain Fine-Tune ────────────────────────────────────────────────

def train_phase2(video_cache: dict, pseudo_df: pd.DataFrame, device: torch.device, soft_labels: dict = None, only_folds: list = None, no_focal: bool = False):
    combined_df = make_combined_df(CFG.TRAIN_CSV, CFG.DEV_CSV)
    if len(pseudo_df) > 0:
        combined_df = pd.concat([combined_df, pseudo_df], ignore_index=True)

    # ── Phase 2 validation split (stratified 10%) ─────────────────────────────
    from sklearn.model_selection import train_test_split
    p2_train_df, p2_val_df = train_test_split(
        combined_df, test_size=CFG.PHASE2_VAL_RATIO,
        stratify=combined_df['label'], random_state=CFG.SEED
    )

    loss_name = 'CE Loss' if no_focal else 'FocalLoss'
    print(f'\n{"="*60}')
    print(f'  Phase 2 — Domain Fine-Tune  |  Epochs: {CFG.PHASE2_EPOCHS}  |  {loss_name}')
    print(f'  Cosine T_max: {CFG.PHASE2_COSINE_TMAX}  |  patience: {CFG.PHASE2_PATIENCE}')
    print(f'  Train: {len(p2_train_df)}개  Val: {len(p2_val_df)}개')
    if only_folds:
        print(f'  대상 Fold: {only_folds}')
    print(f'{"="*60}')

    for fold in range(CFG.N_FOLDS):
        if only_folds and (fold + 1) not in only_folds:
            print(f'\n── Fold {fold+1}/{CFG.N_FOLDS} ── [스킵]')
            continue
        ckpt_in  = CFG.CKPT_DIR / f'fold{fold+1}_phase1.pth'
        ckpt_out = CFG.CKPT_DIR / f'fold{fold+1}_phase2.pth'
        if not ckpt_in.exists():
            print(f'  Fold {fold+1}: Phase 1 체크포인트 없음, 스킵')
            continue

        print(f'\n── Fold {fold+1}/{CFG.N_FOLDS} Fine-Tune ──')
        model = build_model(pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt_in, map_location=device))
        ema = ModelEMA(model, decay=CFG.EMA_DECAY) if CFG.USE_EMA else None

        # train/val dataset 구성
        def make_p2_ds(df, is_aug):
            trn_part    = df[df['source'] == 'train']
            dev_part    = df[df['source'] == 'dev']
            pseudo_part = df[df['source'] == 'pseudo']
            tfm = get_train_transform(CFG.IMAGE_SIZE) if is_aug else get_val_transform(CFG.IMAGE_SIZE)
            datasets = []
            if len(trn_part) > 0:
                datasets.append(StructureDataset(trn_part, CFG.TRAIN_DIR, tfm,
                                                 video_cache=video_cache, soft_labels=soft_labels))
            if len(dev_part) > 0:
                dev_tfm = get_train_transform(CFG.IMAGE_SIZE) if is_aug else get_val_transform(CFG.IMAGE_SIZE)
                datasets.append(StructureDataset(dev_part, CFG.DEV_DIR, dev_tfm, soft_labels=soft_labels))
            if len(pseudo_part) > 0:
                datasets.append(StructureDataset(pseudo_part, CFG.TEST_DIR, tfm))
            return ConcatDataset(datasets) if datasets else None

        trn_ds = make_p2_ds(p2_train_df, is_aug=True)
        val_ds = make_p2_ds(p2_val_df,   is_aug=False)

        if len(pseudo_df) > 0:
            print(f'  Pseudo {len(pseudo_df)}개 포함')

        trn_loader = DataLoader(trn_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                                num_workers=CFG.NUM_WORKERS, pin_memory=True,
                                persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                                num_workers=2, pin_memory=True,
                                persistent_workers=False)

        criterion   = nn.CrossEntropyLoss() if no_focal else FocalLoss(gamma=2.0)
        optimizer   = optim.AdamW(model.parameters(), lr=CFG.PHASE2_LR, weight_decay=CFG.WEIGHT_DECAY)
        scheduler   = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.PHASE2_COSINE_TMAX, eta_min=CFG.MIN_LR
        )
        scaler      = GradScaler('cuda', enabled=CFG.AMP)
        resume_path = CFG.CKPT_DIR / f'fold{fold+1}_phase2_resume.pth'
        start_epoch = 1
        best_val    = float('inf')
        patience_cnt = 0

        # ── 중단된 Phase2 재개 ──────────────────────────────────────────────
        if resume_path.exists():
            state = torch.load(resume_path, map_location=device)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            scaler.load_state_dict(state['scaler'])
            start_epoch  = state['epoch'] + 1
            best_val     = state.get('best_val', float('inf'))
            patience_cnt = state.get('patience_cnt', 0)
            print(f'  [Resume] Phase2 Fold {fold+1} epoch {start_epoch}부터 재개 (best_val={best_val:.4f})')

        for epoch in range(start_epoch, CFG.PHASE2_EPOCHS + 1):
            t0       = time.time()
            trn_loss = train_epoch(model, trn_loader, optimizer, criterion, scaler, device, ema=ema)
            eval_model = ema.module if ema is not None else model
            val_loss, _ = val_epoch(eval_model, val_loader, device)
            if epoch <= CFG.PHASE2_COSINE_TMAX:
                scheduler.step()
            # epoch > PHASE2_COSINE_TMAX: LR = MIN_LR 유지

            improved = val_loss < best_val
            mark = ' ★' if improved else ''
            print(f'  Epoch {epoch:02d}/{CFG.PHASE2_EPOCHS} | Train: {trn_loss:.4f} | Val: {val_loss:.4f} | {time.time()-t0:.1f}s{mark}', flush=True)

            if improved:
                best_val = val_loss
                patience_cnt = 0
                save_state = ema.state_dict() if ema is not None else model.state_dict()
                torch.save(save_state, ckpt_out)
            else:
                patience_cnt += 1
                if patience_cnt >= CFG.PHASE2_PATIENCE:
                    print(f'  Early stopping (patience={CFG.PHASE2_PATIENCE})')
                    break

            gc.collect()
            torch.cuda.empty_cache()
            resume_state = {
                'epoch':       epoch,
                'model':       model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'scheduler':   scheduler.state_dict(),
                'scaler':      scaler.state_dict(),
                'best_val':    best_val,
                'patience_cnt': patience_cnt,
            }
            if ema is not None:
                resume_state['ema'] = ema.state_dict()
            torch.save(resume_state, resume_path)
            del resume_state
            gc.collect()

        if not ckpt_out.exists():
            save_state = ema.state_dict() if ema is not None else model.state_dict()
            torch.save(save_state, ckpt_out)
        print(f'  Best Val: {best_val:.4f} → {ckpt_out}', flush=True)
        resume_path.unlink(missing_ok=True)
        del model, trn_loader, val_loader
        if ema is not None:
            del ema
        gc.collect()
        torch.cuda.empty_cache()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='재학습할 fold 번호 (예: --folds 1 4 5). 미지정 시 전체 실행.')
    parser.add_argument('--phase1-only', action='store_true',
                        help='Phase 1만 실행하고 종료.')
    parser.add_argument('--phase2-only', action='store_true',
                        help='Phase 2만 실행 (Phase 1 체크포인트 필요).')
    parser.add_argument('--seed', type=int, default=None,
                        help='랜덤 시드 override (예: --seed 44). 미지정 시 config.py SEED 사용.')
    parser.add_argument('--no-warmup', action='store_true',
                        help='Warmup 없이 CosineAnnealingLR만 사용.')
    parser.add_argument('--no-focal', action='store_true',
                        help='Focal Loss 대신 CE Loss만 사용.')
    args = parser.parse_args()

    if args.seed is not None:
        CFG.SEED = args.seed
    seed_everything(CFG.SEED)
    device = torch.device(CFG.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Run 폴더 설정 (타임스탬프 기반 버전 관리) ───────────────────────────
    CFG.CKPT_DIR = resolve_run_dir()
    print(f'Checkpoint dir: {CFG.CKPT_DIR}')

    # 영상 특징 캐시 로드 (preprocess_video.py 선행 필요)
    video_cache = load_video_cache(CFG.VIDEO_CACHE)
    if video_cache:
        print(f'영상 특징 로드: {len(video_cache)}개 샘플')
    else:
        print('영상 특징 캐시 없음 — preprocess_video.py를 먼저 실행하세요')

    train_df = pd.read_csv(CFG.TRAIN_CSV)
    dev_df   = pd.read_csv(CFG.DEV_CSV)

    soft_labels = load_soft_labels(CFG.SOFT_LABEL_PATH)
    if soft_labels:
        print(f'Soft Label KD 활성화 (weight={CFG.SOFT_LABEL_WEIGHT})')

    if not args.phase2_only:
        train_phase1(train_df, video_cache, device, soft_labels=soft_labels, only_folds=args.folds,
                     no_warmup=args.no_warmup, no_focal=args.no_focal)
    if not args.phase1_only:
        pseudo_df = generate_pseudo_labels(video_cache, device)
        train_phase2(video_cache, pseudo_df, device, soft_labels=soft_labels,
                     only_folds=args.folds, no_focal=args.no_focal)

    # 완료 마킹 (다음 실행 시 새 run 폴더 생성)
    (CFG.CKPT_BASE_DIR / '.current_run').unlink(missing_ok=True)
    print(f'\n학습 완료. 체크포인트: {CFG.CKPT_DIR}')
    print('python inference.py 실행하여 제출 파일 생성하세요.')


if __name__ == '__main__':
    main()
