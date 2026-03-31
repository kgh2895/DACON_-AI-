"""
Video Teacher 모델 학습 (Step 2: Soft Label KD)

simulation.mp4의 마지막 프레임으로 Teacher 분류기 학습.
→ OOF soft labels (train) + 앙상블 soft labels (dev) 생성
→ data/soft_labels.json 저장

실행:
    python train_video_teacher.py

출력:
    data/soft_labels.json  ← train.py가 자동으로 로드
"""

import cv2
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG

# ── 하이퍼파라미터 ─────────────────────────────────────────────────────────────
VT_BACKBONE  = 'efficientnet_b0'    # 경량 모델 (video는 신호가 강해서 작은 모델로 충분)
VT_IMAGE_SIZE = 384
VT_EPOCHS     = 40
VT_LR         = 3e-4
VT_BATCH      = 16
VT_N_FOLDS    = 5
VT_N_FRAMES   = 5                   # 마지막 N 프레임 평균

# ── 마지막 프레임 추출 ─────────────────────────────────────────────────────────

def extract_last_frames(video_path: Path, n_frames: int = VT_N_FRAMES) -> np.ndarray:
    """simulation.mp4에서 마지막 n_frames 프레임을 평균낸 이미지 반환 (H, W, 3)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    start_frame = max(0, total - n_frames)
    for i in range(start_frame, total):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return np.mean(frames, axis=0).astype(np.uint8)


# ── Dataset ───────────────────────────────────────────────────────────────────

def get_transform(is_train: bool):
    if is_train:
        return A.Compose([
            A.Resize(VT_IMAGE_SIZE, VT_IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.3, 0.3, p=0.7),
            A.Affine(translate_percent=(-0.1, 0.1), scale=(0.85, 1.15), rotate=(-15, 15), p=0.6),
            A.OneOf([A.GaussNoise(), A.GaussianBlur(blur_limit=3)], p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(VT_IMAGE_SIZE, VT_IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class VideoFrameDataset(Dataset):
    LABEL2IDX = {'stable': 0, 'unstable': 1}

    def __init__(self, df: pd.DataFrame, data_dir: Path, transform=None, is_test: bool = False):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = Path(data_dir)
        self.transform = transform
        self.is_test   = is_test
        self._cache    = {}   # 메모리 캐시 (프레임 추출 비용 절약)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        sample_id = row['id']

        if sample_id not in self._cache:
            video_path = self.data_dir / sample_id / 'simulation.mp4'
            self._cache[sample_id] = extract_last_frames(video_path)
        img = self._cache[sample_id]

        if self.transform:
            img = self.transform(image=img)['image']
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        if self.is_test:
            return img, sample_id

        label = self.LABEL2IDX[row['label'].lower()]
        return img, torch.tensor(label, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_video_model(pretrained: bool = True) -> nn.Module:
    model = timm.create_model(VT_BACKBONE, pretrained=pretrained, num_classes=2, drop_rate=0.3)
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_vt_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.05)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast('cuda', enabled=True):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_vt_epoch(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        with autocast('cuda', enabled=True):
            logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    probs  = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return log_loss(labels, probs), probs


@torch.no_grad()
def predict_vt(model, loader, device):
    model.eval()
    all_probs = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        with autocast('cuda', enabled=True):
            logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Video Teacher: {VT_BACKBONE} | Epochs: {VT_EPOCHS} | Folds: {VT_N_FOLDS}')

    train_df = pd.read_csv(CFG.TRAIN_CSV)
    dev_df   = pd.read_csv(CFG.DEV_CSV)

    # OOF 예측값 저장 배열 (train용)
    oof_preds = np.zeros((len(train_df), 2))

    skf = StratifiedKFold(n_splits=VT_N_FOLDS, shuffle=True, random_state=CFG.SEED)

    # dev 앙상블 예측 (모든 fold 평균)
    dev_fold_preds = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        print(f'\n── Fold {fold+1}/{VT_N_FOLDS} ──')
        trn_df = train_df.iloc[trn_idx]
        val_df = train_df.iloc[val_idx]

        trn_ds = VideoFrameDataset(trn_df, CFG.TRAIN_DIR, get_transform(True))
        val_ds = VideoFrameDataset(val_df, CFG.TRAIN_DIR, get_transform(False))
        dev_ds = VideoFrameDataset(dev_df, CFG.TRAIN_DIR, get_transform(False))  # dev에는 video 없음

        trn_loader = DataLoader(trn_ds, batch_size=VT_BATCH, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=VT_BATCH * 2, shuffle=False, num_workers=4, pin_memory=True)

        model     = build_video_model(pretrained=True).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=VT_LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=VT_EPOCHS, eta_min=1e-6)
        scaler    = GradScaler('cuda')

        best_val  = float('inf')
        best_preds = None
        ckpt_path = CFG.CKPT_BASE_DIR / f'video_teacher_fold{fold+1}.pth'

        for epoch in range(1, VT_EPOCHS + 1):
            t0       = time.time()
            trn_loss = train_vt_epoch(model, trn_loader, optimizer, scaler, device)
            val_loss, val_preds = val_vt_epoch(model, val_loader, device)
            scheduler.step()

            mark = ' ★' if val_loss < best_val else ''
            print(f'  Epoch {epoch:02d}/{VT_EPOCHS} | Train: {trn_loss:.4f} | Val: {val_loss:.4f} | {time.time()-t0:.1f}s{mark}', flush=True)

            if val_loss < best_val:
                best_val   = val_loss
                best_preds = val_preds
                torch.save(model.state_dict(), ckpt_path)

        # OOF 저장
        oof_preds[val_idx] = best_preds
        oof_logloss = log_loss(train_df.iloc[val_idx]['label'].map({'stable': 0, 'unstable': 1}), best_preds)
        print(f'  Fold {fold+1} OOF LogLoss: {oof_logloss:.4f}')

        del model, trn_ds, val_ds
        torch.cuda.empty_cache()

    # 전체 OOF LogLoss
    all_labels = train_df['label'].map({'stable': 0, 'unstable': 1}).values
    total_oof  = log_loss(all_labels, oof_preds)
    print(f'\nVideo Teacher OOF LogLoss (전체): {total_oof:.4f}')

    # dev 앙상블 예측 (각 fold 모델로 예측)
    print('\nDev set 앙상블 예측 중...')
    dev_preds_list = []
    for fold in range(VT_N_FOLDS):
        ckpt_path = CFG.CKPT_BASE_DIR / f'video_teacher_fold{fold+1}.pth'
        if not ckpt_path.exists():
            continue
        model = build_video_model(pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        dev_ds     = VideoFrameDataset(dev_df, CFG.TRAIN_DIR, get_transform(False))
        dev_loader = DataLoader(dev_ds, batch_size=VT_BATCH * 2, shuffle=False, num_workers=4, pin_memory=True)

        # dev에는 simulation.mp4 없음 → VideoFrameDataset이 파일 없으면 에러
        # dev 디렉토리에서 영상 없는 경우를 처리 (simulation.mp4 존재 여부 확인)
        dev_has_video = (CFG.DEV_DIR / dev_df.iloc[0]['id'] / 'simulation.mp4').exists()
        if not dev_has_video:
            print(f'  Dev set에 simulation.mp4 없음 → dev soft labels 생성 불가')
            break

        preds = predict_vt(model, dev_loader, device)
        dev_preds_list.append(preds)
        del model; torch.cuda.empty_cache()

    # ── soft_labels.json 저장 ─────────────────────────────────────────────────
    soft_labels = {}

    # train OOF predictions
    for i, row in train_df.iterrows():
        soft_labels[row['id']] = oof_preds[i].tolist()   # [p_stable, p_unstable]

    # dev predictions (앙상블 가능한 경우만)
    if dev_preds_list:
        dev_ensemble = np.exp(np.mean([np.log(p + 1e-8) for p in dev_preds_list], axis=0))
        dev_ensemble = dev_ensemble / dev_ensemble.sum(axis=1, keepdims=True)
        for i, row in dev_df.iterrows():
            soft_labels[row['id']] = dev_ensemble[i].tolist()
        print(f'Dev soft labels 저장: {len(dev_df)}개')

    out_path = CFG.DATA_DIR / 'soft_labels.json'
    with open(out_path, 'w') as f:
        json.dump(soft_labels, f, indent=2)
    print(f'\nSoft labels 저장 완료: {out_path}')
    print(f'  총 {len(soft_labels)}개 샘플')


if __name__ == '__main__':
    main()
