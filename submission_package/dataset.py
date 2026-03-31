import cv2
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG

# 영상 피처 키 목록 (preprocess_video.py와 순서 일치해야 함)
VIDEO_FEAT_KEYS = [
    'motion_score_norm', 'frame_diff_norm', 'max_flow_norm',
    'dominant_freq_norm', 'motion_accel_norm', 'collapse_frame_norm',
    'top_motion_norm', 'bottom_motion_norm',
]


# ── 이미지 로더 ────────────────────────────────────────────────────────────────

def load_views(sample_dir: Path, num_views: int = 2) -> list[np.ndarray]:
    """front.png(정면), top.png(위) 순서로 로드."""
    view_names = ['front.png', 'top.png']
    imgs = []
    for name in view_names[:num_views]:
        path = sample_dir / name
        img  = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return imgs


# ── 영상 특징 캐시 로드 ────────────────────────────────────────────────────────

def load_video_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Augmentation ──────────────────────────────────────────────────────────────

def get_train_transform(image_size: int):
    return A.Compose([
        A.Resize(image_size, image_size),

        # 공간 변환
        A.Affine(
            translate_percent=(-0.1, 0.1), scale=(0.85, 1.15), rotate=(-15, 15), p=0.7
        ),
        A.Perspective(scale=(0.05, 0.15), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),

        # 광원/색상 변화
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
            A.RandomGamma(gamma_limit=(60, 140)),
            A.CLAHE(clip_limit=4.0),
        ], p=0.8),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=0.5),
        A.RandomShadow(p=0.3),

        # 노이즈 & 블러
        A.OneOf([
            A.GaussNoise(),
            A.ISONoise(color_shift=(0.01, 0.05)),
        ], p=0.4),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.Sharpen(alpha=(0.2, 0.5), p=1.0),
        ], p=0.3),

        # 차단
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 48),
            hole_width_range=(8, 48),
            p=0.35
        ),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_tta_transform(image_size: int):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-10, 10), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.Perspective(scale=(0.03, 0.08), p=1.0),
            A.GridDistortion(num_steps=3, distort_limit=0.2, p=1.0),
        ], p=0.9),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class StructureDataset(Dataset):
    LABEL2IDX = {'stable': 0, 'unstable': 1}

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        transform=None,
        is_test: bool = False,
        video_cache: dict = None,
        soft_labels: dict = None,
    ):
        self.df          = df.reset_index(drop=True)
        self.data_dir    = Path(data_dir)
        self.transform   = transform
        self.is_test     = is_test
        self.video_cache = video_cache or {}
        self.soft_labels = soft_labels or {}   # {sample_id: [p_stable, p_unstable]}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        sample_id  = row['id']
        sample_dir = self.data_dir / sample_id

        views = load_views(sample_dir, num_views=CFG.NUM_VIEWS)

        tensors = []
        for img in views:
            aug = self.transform(image=img)['image'] if self.transform else \
                  torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            tensors.append(aug)
        views_tensor = torch.stack(tensors, dim=0)   # (V, C, H, W)

        # 영상 피처 (train만 존재, 없으면 0 — test/pseudo 포함)
        vfeat = self.video_cache.get(sample_id, {})
        video_feats = torch.tensor(
            [vfeat.get(k, 0.0) for k in VIDEO_FEAT_KEYS], dtype=torch.float32
        )  # (VIDEO_FEAT_DIM,)

        if self.is_test:
            return views_tensor, sample_id

        label = self.LABEL2IDX[row['label'].lower()]

        # soft label: [p_stable, p_unstable], 없으면 sentinel [-1, -1]
        if sample_id in self.soft_labels:
            soft = torch.tensor(self.soft_labels[sample_id], dtype=torch.float32)
        else:
            soft = torch.tensor([-1.0, -1.0], dtype=torch.float32)

        return views_tensor, torch.tensor(label, dtype=torch.long), video_feats, soft


# ── Mixup / CutMix ────────────────────────────────────────────────────────────

def mixup_data(views, labels, video_feats, alpha: float = CFG.MIXUP_ALPHA):
    """
    views      : (B, V, C, H, W)
    labels     : (B,) LongTensor
    video_feats: (B, VIDEO_FEAT_DIM) FloatTensor
    """
    lam   = np.random.beta(alpha, alpha)
    index = torch.randperm(views.size(0), device=views.device)
    mixed_views      = lam * views + (1 - lam) * views[index]
    mixed_video_feats = lam * video_feats + (1 - lam) * video_feats[index]
    return mixed_views, labels, labels[index], mixed_video_feats, lam


def cutmix_data(views, labels, video_feats, alpha: float = CFG.CUTMIX_ALPHA):
    """
    같은 패치를 모든 뷰에 적용하여 물리적 일관성 유지.
    """
    lam   = np.random.beta(alpha, alpha)
    B, V, C, H, W = views.shape
    index = torch.randperm(B, device=views.device)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0);  x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0);  y2 = min(cy + cut_h // 2, H)

    mixed_views = views.clone()
    mixed_views[:, :, :, y1:y2, x1:x2] = views[index, :, :, y1:y2, x1:x2]

    lam_actual    = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    mixed_video_feats = lam_actual * video_feats + (1 - lam_actual) * video_feats[index]
    return mixed_views, labels, labels[index], mixed_video_feats, lam_actual


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def make_combined_df(train_csv: Path, dev_csv: Path) -> pd.DataFrame:
    train_df = pd.read_csv(train_csv)
    dev_df   = pd.read_csv(dev_csv)
    train_df['source'] = 'train'
    dev_df['source']   = 'dev'
    return pd.concat([train_df, dev_df], ignore_index=True)
