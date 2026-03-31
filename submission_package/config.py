import os
from pathlib import Path

class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    ROOT        = Path(__file__).parent
    DATA_DIR    = ROOT / 'data'
    TRAIN_DIR   = DATA_DIR / 'train'
    DEV_DIR     = DATA_DIR / 'dev'
    TEST_DIR    = DATA_DIR / 'test'
    TRAIN_CSV   = DATA_DIR / 'train.csv'
    DEV_CSV     = DATA_DIR / 'dev.csv'
    SAMPLE_SUB  = DATA_DIR / 'sample_submission.csv'
    CKPT_BASE_DIR = ROOT / 'checkpoints'
    CKPT_DIR      = ROOT / 'checkpoints'   # train.py에서 run 폴더로 덮어씌워짐
    SUBMIT_DIR    = ROOT / 'submissions'
    VIDEO_CACHE = DATA_DIR / 'video_features.json'

    # ── Image layout ───────────────────────────────────────────────────────
    IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    NUM_VIEWS      = 2

    # ── Model ──────────────────────────────────────────────────────────────
    BACKBONE    = 'convnext_large.fb_in22k_ft_in1k'
    IMAGE_SIZE  = 512                # 384 → 512 (BATCH_SIZE=4, GRAD_ACCUM=4로 VRAM 확보)
    NUM_CLASSES = 2
    DROP_RATE   = 0.5                # 0.3 → 0.5 (소규모 데이터 정규화 강화)
    DROP_PATH   = 0.3
    AUX_WEIGHT  = 0.3               # 보조 손실(영상 운동량 회귀) 가중치

    # ── Video Features ─────────────────────────────────────────────────────────
    VIDEO_FEAT_DIM  = 8             # head에 직접 concat하는 영상 피처 차원
    VIDEO_FEAT_DROP = 0.5           # 학습 시 영상 피처 per-sample masking 확률 (test 분포 맞춤)

    # ── Pseudo Labeling ────────────────────────────────────────────────────────
    PSEUDO_CONF = 0.80

    # ── Soft Label (Video Teacher KD) ─────────────────────────────────────────
    SOFT_LABEL_PATH   = DATA_DIR / 'soft_labels.json'
    SOFT_LABEL_WEIGHT = 0.7

    # ── Phase 2 Validation ────────────────────────────────────────────────────
    PHASE2_VAL_RATIO = 0.1          # combined data 중 validation으로 사용할 비율
    PHASE2_PATIENCE  = 20

    # ── Heterogeneous Ensemble ─────────────────────────────────────────────────
    EXTRA_CKPT_DIRS = ['checkpoints/20260325_2212']  # v14 (v12+v14 앙상블)

    # ── Training ───────────────────────────────────────────────────────────
    SEED        = 47                   # v12: SEED=47, v14: SEED=49
    N_FOLDS     = 5
    EPOCHS      = 50
    BATCH_SIZE  = 2
    GRAD_ACCUM  = 8                  # 유효 BS=16
    NUM_WORKERS = 2

    # Optimizer
    LR           = 2e-4
    MIN_LR       = 1e-6
    WEIGHT_DECAY = 1e-4

    # Loss
    LABEL_SMOOTHING = 0.0   # CE Loss 사용 (--no-focal)

    # SAM (Sharpness-Aware Minimization)
    USE_SAM  = False                 # SAM 폐기 (AMP 호환 불가)
    SAM_RHO  = 0.05

    # EMA (Exponential Moving Average)
    USE_EMA   = False
    EMA_DECAY = 0.999

    # ── Mixup / CutMix ─────────────────────────────────────────────────────
    MIXUP_ALPHA = 0.4
    MIXUP_PROB  = 0.5
    CUTMIX_ALPHA = 0.4
    CUTMIX_PROB  = 0.3

    # ── Domain Adaptation ──────────────────────────────────────────────────
    PHASE1_EPOCHS      = 100
    PHASE1_PATIENCE    = 35
    PHASE1_COSINE_TMAX = 100
    PHASE2_EPOCHS      = 60
    PHASE2_COSINE_TMAX = 60       # Phase2 cosine 스케줄 길이
    PHASE2_LR       = 5e-5

    # ── Fold 자동 재시도 ─────────────────────────────────────────────────
    FOLD_QUALITY_THRESHOLD  = 0.02   # 이하면 합격
    FOLD_COLLAPSE_THRESHOLD = 0.5    # 이상이면 collapse 판정
    FOLD_COLLAPSE_CHECK_EPOCH = 20   # 이 epoch까지 best_val > 0.5면 즉시 중단
    FOLD_MAX_RETRIES = 2             # fold당 최대 재시도 횟수

    # ── Inference ──────────────────────────────────────────────────────────
    TTA_TIMES    = 8
    USE_ALL_FOLD = True

    # ── Misc ───────────────────────────────────────────────────────────────
    AMP    = True
    DEVICE = 'cuda'

CFG = Config()

CFG.CKPT_BASE_DIR.mkdir(parents=True, exist_ok=True)
CFG.SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
