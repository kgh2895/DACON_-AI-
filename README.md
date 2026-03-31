# DACON 구조물 안정성 물리 추론 AI 경진대회 - 3rd Place Solution

> **Public 2위 (0.00324) / Private 3위 (0.01063)**
>
> [대회 링크](https://dacon.io/competitions/official/236686)

2개 시점(front, top) 이미지로 구조물의 안정/붕괴를 예측하는 분류 문제입니다.
학습 데이터(1,000개)는 고정 환경, 평가 데이터(1,000개)는 무작위 광원/카메라로 촬영되어 **도메인 갭 극복**이 핵심 과제였습니다.

## Solution Overview

### Architecture

```
Input: front + top 이미지 (2 x 3 x 512 x 512)
    |                   |
ConvNeXt-Large     ConvNeXt-Large    (weight sharing)
    |                   |
  1x1 Conv           1x1 Conv        (1536 -> 512ch)
    |                   |
    Cross-View Attention (8-head, bidirectional)
    |                   |
  GeM Pool           GeM Pool
    |                   |
    +------- cat -------+
            |
      [1024] + [8]    <- video physics features (50% masked during training)
            |
      FC 512 -> 256 -> 2
            |
      [unstable, stable]
```

- **Backbone**: `convnext_large.fb_in22k_ft_in1k` (ImageNet-22K pretrained, 197M params)
- **Cross-View Attention**: 두 시점의 feature map을 bidirectional multi-head attention으로 상호 참조
- **GeM Pooling**: 판별력 높은 영역에 가중치를 부여하는 learnable pooling (p=3.0)

### Video Knowledge Distillation

학습 데이터에만 존재하는 시뮬레이션 영상(simulation.mp4)에서 Optical Flow 기반 물리 특징 8개를 추출하여 활용했습니다.

| Feature | Description |
|---------|-------------|
| `motion_score` | Farneback Optical Flow 평균 이동량 |
| `frame_diff` | 첫 vs 마지막 프레임 차이 |
| `max_flow` | 최대 픽셀 이동량 |
| `dominant_freq` | FFT 지배 주파수 (진동 주기) |
| `motion_accel` | 운동량 변화율 (가속도) |
| `collapse_frame` | 최대 운동 발생 시점 (정규화) |
| `top_motion` | 상부 구조 움직임 |
| `bottom_motion` | 하부 기반 움직임 |

- 학습 시 50% 확률로 masking하여 평가 시 영상 없이도 동작하도록 적응
- 보조 헤드가 이미지만으로 `motion_score`를 회귀 예측 (knowledge distillation)

### 2-Phase Training

| | Phase 1: Supervised | Phase 2: Pseudo Label FT |
|---|---|---|
| **Data** | train(1,000) + dev(100), StratifiedKFold(5) | Phase1 OOF 예측으로 test에 pseudo label 부여 (conf >= 0.80) |
| **Epochs** | 100 (patience 35) | 60 (patience 20) |
| **LR** | 2e-4, CosineAnnealing | 5e-5, CosineAnnealing |
| **Loss** | CE + Soft Label KD (0.7) + Auxiliary MSE (0.3) | CE + Soft Label KD (0.7) + Auxiliary MSE (0.3) |

- **BS=2 + Gradient Accumulation 8** (effective BS=16) -- BS>=4에서 수렴 실패
- **CE Loss only** (`--no-warmup --no-focal`) -- FocalLoss+Mixup이 fold collapse 유발
- **bfloat16 AMP**

### Ensemble & Post-processing

- SEED 47 (v12) + SEED 49 (v14), 각 5-fold = **10개 모델의 geometric mean**
- v14는 단독 성능 부족(0.0133)이지만 앙상블에서 다양성 기여
- Temperature Scaling: T=0.550 (제출 피드백 기반, dev/OOF에서는 결정 불가)
- TTA: 1회 clean + 7회 random augmentation, log-space geometric mean

## Key Findings

| Finding | Detail |
|---------|--------|
| BS=2 필수 | BS=4 DevLL=0.2336 vs BS=2 DevLL=0.0687 |
| CE > Focal | FocalLoss+Mixup이 fold collapse(val_loss 0.693 고착) 유발 |
| Epoch 상한선 | 100/35 최적, 300/50은 OOF 최고(0.0027)이나 test 7배 악화 |
| SEED 민감도 | 동일 설정 + SEED만 변경으로 test 성능 3.7배 차이 |
| T-scaling 절벽 | T<=0.210에서 LogLoss 급등 (0.0078 -> 0.025) |

## Experiment Log

| Version | SEED | Change | Public Score |
|---------|------|--------|-------------|
| v5 | 42 | ConvNeXt-Large + CE + no-warmup (base recipe) | 0.00782 |
| v7 | - | ConvNeXt-XLarge | 0.20380 |
| v9 | - | BS=4 | 0.23360 |
| v11 | 46 | Multi-seed 시작 | 0.00574 |
| v12 | 47 | epoch/patience 증가 (100/35, 60/20) | 0.00364 |
| v13 | 48 | epoch/patience 대폭 증가 (300/50, 100/30) | 0.02460 |
| v14 | 49 | v12 설정 동일, SEED만 변경 | 0.01330 |
| **v12+v14** | - | **Geometric mean ensemble, T=0.550** | **0.00324** |

## Reproduction

### Checkpoints

학습된 모델 가중치는 Google Drive에서 다운로드할 수 있습니다.

**[Download Checkpoints (Google Drive)](https://drive.google.com/drive/folders/1W0OSAbDGW9OLo1iPiIDkBX0fCjuxBJrv?usp=drive_link)**

다운로드 후 아래와 같이 배치:
```
checkpoints/
├── 20260319_2248/    # v12 (SEED=47)
│   ├── fold1_phase2.pth ~ fold5_phase2.pth
│   ├── oof_labels.npy
│   └── oof_preds.npy
└── 20260325_2212/    # v14 (SEED=49)
    └── fold1_phase2.pth ~ fold5_phase2.pth
```

### Score Reproduction (from checkpoints)

```bash
pip install -r requirements.txt

# 1. Ensemble inference
python inference.py

# 2. Temperature scaling
python temperature_scale.py --T 0.550 --sub submissions/<raw_submission>.csv
```

### Full Retraining

```bash
# 1. Video feature extraction
python preprocess_video.py

# 2. Soft label generation
python train_video_teacher.py

# 3. Train v12 (SEED=47) -- set SEED=47 in config.py
HF_HUB_OFFLINE=1 python -u train.py --no-warmup --no-focal

# 4. Train v14 (SEED=49) -- set SEED=49 in config.py
HF_HUB_OFFLINE=1 python -u train.py --no-warmup --no-focal

# 5. Inference + T-scaling (same as above)
```

> **Note**: Fold collapse 자동 재시도가 내장되어 있습니다 (ep20까지 val>=0.5 시 최대 2회 재시도).
> GPU 비결정성으로 인해 체크포인트 기반 추론을 권장합니다.

### Project Structure

```
├── config.py                  # Hyperparameter config
├── model.py                   # ConvNeXt-Large + CrossView Attention + GeM
├── dataset.py                 # Dataset, Augmentation, Mixup/CutMix
├── train.py                   # 2-Phase training pipeline
├── inference.py               # TTA inference + fold ensemble
├── temperature_scale.py       # Temperature scaling
├── preprocess_video.py        # Video physics feature extraction
├── train_video_teacher.py     # Video teacher for soft labels
├── utils.py                   # Utilities
├── requirements.txt
├── soft_labels.json           # Video teacher soft labels
├── video_features.json        # Pre-extracted video features
├── checkpoints/               # Trained model weights (Google Drive)
└── data/                      # Image data & CSV (not included)
    ├── train/, dev/, test/
    ├── train.csv, dev.csv
    └── sample_submission.csv
```

## Environment

| | |
|---|---|
| GPU | NVIDIA RTX 5060 Ti 16GB |
| OS | Ubuntu (WSL2) |
| Python | 3.11 |
| PyTorch | 2.10.0 |
| CUDA | 12.x |
