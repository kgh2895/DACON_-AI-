# 구조물 안정성 물리 추론 AI 경진대회 - 솔루션

## 1. 문제 정의 및 접근 전략

### 1.1 문제 요약
- **과제**: 2개 시점(front, top) 이미지로부터 구조물의 붕괴 확률(unstable) vs 안정 확률(stable) 예측
- **핵심 난이도**: 학습 데이터(1,000개)는 고정 환경, 평가 데이터(1,000개)는 무작위 광원/카메라 → **도메인 갭** 극복 필요
- **평가**: LogLoss (확률 calibration이 핵심)

### 1.2 접근 전략
1. **강력한 백본 + 소규모 데이터 최적화**: ConvNeXt-Large (197M params)를 BS=2로 안정 학습
2. **Cross-View Attention**: 두 시점 간 구조적 관계를 명시적으로 모델링
3. **Video Knowledge Distillation**: 시뮬레이션 영상에서 추출한 물리 특징을 이미지 모델에 전이
4. **2-Phase Training**: Supervised → Pseudo Label Fine-tuning으로 도메인 적응
5. **Temperature Scaling**: 제출 피드백 기반 확률 calibration
6. **Multi-Seed Ensemble**: 동일 구조, 다른 시드로 학습한 모델의 기하평균 앙상블

---

## 2. 모델 아키텍처

### 2.1 전체 구조

```
Input: (B, 2, 3, 512, 512)  <- front + top 이미지
            |
    +-------+-------+
    v               v
ConvNeXt-Large  ConvNeXt-Large   (가중치 공유)
    |               |
  1x1 Conv        1x1 Conv       (1536ch -> 512ch 투영)
    |               |
    v               v
  Cross-View Attention (8-head)  <- 두 시점 간 상호 참조
    |               |
  GeM Pool        GeM Pool
    |               |
    +-------+-------+
            |
      [1024] + [8]  <- 영상 물리 피처 concat
            |
      FC -> 512 -> 256 -> 2  (분류 헤드)
            |
        [unstable, stable]
```

### 2.2 핵심 컴포넌트

| 컴포넌트 | 설명 |
|----------|------|
| **Backbone** | `convnext_large.fb_in22k_ft_in1k` (ImageNet-22K pretrained, 197M params) |
| **Cross-View Attention** | Bidirectional Multi-Head Attention (8 heads, dim=512) + FFN. 두 시점의 feature map을 sequence로 펼쳐 상호 참조 |
| **GeM Pooling** | Generalized Mean Pooling (학습 가능 p=3.0). Global Average Pooling 대비 판별력 높은 영역에 가중치 부여 |
| **Video Feature Head** | 시뮬레이션 영상에서 추출한 8차원 물리 피처를 분류 헤드에 직접 concat |
| **Auxiliary Head** | 이미지만으로 motion_score를 회귀 예측하는 보조 손실 (Knowledge Distillation) |

### 2.3 Video Feature (Knowledge Distillation)

학습 데이터에만 존재하는 시뮬레이션 영상(simulation.mp4)에서 Optical Flow 기반 물리 특징 8개를 사전 추출:

| 피처 | 설명 |
|------|------|
| `motion_score` | Farneback Optical Flow 평균 이동량 |
| `frame_diff` | 첫 vs 마지막 프레임 차이 |
| `max_flow` | 최대 픽셀 이동량 |
| `dominant_freq` | FFT 지배 주파수 (진동 주기) |
| `motion_accel` | 운동량 변화율 (가속도) |
| `collapse_frame` | 최대 운동 발생 시점 (정규화) |
| `top_motion` | 상부 구조 움직임 |
| `bottom_motion` | 하부 기반 움직임 |

- 모든 피처는 min-max 정규화 [0, 1]
- 학습 시 50% 확률로 per-sample masking → test 시 피처 없는 상황에 적응
- 보조 헤드가 이미지만으로 motion_score 예측 학습 → 경계 케이스 판별력 향상

---

## 3. 학습 파이프라인

### 3.1 2-Phase Training

#### Phase 1: Supervised 5-Fold CV
- **데이터**: train(1,000) + dev(100) 합산 → StratifiedKFold(5)
- **목적**: 기본 분류 능력 학습
- **설정**: 100 epochs, patience=35, Cosine LR (T_max=100)
- **손실**: CrossEntropyLoss + Soft Label KD (weight=0.7) + Auxiliary MSE (weight=0.3)

#### Phase 2: Pseudo Label Fine-tuning
- **데이터**: Phase1 OOF 예측으로 test 데이터에 pseudo label 부여 (confidence >= 0.80)
- **목적**: test 도메인(무작위 환경) 적응
- **설정**: 60 epochs, patience=20, Cosine LR (T_max=60), LR=5e-5
- **검증**: combined data의 10%를 validation으로 분리

### 3.2 Data Augmentation

도메인 갭(고정 환경 → 무작위 환경) 극복을 위한 강한 augmentation:

| 카테고리 | 기법 |
|----------|------|
| **공간 변환** | Affine(+-10%, +-15deg), Perspective, HorizontalFlip, GridDistortion, ElasticTransform |
| **광원/색상** | RandomBrightnessContrast, RandomGamma, CLAHE, ColorJitter, HueSaturation, RandomShadow |
| **노이즈/블러** | GaussNoise, ISONoise, MotionBlur, GaussianBlur, Sharpen |
| **차단** | CoarseDropout (1~8 holes, 8~48px) |
| **혼합** | Mixup (alpha=0.4, p=0.5), CutMix (alpha=0.4, p=0.3) — 물리적 일관성을 위해 모든 뷰에 동일 패치 적용 |

### 3.3 학습 세부 설정

| 항목 | 값 |
|------|-----|
| Optimizer | AdamW (LR=2e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Batch Size | 2 (Gradient Accumulation 8 → 유효 BS=16) |
| Precision | bfloat16 (AMP) |
| Image Size | 512x512 |
| Fold Collapse 대응 | ep20까지 val>=0.5 시 자동 재시도 (최대 2회) |
| 재현성 | `torch.backends.cudnn.deterministic=True`, 고정 seed |

---

## 4. 추론 및 후처리

### 4.1 Test-Time Augmentation (TTA)
- **8회 추론**: 1회 clean + 7회 랜덤 augmentation (HorizontalFlip, Affine, BrightnessContrast, Perspective, GridDistortion)
- **앙상블**: Log-space geometric mean → renormalize

### 4.2 Temperature Scaling
- 모델 출력 logit을 temperature T로 나누어 확률 calibration
- **핵심 발견**: Dev/OOF 기반 T 최적화는 무의미 (모델이 train/dev에서 100% 정확 → T 구별 불가)
- T는 **제출 피드백(Public Score)으로만** 결정
- `log_odds / T → sigmoid → 제출`

### 4.3 Multi-Seed Ensemble (최종 제출)

동일 아키텍처 + 동일 하이퍼파라미터, SEED만 변경하여 2개 모델을 학습한 뒤 기하평균 앙상블:

| 버전 | SEED | Phase1 ep/patience | Phase2 ep/patience | Phase1 OOF | 단독 최적 T | 단독 Public Score |
|------|------|-------------------|-------------------|-----------|-----------|-----------------|
| **v12** | 47 | 100/35 | 60/20 | 0.003546 | 0.450 | 0.0036417 |
| **v14** | 49 | 100/35 | 60/20 | 0.003100 | - | 0.0132931 (T=0.5) |

- v14는 단독 성능은 부족하지만 (SEED 민감도), 앙상블에서 다양성을 제공하여 성능 향상
- **최종 앙상블**: v12(5fold) + v14(5fold) = 10개 모델의 geometric mean → T-scaling

### 4.4 Fold Collapse 및 재학습

소규모 데이터(~1,000장)에서 특정 fold가 학습 초기 val_loss >= 0.5에 고착되는 **fold collapse** 현상이 반복 발생:

- **자동 감지/재시도**: epoch 20까지 best_val >= 0.5이면 해당 fold를 자동 중단 후 재시도 (최대 2회, train.py 내장)
- **v14 Fold 3**: collapse가 3회 발생하여 수동 재학습으로 해결 (최종 best val 0.0043)
- **원인**: SEED에 따른 초기 가중치 + 데이터 분할 조합이 불안정한 gradient landscape를 형성
- **재현 시 주의**: SEED 변경 시 동일 fold에서 collapse가 발생하지 않을 수도 있고, 다른 fold에서 발생할 수도 있음. 자동 재시도 메커니즘이 이를 처리함

---

## 5. 핵심 발견 및 실험 이력

### 5.1 핵심 발견

1. **BS=2 필수**: BS=4에서 DevLL=0.2336 vs BS=2에서 DevLL=0.0687 → 1,000샘플에서 BS>=4는 수렴 실패
2. **CE Loss > Focal Loss**: FocalLoss + Mixup이 초기 gradient 불안정 유발 → fold collapse (val_loss ≈ 0.693 고착)
3. **Warmup 제거**: `--no-warmup`으로 CE Loss만 사용 시 안정적 수렴
4. **T-scaling 절벽**: T<=0.210에서 LogLoss 0.0078→0.025로 급등 → 하한선 존재
5. **Phase1 epoch/patience 최적점**: 100/35가 최적. 300/50 (v13)은 OOF 최고(0.0027)지만 test 7배 악화 → 과적합
6. **SEED 민감도**: 동일 설정 + SEED 변경만으로 test 성능 3.7배 차이 (v12 vs v14 단독)
7. **앙상블 효과**: 단독 성능 부족한 모델도 앙상블에서 다양성 기여 가능 (v12+v14 > v12 단독)

### 5.2 실험 이력 요약

| 버전 | SEED | 변경점 | Public Score | 교훈 |
|------|------|--------|-------------|------|
| v5 | 42 | ConvNeXt-Large + CE Loss + no-warmup | 0.0078247 | 기본 레시피 확립 |
| v7 | - | ConvNeXt-XLarge | 0.2038 | 오버피팅, 파라미터 과다 |
| v9 | - | BS=4 | 0.2336 | BS=2 필수 확인 |
| v11 | 46 | seed=46, 동일 레시피 | 0.0057420 | multi-seed 시작 |
| v12 | 47 | epoch/patience 증가 (100/35, 60/20) | 0.0036417 | 충분한 학습이 핵심 |
| v13 | 48 | epoch/patience 대폭 증가 (300/50, 100/30) | 0.0246 | 과적합 — epoch 상한선 확인 |
| v14 | 49 | v12 설정 동일, SEED만 변경 | 0.0133 | SEED 민감도, 앙상블용 |
| **v12+v14** | - | **2모델 기하평균 앙상블** | **0.0032386** | **최종 제출** |

---

## 6. 최종 결과

| 항목 | 값 |
|------|-----|
| **최종 Private Score** | **0.01063** |
| **Private 순위** | **3위** |
| **최종 Public Score** | **0.0032385603** |
| **Public 순위** | **2위** |
| **사용 모델** | v12 (SEED=47) + v14 (SEED=49) 앙상블 |
| **최적 Temperature** | T=0.550 |
| **앙상블 방법** | 10-fold geometric mean (v12 5fold + v14 5fold) |

### T-scaling 탐색 이력

```
v12+v14 앙상블:
  T=0.400 → 0.0036226
  T=0.500 → 0.0032691
  T=0.550 → 0.0032386 (최종 제출)
```

---

## 7. 개발 환경 및 재현 방법

### 7.1 개발 환경

| 항목 | 값 |
|------|-----|
| OS | Ubuntu (WSL2) / Linux 6.6.87 |
| GPU | NVIDIA RTX 5060 Ti 16GB |
| Python | 3.11 |
| PyTorch | 2.10.0 (bfloat16 AMP) |
| timm | 1.0.25 |
| albumentations | 2.0.8 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |
| CUDA | 12.x |

### 7.2 재현 순서

```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 데이터 배치 (상대 경로)
# data/train/, data/dev/, data/test/, data/train.csv, data/dev.csv, data/sample_submission.csv

# 3. 영상 피처 추출 (1회)
python preprocess_video.py

# 4. Soft Label 생성 (선택, train_video_teacher.py 실행 후 data/soft_labels.json 생성)

# === v12 학습 (SEED=47) ===
# 5a. config.py에서 SEED=47 확인 후 실행
HF_HUB_OFFLINE=1 python -u train.py --no-warmup --no-focal 2>&1 | tee logs/v12.log

# === v14 학습 (SEED=49) ===
# 5b. config.py에서 SEED=49로 변경 후 실행
HF_HUB_OFFLINE=1 python -u train.py --no-warmup --no-focal 2>&1 | tee logs/v14.log

# 6. 앙상블 추론 (config.py EXTRA_CKPT_DIRS에 v14 체크포인트 경로 설정)
python inference.py

# 7. Temperature Scaling 적용 (T=0.550)
python temperature_scale.py --T 0.550 --sub submissions/<raw_submission>.csv

# 8. 제출 파일 확인
# submissions/submission_T0.550_<timestamp>.csv
```

### 7.3 파일 구조

```
structural_stability/
├── config.py                 # 하이퍼파라미터 중앙 관리
├── model.py                  # ConvNeXt-Large + CrossView Attention + GeM
├── dataset.py                # Dataset, Augmentation, Mixup/CutMix
├── train.py                  # 2-Phase 학습 파이프라인
├── inference.py              # TTA 추론 + fold 앙상블
├── temperature_scale.py      # T-scaling 적용
├── preprocess_video.py       # 영상 물리 피처 추출
├── train_video_teacher.py    # Soft label 생성용 video teacher
├── utils.py                  # 유틸리티 함수
├── requirements.txt          # 라이브러리 버전
├── data/
│   ├── train/                # 학습 데이터 (1,000 샘플)
│   ├── dev/                  # 개발 데이터 (100 샘플)
│   ├── test/                 # 평가 데이터 (1,000 샘플)
│   ├── train.csv
│   ├── dev.csv
│   ├── sample_submission.csv
│   ├── video_features.json   # 사전 추출 영상 피처
│   └── soft_labels.json      # Video teacher soft labels
├── checkpoints/
│   ├── 20260319_2248/        # v12 (SEED=47) phase2 체크포인트
│   └── 20260325_2212/        # v14 (SEED=49) phase2 체크포인트
└── submissions/              # 제출 파일
```
