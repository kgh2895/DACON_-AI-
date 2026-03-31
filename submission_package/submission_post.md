## Video Knowledge Distillation
학습에만 있는 시뮬레이션 영상에서 Optical Flow 기반 물리 특징 8개를 추출, 분류 헤드에 concat하되 50% masking으로 영상 없는 평가 환경에 적응. 보조 헤드로 이미지에서 motion_score 회귀 학습하여 물리 단서를 내재화했습니다.

## Cross-View Attention
front/top 2시점을 ConvNeXt-Large로 인코딩 후 Bidirectional Attention(8-head)으로 시점 간 구조적 관계를 모델링, GeM Pooling 적용.

## 2-Phase Training
1,000샘플에서 BS>=4 수렴 실패, FocalLoss+Mixup은 fold collapse 유발 → BS=2 + CE Loss 사용. Phase1(5-Fold CV) → Phase2(pseudo label FT)로 고정/무작위 환경 간 도메인 갭 대응.

## 후처리
train/dev 100% 정확하여 T는 제출 피드백으로만 결정. SEED 47/49 × 5fold = 10모델 기하평균 앙상블 + T=0.550.

## 재현
Score 복원: inference.py → temperature_scale.py --T 0.550
재학습: preprocess_video.py → train_video_teacher.py → train.py --no-warmup --no-focal (SEED 47/49)
폴더: ./data/(데이터), ./checkpoints/(가중치), ./submissions/(결과)
※ fold collapse 자동 재시도 내장(ep20 val>=0.5 시 최대 2회). GPU 비결정성으로 체크포인트 추론 권장.
