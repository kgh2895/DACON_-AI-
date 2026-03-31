"""
train/simulation.mp4 에서 물리 운동 특징 추출 → data/video_features.json 저장
학습 전 딱 1회만 실행: python preprocess_video.py

추출 특징 (8개):
  motion_score   : Farneback Optical Flow 평균 이동량 (불안정할수록 큼)
  frame_diff     : 첫 프레임 vs 마지막 프레임 평균 절대 차이
  max_flow       : 프레임당 최대 픽셀 이동량 최댓값
  dominant_freq  : 운동량 시계열의 지배 주파수 (FFT — 진동 주기)
  motion_accel   : 운동량 변화율 평균 (가속도/jerk — 급격한 변화 = 불안정)
  collapse_frame : 최대 운동 발생 정규화 시점 (0=초반 붕괴, 1=말미 붕괴)
  top_motion     : 이미지 상단 절반 평균 flow (상부 구조 움직임)
  bottom_motion  : 이미지 하단 절반 평균 flow (하부/기반 움직임)

저장 후 min-max 정규화해서 [0, 1] 범위로 변환.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

TRAIN_DIR  = Path(__file__).parent / 'data' / 'train'
OUT_PATH   = Path(__file__).parent / 'data' / 'video_features.json'
MAX_FRAMES = 60   # 최대 60프레임(=10초 @ 6fps)
N_WORKERS  = 4

RAW_KEYS = [
    'motion_score', 'frame_diff', 'max_flow',
    'dominant_freq', 'motion_accel', 'collapse_frame',
    'top_motion', 'bottom_motion',
]


# ── 단일 영상 처리 ─────────────────────────────────────────────────────────────

def extract_features(sample_dir: Path) -> dict:
    video_path = sample_dir / 'simulation.mp4'
    zero = {k: 0.0 for k in RAW_KEYS}
    if not video_path.exists():
        return zero

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    if len(frames) < 2:
        return zero

    H = frames[0].shape[0]
    half = H // 2

    # ── Optical Flow ──────────────────────────────────────────────────────────
    motion_scores, max_flows = [], []
    top_motions, bottom_motions = [], []

    for i in range(1, len(frames)):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i-1], frames[i], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        motion_scores.append(float(mag.mean()))
        max_flows.append(float(mag.max()))
        top_motions.append(float(mag[:half].mean()))
        bottom_motions.append(float(mag[half:].mean()))

    # ── 첫 프레임 vs 마지막 프레임 차이 ─────────────────────────────────────────
    frame_diff = float(np.abs(frames[-1].astype(float) - frames[0].astype(float)).mean())

    # ── 지배 주파수 (FFT) ────────────────────────────────────────────────────
    if len(motion_scores) >= 4:
        fft  = np.abs(np.fft.rfft(motion_scores))
        freq = np.fft.rfftfreq(len(motion_scores))
        dominant_freq = float(freq[np.argmax(fft[1:]) + 1]) if len(fft) > 1 else 0.0
    else:
        dominant_freq = 0.0

    # ── 가속도 (jerk: 운동량 변화율) ─────────────────────────────────────────
    motion_accel = float(np.mean(np.abs(np.diff(motion_scores)))) if len(motion_scores) >= 2 else 0.0

    # ── 붕괴 시점 (최대 운동 발생 정규화 시점) ──────────────────────────────────
    collapse_frame = float(np.argmax(motion_scores) / len(motion_scores)) if motion_scores else 0.0

    return {
        'motion_score'  : float(np.mean(motion_scores)),
        'frame_diff'    : frame_diff,
        'max_flow'      : float(np.max(max_flows)),
        'dominant_freq' : dominant_freq,
        'motion_accel'  : motion_accel,
        'collapse_frame': collapse_frame,
        'top_motion'    : float(np.mean(top_motions)),
        'bottom_motion' : float(np.mean(bottom_motions)),
    }


def process_one(sample_dir: Path) -> tuple[str, dict]:
    return sample_dir.name, extract_features(sample_dir)


# ── 전체 처리 ─────────────────────────────────────────────────────────────────

def main():
    sample_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    print(f'처리할 샘플: {len(sample_dirs)}개')

    raw = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_one, d): d for d in sample_dirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc='영상 처리'):
            name, feats = future.result()
            raw[name] = feats

    # ── Min-Max 정규화 ────────────────────────────────────────────────────────
    for key in RAW_KEYS:
        vals  = np.array([v[key] for v in raw.values()])
        vmin, vmax = vals.min(), vals.max()
        denom = vmax - vmin if vmax > vmin else 1.0
        for name in raw:
            raw[name][f'{key}_norm'] = float((raw[name][key] - vmin) / denom)

    # ── 요약 출력 ─────────────────────────────────────────────────────────────
    for key in RAW_KEYS:
        norms = [v[f'{key}_norm'] for v in raw.values()]
        print(f'  [{key}_norm] min={min(norms):.3f}  max={max(norms):.3f}  mean={np.mean(norms):.3f}')

    with open(OUT_PATH, 'w') as f:
        json.dump(raw, f, indent=2)
    print(f'\n저장 완료 → {OUT_PATH}')


if __name__ == '__main__':
    main()
