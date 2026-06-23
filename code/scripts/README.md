# Scripts Directory

Threshold Sweep 및 메트릭 계산을 위한 논문 재현 스크립트 모음입니다.

## 수행 순서 (Workflow)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. run_threshold_sweep.py                                               │
│     ├─ Inpainting → Classification → Enhancement                       │
│     ├─ 출력: {threshold}/*_image_b.raw, *_image_m.raw                   │
│     └─ 시간: Head ~2h, Body ~15h (GPU)                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  2. recalculate_metrics.py  (선택적, 메트릭 재계산 필요 시)              │
│     ├─ GT 직접 로딩 → SSIM/RMSE 계산                                     │
│     ├─ 출력: metrics_0.XX.csv                                           │
│     └─ 시간: Head 3분, Body 30분 (병렬 P=20)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  3. merge_threshold_metrics.py                                           │
│     ├─ 개별 CSV → 통합                                                   │
│     ├─ 출력: metrics_by_threshold.csv, threshold_stats.csv              │
│     └─ 시간: 즉시                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1. Paper Scripts (논문 재현용)

| 스크립트 | 설명 | 용도 |
|----------|------|------|
| `run_threshold_sweep.py` | Threshold Sweep 실행 | 전체 파이프라인 (MAR + Enhancement) 실행 |
| `recalculate_metrics.py` | 메트릭 재계산 | 기존 결과에서 SSIM/RMSE 재계산 |
| `merge_threshold_metrics.py` | CSV 병합 | 개별 threshold CSV를 통합 |
| `apply_enhancement.py` | Enhancement 후처리 | 기존 결과에 Enhancement 모델 적용 |

### 주요 스크립트 상세

#### `run_threshold_sweep.py`
전체 MAR 파이프라인(Inpainting + Classification + Enhancement)을 실행하고 메트릭을 계산합니다.
```bash
python scripts/run_threshold_sweep.py \
    --anatomy head --min_threshold 0.01 --max_threshold 0.40 --step 0.01 \
    --output_dir results_threshold_sweep
```

#### `recalculate_metrics.py`
GT 이미지를 직접 로딩하여 메트릭을 고속으로 재계산합니다. (BP 재구성 과정 생략)
```bash
# 병렬 처리 예시
python scripts/recalculate_metrics.py --anatomy head --threshold 0.10

# CSV 병합 모드
python scripts/recalculate_metrics.py --merge
```

#### `merge_threshold_metrics.py`
재계산된 개별 CSV 파일들을 통합합니다. (의존성 없음, 독립 실행 가능)
```bash
python scripts/merge_threshold_metrics.py
```

## 2. Helper Scripts (Shell)

백그라운드에서 Threshold Sweep을 실행하기 위한 쉘 스크립트입니다.

| 스크립트 | 설명 |
|----------|------|
| `run_head_sweep.sh` | Head Anatomy Sweep 실행 (Log: `logs/head_sweep.log`) |
| `run_body_sweep.sh` | Body Anatomy Sweep 실행 (Log: `logs/body_sweep.log`) |

**사용법:**
```bash
./scripts/run_head_sweep.sh
./scripts/run_body_sweep.sh
```
