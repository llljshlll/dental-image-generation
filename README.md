# Real-Time Dental Image Generation with Stream Diffusion + ctrLoRA
**Goal:** 실시간 Dental 영상 생성에서 구조적 일관성과 디테일 유지 개선  
**Keywords:** Stable Diffusion, StreamDiffusion, ctrLoRA, ControlNet++, Weighted LoRA

> - **문제**: Stable Diffusion 기본 파이프라인은 **condition의 디테일 보존 한계**, **실시간성 미달**
> - **목표**: end-to-end **< 1s** + **치아/잇몸 디테일** + **조건 정합성**
> - **접근**: **Stream Diffusion**(속도) + **ctrLoRA**(조건 일관성/저자원)

---

## 🔗 Quick Links
- 시스템 개요: **[00_overview.md](0_0_overview.md)**
- real-time:
  - DMD vs Stream / 원리: **[1_0_stream_diffusion.md](1_0_stream_diffusion.md)**
  - TAESD ↔ VAE 비교: **[1_1_taesd.md](1_1_taesd.md)**
  - 스케줄러(LCM/UniPC): **[1_2_schedulers.md](1_2_schedulers.md)**
- control:
  - controlLoRA/ctrLoRA/SDXS 비교: **[2_0_ctrl_families.md](2_0_ctrl_families.md)**
  - ctrLoRA 메커니즘: **[2_1_ctrlora_mechanics.md](2_1_ctrlora_mechanics.md)**
  - ctrLoRA 학습(singe condition): **[2_2_ctrlora_training.md](2_2_ctrlora_training.md)**
  - multi-condition interference : **[2_3_multi_condition.md](2_3_multi_condition.md)**
- 다음 단계: **[3_0_controlnetpp_plan.md](3_0_controlnetpp_plan.md)**

---

## 1. Problem & Requirements
- **Pain points**
  - 프레임 간 **일관성** 부족(ghosting/duplication 등)
  - **치아/잇몸 경계**, **에나멜 질감** 등 고주파 디테일 손실
  - **실시간** 추론 미달
- **Targets**
  - **Latency**: end-to-end **< 1s**
  - **Quality**: 치아 경계/질감 보존, 잇몸 bleed-in 최소화
  - **Controllability**: lighting/segmentation 등 **조건과의 정합성**
 
---

## 2. Candidates & Decisions (요약)

### 2.1 실시간 후보
| 후보 | 장점 | 단점 | 관찰 | 결정 |
|---|---|---|---|---|
| **DMD** | 이론적 실시간성 | 재현 비용/세팅 복잡 | 제한적 벤치만 진행 |  |
| **Stream Diffusion** | **빠름**, 프레임 파이프라인 최적화 | TAESD 품질 도메인 의존 | **더 빠름 → 채택** | ✅ |
→ 자세히: [1_0_stream_diffusion.md](1_0_stream_diffusion.md)

### 2.2 디테일/일관성 후보
| 후보 | 장점 | 단점 | 결정 |
|---|---|---|---|
| controlLoRA | LoRA 기반 제어 용이 | 도메인에서 컨디션 정합성 제한 |  |
| **ctrLoRA** | **조건 일관성 양호**, **저자원 학습**, 멀티컨디션 지원 | 멀티컨디션 간섭 발생 | **채택** ✅ |
| SDXS | 경량/속도 장점 | 컨디션 정합성/품질 한계 |  |
→ 비교/선정 근거: [2_0_ctrl_families.md](2_0_ctrl_families.md), 메커니즘: [2_1_ctrlora_mechanics.md](2_1_ctrlora_mechanics.md)


---

## 3. System Overview


Input (lighting map, segmentation map, ...)
└─ Preprocess (resize/normalize, optional blending)
└─ Encoder / VAE
└─ UNet (+ ctrLoRA adapters)
└─ Scheduler: LCM + UniPCMultistep
└─ Decoder / VAE → Output frame (< 1s)

- **속도**: Stream Diffusion + **LCM**(스텝 단축) + **UniPC**(안정화)
- **정합성**: **ctrLoRA**로 조건 일관성 확보
- **도메인 특화**: **segmentation-weighted loss**로 치아 디테일 강화

더 자세히: 개요 [00_overview.md](0_0_overview.md), 실시간 최적화 [1_0/1_1/1_2], ctrLoRA [2_1/2_2]

---

## 4. Timeline & Decision Log (원인→완화→대안)

- **T0**: DMD vs Stream 비교 → **Stream 채택** ([1_0])  
- **T1**: **TAESD 적용 시도**(샘플 OK) → **치아 도메인 품질 저하** + 학습 코드 미공개 → **VAE 유지** ([1_1])  
- **T2**: **LCM**로 스텝 대감 → **과매끄러움** → **UniPC**로 고주파 복원 ([1_2])  
- **T3**: 제어 후보 비교 → **ctrLoRA 선정** ([2_0], [2_1])  
- **T4**: **ctrLoRA 학습(single condition)** 결과 정리 ([2_2])  
- **T5**: **멀티컨디션 간섭 규명** -> **Blending으로 multi condition을 single 이미지로 넣음**, **Segmentation-Weighted Loss**로 간섭 완화, **치아 경계/질감 개선** ([2_3])
- **T6 (예정)**: **ControlNet++** 구조적 대안 설계/평가 계획 ([3_0])


---

## 5. Key Experiments (Summary)

### 5.1 TAESD ↔ VAE (치아 도메인)
- 샘플에서는 TAESD OK이나 **치아 데이터**에서는 복원 품질 저하 → **VAE 유지**  
→ 비교/결정: [1_1_taesd.md](1_1_taesd.md)

### 5.2 Speed (실시간성)
| 세팅 | 스텝 수 | 스케줄러 | VAE/TAESD | Latency (s) | 비고 |
|---|---:|---|---|---:|---|
| Baseline SD(DDIM) | 20–50 | DDIM | VAE | … | 지연 큼 |
| **LCM** | 4–8 | LCM | VAE | **< 1s** | **속도↑, 질감↓** |
| **UniPC** | 4–8 | **UniPC** | VAE | **< 1s** | **속도 유지, 질감 회복** ✅ |
→ 세부: [1_2_schedulers.md](1_2_schedulers.md)

### 5.3 Multi-Condition Interference → Seg-Weighted 완화
- **증상**: lighting ↔ segmentation 충돌(ghosting/duplication/콘셉트 누출)  
- **가설/완화**: 치아 영역 **gradient 비중↑** → 경계/에나멜 텍스처 향상  
- **결과 요약**: \(w_{\text{tooth}}=\) **5.0**에서 품질–안정성 균형 최적  
- **자세히**: 원인 분석 [2_3_multi_condition.md](2_3_multi_condition.md), 완화 실험 [2_4_seg_weighting.md](2_4_seg_weighting.md)


### 6. Limitations & Next

- 한계: 극단 조명/각도에서 드물게 경계 깨짐, 멀티컨디션 완전한 disentangle 미흡
- 다음: ControlNet++(layout/content 분리) 적용














