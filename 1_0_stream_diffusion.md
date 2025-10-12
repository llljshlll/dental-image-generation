# Stream Diffusion – A Pipeline-Level Solution for Real-Time Diffusion Generation  


## 🧩 1. Motivation: Why Stream Diffusion?

기존 Stable Diffusion은 **수십 회의 순차적 디노이징(Sequential Denoising)** 을 수행해야 하므로  
실시간 상호작용(AR, VTuber, 방송 등)에는 부적합하다.  

기존 가속화 연구는 **모델 레벨(Model-Level)** 접근이었다:  
- **DMD / DMD2**: teacher 모델을 few-step 학생으로 distillation:contentReference[oaicite:0]{index=0}  
- **Progressive / Consistency Distillation**: step 수를 줄이는 훈련 기법  
→ 품질 저하나 훈련 비용이 높음.

**Stream Diffusion**은 반대로 **파이프라인 레벨(Pipeline-Level)** 에서 접근하여  
기존 모델을 유지하면서도 **throughput(초당 프레임 수)** 을 극적으로 높인다:contentReference[oaicite:1]{index=1}.

---

## ⚙️ 2. Core Components of Stream Diffusion

Stream Diffusion은 세 가지 주요 메커니즘으로 구성된다:contentReference[oaicite:2]{index=2}.

---

### 🌀 2.1 Stream Batch — *“Sequential → Batched”*
> **핵심 개념:** 여러 타임스텝의 디노이징을 한 번의 U-Net 패스로 병렬 처리

- 기존: step 수만큼 U-Net을 반복 실행 → 시간 ∝ step 수  
- Stream Batch: 여러 step을 **대각선(batch)** 으로 묶어 한 번의 U-Net 실행으로 각각 1 step씩 전진:contentReference[oaicite:3]{index=3}  
- 결과적으로 “시간” 대신 “VRAM”이 병목이 된다 → **시간 ↔ VRAM ↔ 품질** 트레이드오프 구조:contentReference[oaicite:4]{index=4}

**효과:**  
- Throughput 최대 **1.5× 향상**:contentReference[oaicite:5]{index=5}  
- Future Frame을 참조하여 **temporal consistency** 향상:contentReference[oaicite:6]{index=6}

---

### ⚡ 2.2 Residual Classifier-Free Guidance (R-CFG)
> **핵심 개념:** 음(negative) 조건의 중복 계산 제거로 속도 향상

- 기존 CFG: 각 step마다 `ε_c`(조건), `ε_uc`(비조건)을 모두 계산  
- R-CFG: 음조건 `ε_uc`를 **잔차(residual)** 로 근사 → 한 번(또는 0회)만 계산:contentReference[oaicite:7]{index=7}  
- Self-Negative(0회), One-time-Negative(1회) 방식 존재

**효과:**  
- Step 5 기준, **2.05× (Self-Neg)** / **1.79× (One-time-Neg)** 속도 개선:contentReference[oaicite:8]{index=8}

---

### 🌿 2.3 Stochastic Similarity Filter (SSF)
> **핵심 개념:** 연속 프레임 간 유사도를 기준으로 확률적 스킵

- 정적(static) 구간에서는 디퓨전 호출을 건너뛰어 전력 절감  
- 유사도 \(SC(I_t, I_{ref})\) 가 임계값 η 이상이면 skip 확률 \(P = \max\{0, \frac{SC - \eta}{1 - \eta}\}\) 적용:contentReference[oaicite:9]{index=9}

**효과:**  
- RTX 3060: 전력 소비 **85.9 W → 35.9 W (2.39× 절감)**  
- RTX 4090: **238.7 W → 119.8 W (1.99× 절감)**:contentReference[oaicite:10]{index=10}

---

## 📊 3. Quantitative Highlights
| Metric | Stream Batch | R-CFG | SSF | Combined |
|:--------|:--------------|:------|:----|:-----------|
| Throughput | **1.5×** | — | — | — |
| Speedup | — | **2.05×** | — | — |
| Power Saving | — | — | **~2×** | — |
| Overall | — | — | — | **91.07 FPS @ RTX 4090 (AutoPipeline 대비 59.6×)**:contentReference[oaicite:11]{index=11} |

---

## 🧠 4. Comparison: Stream Diffusion vs DMD2

| 구분 | Stream Diffusion | DMD2 |
|------|------------------|------|
| **접근 방식** | Pipeline-level 최적화 | Model-level distillation |
| **핵심 기법** | Stream Batch / R-CFG / SSF | Regression-free DMD + GAN + Backward Simulation:contentReference[oaicite:12]{index=12} |
| **Step 수** | N-step 유지 (기존 모델 사용) | Few-step (1 ~ 4 step) |
| **목표** | FPS/Latency 개선 | Sampling step 단축 |
| **장점** | 기존 파이프라인과 호환 / 모듈 교체 용이 | 모델 자체 경량화 가능 |
| **한계** | VRAM 사용량 증가 / 정적 장면에 유리 | 학습 비용 높고 증류 불안정 가능 |

요약하자면,  
- **DMD2**는 *“모델을 더 빠르게”* 만드는 접근,  
- **Stream Diffusion**은 *“파이프라인을 더 효율적으로”* 만드는 접근이다.  
두 방법은 **직교적**이며 병행 적용 가능하다:contentReference[oaicite:13]{index=13}:contentReference[oaicite:14]{index=14}.

---

## ⚙️ 5. Implementation Notes
- Stream Batch → Batch size와 offset을 조정해 VRAM–FPS 균형 맞춤  
- R-CFG → Negative pass 1회만 수행, residual 재사용  
- SSF → 유사도 계산: CLIP feature / cosine similarity  
- I/O Queue + Pre-compute → Encoder ↔ UNet ↔ Decoder 병렬화  
- TensorRT / CUDA Graph → 추론 모듈 단위 최적화:contentReference[oaicite:15]{index=15}

---

## ⚖️ 6. Limitations & Trade-offs
- **VRAM Dependency**: Stream Batch 폭이 커질수록 GPU 메모리 급증:contentReference[oaicite:16]{index=16}  
- **Scene Dynamics**: SSF는 정적 장면엔 효과적이나, 급격한 장면 변화에서는 이득 제한:contentReference[oaicite:17]{index=17}

---

## 🏁 7. Takeaways
✅ Stream Diffusion은 **모델 구조 변경 없이**  
 - Stream Batch로 **FPS ↑**  
 - R-CFG로 **지연 ↓**  
 - SSF로 **전력 ↓**  
 → **RTX 4090에서 91 FPS**, Diffusers AutoPipeline 대비 **59.6× throughput 향상**:contentReference[oaicite:18]{index=18}  

> **Next:**  
> - [1_1_taesd.md] – 인코더(TAESD vs VAE) 속도/품질 비교  
> - [1_2_schedulers.md] – LCM 및 UniPC 스케줄러 최적화
