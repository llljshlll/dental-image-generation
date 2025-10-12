# System Overview – Stream Diffusion + ctrLoRA for Real-Time Dental Image Generation  

---

## 1. Architecture Overview
본 프로젝트는 Stable Diffusion 기반의 이미징 파이프라인을 **실시간(1초 이하)** 으로 구현하면서, **치아 디테일과 구조적 일관성**을 유지하는 것을 목표로 한다.  
이를 위해 **=Real-time** 과 **Controllability** 의 두 축을 중심으로 설계되었다.

- **Real-time 축**: Stream Diffusion → TAESD / VAE → LCM / UniPCMultistep  
- **Controllability 축**: ctrLoRA → Multi-condition Control → Segmentation-weighted Loss → ControlNet++


---
## 2. Architecture Overview
(전체 아키텍처 다이어그램 추후 삽입: Input → Encoder → UNet(+ctrLoRA) → Scheduler → Decoder)  


---

## 3. Data Flow
1. **Input**  
   - Condition maps: lighting, normal, curvature, segmentation 등
   - 치아/잇몸 영역을 세분화하여 condition별 시각 정보를 명확히 전달  
  
2. **Encoding (VAE)**  
   - 입력 이미지를 latent space로 변환  
   - 치아 도메인에서는 **TAESD 품질 저하**로 인해 VAE 유지 결정  
→ 세부 비교: [1_1_taesd.md](1_1_taesd.md)  
  
3. **Denoising (UNet + ctrLoRA)**  
  - **UNet**: 노이즈 예측 및 조건 정보 통합의 핵심 모듈  
  - **ctrLoRA**: ControlNet 대비 적은 자원으로 multi-condition control 수행  
    - base ControlNet weight는 고정(frozen)  
    - LoRA branch만 학습 (single/multi condition 모두 지원)  
  → 세부 구조: [2_1_ctrlora_mechanics.md](2_1_ctrlora_mechanics.md)
    
4. **Scheduler (LCM, UniPCMultistep)**  
  - 기존 20~50 step DDIM → 4~8 step로 축소  
  - LCM은 속도를 개선하지만 결과가 과매끄럽게 되는 경향  
  - UniPCMultistep으로 바꿔서 세부 질감 복원 성공  
  → 세부 설정: [1_2_schedulers.md](1_2_schedulers.md)
  
5. Loss & Weighting  
  - 기본 objective: predicted noise와 target noise의 MSE  
  - **Segmentation-weighted Loss**로 치아 영역 gradient 증폭  
  - weighting factor: 1.0 / 2.0 / **5.0(최적)** / 8.0  
  → 실험 결과: [2_4_seg_weighting.md](2_4_seg_weighting.md)

6. **Decoding (VAE)**  
   - latent → RGB image로 복원  

---

## 4. Module Description  

### (1) Stream Diffusion  
- 기존 Stable Diffusion의 프레임별 독립 inference 문제를 해결  
- 프레임 간 temporal consistency 확보  
- 핵심 구성: Encoder, UNet, Scheduler, Decoder의 연속 처리  

### (2) ctrLoRA (Controllable LoRA)  
- **ControlNet의 경량화 버전**으로 condition과의 정합성을 LoRA branch에서 학습  
- Base ControlNet은 frozen, LoRA만 fine-tuning  
- **멀티컨디션 학습 시** 간섭(Interference) 발생 → segmentation-weighted 방식으로 완화  

### (3) Segmentation-weighted Loss  
- 치아 영역 \( M_{tooth} \)에만 gradient를 증폭시켜 디테일 복원  
- 잇몸 영역 \( M_{gum} \)은 1.0, 치아 영역은 2.0~8.0 가중치 부여  
- \( w_{tooth}=5.0 \)일 때 품질과 안정성 균형이 최적  

### (4) Scheduler: LCM + UniPCMultistep  
- LCM: inference step 단축 (속도↑, 질감↓)  
- UniPCMultistep: LCM 결과의 과매끄러움 완화 (질감 복원)  

---

## 5. Design Rationale
- **VAE 유지 이유:** TAESD는 치아 영역 복원력 저하 + TAESD training code 미공개  
- **ctrLoRA 채택 이유:** ControlNet 대비 적은 자원으로 condition 정합성 우수  
- **Segmentation-weighted Loss 적용 이유:** 치아 디테일 향상, multi-condition interference 완화  
- **UniPC 선택 이유:** 실시간성과 품질의 균형 확보  

---

## 6. Integration Flow  
------  
(Flow diagram 예시)  
1️⃣ Input condition maps  
→ 2️⃣ Latent encoding (VAE)  
→ 3️⃣ Denoising (UNet + ctrLoRA)  
→ 4️⃣ Step scheduling (LCM + UniPCMultistep)  
→ 5️⃣ Weighted loss 계산  
→ 6️⃣ Decoding & Output  
------  

---

## 7. Scalability & Future Work  
- **ControlNet++ 통합 계획:**  
  - multi-condition interference 완전 제거 목표  
  - 기존 ctrLoRA 학습에 ControlNet++의 학습 방식 추가
- **Real-time 추가 고안:**  
  - UniPC scheduler에서 디노이징 배치 처리 방식 고안
---


> 차후 ControlNet++ 통합을 통한 더 정확한 디테일 향상이 목표
