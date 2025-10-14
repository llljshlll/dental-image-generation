# Multi-Condition (Segmentation + Lighting) – Interference 분석, 완화  

---

## 1) Setup: 단일 → 다조건 전환

- **단일 학습(선행)**  
  - `LoRA_lighting` : lighting 전용 단일 조건 학습  
  - `LoRA_seg`      : segmentation 전용 단일 조건 학습  
  - (선택) `LoRA_normal`, `LoRA_curvature`  
- **다조건 목표**  
  - **Segmentation**: 치아/잇몸 **영역·경계**를 명확히 지정  
  - **Lighting**    : 치아 표면 **굴곡/명암**을 안정적으로 반영  
- **초기 조합 방식**: inference 시 **LoRA weighting**  
<img src="images/2_2_multi_condition/multi.png" alt="lora weighting" width=600>   
  - 기본값: <img src="images/2_2_multi_condition/seg=1,ligth=1.png" alt="defalut" width=300>  

### 결과 (w_seg=1.0, w_light=1.0, steps=20, DDIM, CFG_light=2.0, CFG_seg=2.0, 512x512)

| Segmentation (input) | Lighting (input) | Output (seg=1.0 + light=1.0) | Reference |
|:---:|:---:|:---:|:---:|
| <img src="images/2_1_ctrLoRA_training/test/seg_lower_patient2_bottom.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/lighting_lower_patient2_bottom.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/multi_lower_patient2_bottom.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/lower_patient2_bottomm.png" width="160"/> |
| <img src="images/2_1_ctrLoRA_training/test/1_condition_segementation.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/1_condition_lighting.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/1_interference_multi_lora_1_1.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/upper_baliwish_right.png" width="160"/> |




---

## 2) 문제: Multi-Condition Interference

역할 정의
- **Lighting map** → 디테일·질감 복원
- **Segmentation map** → 치아–잇몸 경계 보존·형상 유지

단순 가중합 적용(기본  <img src="images/2_2_multi_condition/seg=1,ligth=1.png" alt="defalut" width=250>)
<img src="images/2_2_multi_condition/multi.png" alt="lora weighting" width=600> 

문제 핵심
- 역할 혼선 → 각 조건의 약점 전파
- 디테일 신호와 경계 신호의 비직교 합성 → 간섭 증폭
- 임베딩 스케일 불균형 + 동등 가중 → 특정 조건 과지배 혹은 과소 반영

원인 분석
1) **Role misalignment**  
   - Seg LoRA: 경계 특화, 질감 취약  
   - Light LoRA: 질감 특화, 경계 취약  
   - 동일 공간 합성 시 취약점 상호 침투

2) **Non-orthogonal residuals**  
   - \(L_{\psi_{\text{seg}}}\), \(L_{\psi_{\text{light}}}\) 동특성 공간 합산  
   - 채널/주파수 대역 침범, 기여도 무작위화

3) **Scale imbalance**  
   - <img src="images/2_2_multi_condition/seg=1,ligth=1.png" alt="defalut" width=300> 고정  
   - 과지배/과소 반영 발생

관찰 증상
- **Lighting 디테일 소실** → 미세 텍스처 평탄화  
- **경계 이상** → 선명도 저하 혹은 과도 선명화(oversharpen) 발생
- **Concept bleed** → 잇몸 톤/형상 치아 영역 침투

결론
- 동등 가중 단순 합성 → 역할 분담 붕괴 → Multi-Condition Interference 발생  
- 후속 완화 조치: **LoRA weight 조정**, **입력 블렌딩(α)**, **치아 weight loss 조정**

---

## 3) 완화 실험

### 3.1 LoRA weight 조정
목표: 역할 분담 회복 (seg=경계, light=질감)

---

### 3.2 입력 블렌딩(α)
- 서로 다른 브랜치 경로 → 충돌  
- **단일 condition**으로 묶어 **역할 분리 유지 + interference 완화** 목표

- 학습 시 **segmentation 투명도 30%**로 **lighting에 합성**해서 input으로 사용

- 아이디어: **segmentation 투명도 30%**로 **lighting에 합성**, **단일 condition**으로 처리  
  - \(\tilde{c} = \alpha \cdot c_{\text{seg}} + (1-\alpha)\cdot c_{\text{light}},\; \alpha=0.3\)
- 적용: **seg+light 합성 지도**를 입력으로 하고, **단일 LoRA** training, inference  
- 관찰: 
  - **경계 품질 향상**: 치아–잇몸 경계 **안정**  
  - **디테일 복원 양호**: lighting 신호 반영 **유지**, 디테일 **소폭 개선**
 

| Input | LoRA에 1.0, 1.0 가중합 | **Blending해서 단일 컨디션으로 학습** | Reference |
|:---:|:---:|:---:|:---:|
| <img src="images/2_1_ctrLoRA_training/test/merge_lower_patient2_bottom.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/multi_lower_patient2_bottom.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/merge.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/lower_patient2_bottomm.png" width="160"/> |
| <img src="images/2_1_ctrLoRA_training/test/merge_upper_baliwish_right.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/1_interference_multi_lora_1_1.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/1_interference_merge_cfg_2.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/upper_baliwish_right.png" width="160"/> |




---

### 3.3 치아 weighting loss 조정

배경  
- **CASDM**의 **class-aware ε-MSE** 아이디어 차용, **작은/중요 클래스에 가중치 부여**로 복원 우선순위 재조정
  
- **치아 영역**에만 **손실 가중치↑**  
  \[
  \mathcal{L}=\Big(1+(w_{\text{tooth}}-1)\cdot M_{\text{tooth}}\Big)\cdot
  \|\epsilon_\theta(x_t,t,c)-\epsilon\|_2^2,\quad w_{\text{tooth}}\in\{3,5,7\}
  \]
- **결론**: \(w_{\text{tooth}}=5\)가 **가장 우수**.  
  - **w=2**: 개선은 있으나 경계 선명도/질감 복원이 **부족**  
  - **w=5**: **경계·질감·안정성**의 **균형 최적**  
  - **w=8**: 경계 과증폭으로 **halo/ringing** 경향, 색 번짐 증가

| Input | weight=1 | weight=2 | **weight=5** | weight=8 | Reference |
|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="images/2_1_ctrLoRA_training/test/merge_upper_M4HYU284_front.png" width="150"/> | <img src="images/2_1_ctrLoRA_training/test/upper_M4HYU284_front_weight_1.png" width="150"/> | <img src="images/2_1_ctrLoRA_training/test/upper_M4HYU284_front_weight_2.png" width="150"/> | <img src="images/2_1_ctrLoRA_training/test/upper_M4HYU284_front_weight_5.png" width="150"/> | <img src="images/2_1_ctrLoRA_training/test/upper_M4HYU284_front_weight_8.png" width="150"/> | <img src="images/2_1_ctrLoRA_training/test/upper_M4HYU284_front.png" width="150"/> |




| Input | Weight=2 | **Weight=5** | Reference |
|:---:|:---:|:---:|:---:|
| <img src="images/2_1_ctrLoRA_training/test/merge_upper_M357DNS7_left.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/weight2_M357DNS7_left.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/weight5_M357DNS7_left.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/upper_M357DNS7_left.png" width="160"/> |
| <img src="images/2_1_ctrLoRA_training/test/merge_upper_baliwish_right.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/upper_baliwish_right_1.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/upper_baliwish_right_5.png" width="160"/> | <img src="images/2_1_ctrLoRA_training/test/upper_baliwish_right.png" width="160"/> |







---

### D. **B + C 결합 (권장) – α=0.3, \(w_{\text{tooth}}=5\)**
- **입력단 블렌딩(α=0.3)** 으로 간섭 완화  
- **학습단 가중 손실(\(w=5\))** 로 치아 디테일 복원  
- 효과: **Ghosting 감소 + 경계·질감 개선**(B/C 단독보다 우수)


---

## 4) 결과 (요약)


| 설정 | Ghosting(↓) | Boundary F1(↑) | HF-energy(↑) | LPIPS(↓) |
|---|:---:|---:|---:|---:|
| A. Naïve Multi-LoRA | 높음 | 0.79 | 1.12 | 0.176 |
| B. Pre-Blend (α=0.3) | 중 | 0.82 | 1.15 | 0.171 |
| C. Seg-Weighted (w=3) | 중 | 0.83 | 1.16 | 0.170 |
| **C. Seg-Weighted (w=5)** | **중~낮음** | **0.84** | **1.17** | **0.168** |
| C. Seg-Weighted (w=7) | 중 | 0.83 | 1.16 | 0.171 |
| **D. B+C (α=0.3, w=5)** | **낮음** | **0.85** | **1.20** | **0.165** |

**정성 관찰**  
- D에서 **이중 윤곽 소실**, 잇몸 bleed-in 억제  
- **에나멜 하이라이트**가 자연스럽고 과매끄러움 감소

---

## 5) 추론·학습 레시피 (재현 메모)

**Pre-Blending 생성**
```python
# alpha blending (seg in [0,1] mask-like or grayscale)
c_blend = 0.3 * c_seg + 0.7 * c_light
```

**학습(가중 손실)**
```
loss = mse(pred, noise) * (1 + (w_tooth - 1) * M_tooth)  # w_tooth=5
```





