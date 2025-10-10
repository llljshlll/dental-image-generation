# CtrLoRA: AN EXTENSIBLE AND EFFICIENT FRAMEWORK FOR CONTROLLABLE IMAGE GENERATION

>  원문 논문: [ICLR 2025](https://github.com/xyfJASON/ctrlora)  
>  저자: Yifeng Xu, Zhenliang He, Shiguang Shan, Xilin Chen

---

## 1. Overview

**CtrLoRA = Controllable Image-to-Image Generation Framework**  
```
Input
├── Condition Image (Canny, Depth, Segmentation, Pose, etc.)
├── Text Prompt (optional)
└── Additional Conditions (multi-condition possible)

Process
├── Condition Encoding → Pretrained VAE → Condition Embedding
├── Base ControlNet → 공통 I2I 지식 학습 (shared backbone)
├── LoRA (Low-Rank Adaptation) → 조건별 세부 특성 학습
└── UNet Diffusion Backbone → 노이즈 예측 및 이미지 복원

Output
└── Stable Diffusion Decoder (VAE) → 제어된 고해상도 이미지 생성
```


- **Base + LoRA 구조**  
  → Base ControlNet이 **공통적인 Image-to-Image 생성 지식**을 학습하고,  
    LoRA가 **각 조건의 고유 특성**을 저비용으로 학습함.

- **Few-shot Adaptation (1,000 images / <1hr on 1 GPU)**  
  → 새로운 조건 추가 시 **ControlNet 전체 재학습 없이**,  
    **LoRA만 학습**하여 빠르게 적응 가능.

- **Multi-Conditional Generation 지원**  
  → 여러 LoRA를 조합해 하나의 이미지에  
    **다중 조건 제어 (예: Segmentation + Lighting)** 가능.

- **Pretrained VAE 기반 Condition Embedding**  
  → 기존의 무작위 CNN 대신 Stable Diffusion의 **VAE Encoder**를 활용해  
    조건 이미지를 잠재공간에 임베딩함.  
  → 학습 수렴 속도 향상 및 ControlNet의 **Sudden Convergence 현상 제거**.

---

## 2. Fundamental Principle

**CtrLoRA**는 기존 **ControlNet**의 구조를 확장하여,  
여러 조건 이미지를 효율적으로 학습하고 새로운 조건에도 빠르게 적응할 수 있는  
**확장형 Controllable Diffusion Framework**이다.

| 단계 | 역할 | 핵심 기술 |
|------|------|-----------|
| 1. Condition Embedding | 조건 이미지를 잠재공간(latent space)으로 변환 | **Pretrained VAE Encoder** |
| 2. Base ControlNet | 공통적인 I2I(image-to-image) 생성 지식 학습 | **Shared UNet Backbone** |
| 3. Condition-specific LoRA | 조건별 세부 특성 학습 (저비용, 확장성) | **Low-Rank Adaptation (LoRA)** |
| 4. Image Denoising & Generation | 조건 기반으로 노이즈를 제거하며 이미지 생성 | **Diffusion Process (DDIM / DPM)** |

---

### 2.1 Condition Embedding

- 기존 ControlNet은 **무작위 초기화된 CNN**을 사용해 조건 이미지를 임베딩했으나,  
  이는 학습 초기에 의미 있는 피처를 추출하지 못해 **수렴이 느리고 불안정**했다.  
- **CtrLoRA는 Stable Diffusion의 Pretrained VAE Encoder를 사용**하여  
  조건 이미지를 잠재공간으로 변환한다.  
- 이 방식은  
  - **빠른 수렴 (convergence)**  
  - **안정적인 학습**  
  - **ControlNet의 sudden convergence 현상 제거**  
  를 동시에 달성한다.

> 💡 “Random CNN → Pretrained VAE”  
> 이미지를 임베딩할 때, 사전 학습된 VAE의 표현 공간을 그대로 활용하여  
> 학습 효율을 극대화함.



### 2.2 Base ControlNet (Shared Backbone)

- 여러 조건(canny, depth, segmentation, skeleton 등)을 하나의 네트워크로 학습한다.  
- 모든 condition 데이터를 **공통 Loss Function** 아래서 학습하여  
  **I2I(이미지-이미지) 생성의 일반 지식(common knowledge)** 을 획득한다.  
- 각 condition별 LoRA가 추가되어, Base ControlNet은  
  “공통적 구조 학습”에 집중하고, LoRA는 “조건별 특수성 학습”에 집중한다.

> 💡 Base ControlNet = General I2I Knowledge Learner  
> LoRA = Condition Expert Module


### 2.3 Condition-specific LoRA (Low-Rank Adaptation)

- LoRA는 큰 weight 행렬의 변화를 **저랭크 분해(ΔW = BA)** 형태로 근사하여  
  **학습 가능한 파라미터 수를 90% 이상 절감**한다.  
- 새로운 조건 추가 시, Base ControlNet은 고정하고 LoRA만 학습한다.  
- 약 **1,000개 이미지 / 1시간 미만 / 단일 GPU (RTX 4090)** 으로도 학습 가능.  
- LoRA Rank = 128로 설정 시 약 **37M 파라미터**만 업데이트됨.

> 💡 “Train Once, Adapt Many”  
> ControlNet 전체를 다시 학습하지 않고, LoRA를 추가하는 것만으로  
> 새로운 condition을 빠르게 지원 가능.



### 2.4 Denoising & Image Generation (Diffusion Process)

- Base ControlNet과 LoRA의 출력은 Stable Diffusion의 UNet으로 전달되어  
  **조건 기반 노이즈 제거(reverse diffusion)** 를 수행한다.  
- Sampling 단계에서는 **DDIM(50 steps)** 또는 **DPM-Solver**를 사용하며,  
  classifier-free guidance scale은 일반적으로 **7.5**로 설정된다.  
- 여러 LoRA를 합성하면 **multi-conditional generation**이 가능하며,  
  각 조건의 영향은 가중치로 조절할 수 있다.

---

## 3. Detailed Architecture

CtrLoRA는 **ControlNet + LoRA + Pretrained VAE**를 결합한  
**확장형 Controllable Latent Diffusion 구조**로 설계되어 있다.  

```
Condition Image → Pretrained VAE Encoder → Condition Embedding
↓
Base ControlNet (shared backbone) + Condition-specific LoRA
↓
UNet (Denoising Network)
↑
Cross-Attention → Text 조건 결합 (optional)
↓
VAE Decoder → 제어된 이미지 복원
```


- Diffusion은 Stable Diffusion과 동일하게 **latent space**에서 수행되어 효율적  
- ControlNet의 구조를 **Base + LoRA 모듈**로 분리하여 **확장성** 확보  
- Condition Embedding에는 **VAE Encoder**를 사용해 학습 안정성 향상  

---

### 3.1 Base ControlNet (Shared Backbone)

Base ControlNet은 **여러 조건(Condition)**을 하나의 네트워크로 통합 학습하기 위해 설계된  
**공유형 I2I(Image-to-Image) 생성 모듈**이다.

#### 3.1.1 구조 개요
- 기본적으로 Stable Diffusion의 **UNet encoder 구조**를 따르며,  
  입력으로 condition feature를 받아 latent representation을 변환한다.  
- 각 block은 ControlNet의 residual branch를 포함하며,  
  Stable Diffusion의 feature flow에 추가 정보를 주입한다.

| 구성 요소 | 설명 |
|------------|------|
| **Encoder (SD Encoder 복제)** | Stable Diffusion의 UNet encoder를 그대로 복제하여 초기화. |
| **Residual Block** | Condition feature를 각 stage에 주입 (control signal). |
| **Skip Connection** | Base ControlNet과 SD의 UNet 간 feature alignment를 유지. |
| **Zero Convolution Layer** | 초기 영향 최소화를 위해 모든 residual branch를 0으로 시작. |

> 💡 Base ControlNet은 SD의 UNet 구조를 그대로 공유하지만,  
> 입력으로 condition feature를 받아 “control-aware feature map”을 형성함.



#### 3.1.2 학습 방식
- Base ControlNet은 **9가지 base condition** (Canny, Depth, Skeleton, Segmentation 등)을 동시에 학습.  
- 각 조건은 **개별 LoRA**가 연결된 형태로 주입되며,  
  Base ControlNet은 “공통적인 I2I 생성 지식”을 학습한다.  
- 학습 시, 배치 단위로 condition을 순환하며 다음 과정을 반복한다:
  1. 한 번에 하나의 condition 데이터셋을 선택.  
  2. 해당 condition에 대응하는 LoRA만 활성화.  
  3. Base ControlNet의 공유 파라미터와 해당 LoRA를 함께 업데이트.  

> 💡 여러 조건을 **하나의 손실 함수로 통합**해 학습하므로,  
> 공통된 구조·조명·형태 등의 시각적 패턴을 일반화할 수 있다.



#### 3.1.3 역할
- **공통적 이미지 생성 능력 학습 (General I2I knowledge)**  
  → 다양한 조건을 통해 “이미지 변환의 일반 원리”를 습득.  
- **LoRA 학습의 기반 모델로 사용**  
  → Base ControlNet을 고정(freeze)하고 LoRA만 학습해 새로운 조건에 빠르게 적응.  

> 💡 ControlNet을 조건별로 새로 학습하던 기존 방식과 달리,  
> CtrLoRA는 Base ControlNet을 한 번만 학습하면 된다.

---

### 3.2 LoRA (Low-Rank Adaptation)

LoRA는 **Base ControlNet 위에 부착되는 경량 적응 모듈**로,  
각 condition의 **세부적 특성(local feature)**을 학습한다.

#### 3.2.1 핵심 개념
- 기존 full fine-tuning에서는 모든 weight \( W \)를 직접 업데이트해야 하지만,  
  LoRA는 weight 변화를 **저랭크 근사(ΔW = B·A)** 로 표현한다.  
  - \( A \in \mathbb{R}^{r \times d} \), \( B \in \mathbb{R}^{d \times r} \), \( r \ll d \)
- 학습 가능한 파라미터 수가 약 **90% 이상 감소**하며,  
  Base ControlNet의 파라미터는 고정된 상태로 유지된다.

| 구성 요소 | 설명 |
|------------|------|
| **ΔW = B·A (rank = 128)** | LoRA의 저랭크 업데이트 행렬 (rank는 128로 설정) |
| **Trainable Layer** | Base ControlNet의 각 Linear/Conv 레이어마다 LoRA 부착 |
| **Frozen Backbone** | Base ControlNet의 파라미터는 고정됨 |
| **Lightweight Parameter** | 약 37M 파라미터 (ControlNet 대비 1/10 수준) |



#### 3.2.2 학습 및 적용
1. Base ControlNet을 고정하고, 새로운 condition 데이터로 LoRA를 학습.  
2. LoRA는 각 Linear Layer의 **Residual Update** 형태로 삽입됨.  
3. 학습 완료 후, LoRA를 Base ControlNet과 합성하여 inference 수행:
   \[
   W' = W + \Delta W = W + BA
   \]
4. 여러 LoRA를 동시에 합산하여 **Multi-Conditional Generation** 가능.

> 💡 LoRA는 condition별로 독립적으로 저장·배포 가능하며,  
> 용량이 작아 **공유와 재사용이 용이**하다.

---

### 3.3 Condition Embedding Network

CtrLoRA의 핵심 중 하나는 **조건(condition) 이미지를 효율적으로 임베딩**하는 방식이다.  
기존 ControlNet은 단순한 **랜덤 초기화 CNN(convolutional encoder)** 를 사용했지만,  
CtrLoRA는 **Stable Diffusion의 Pretrained VAE Encoder**를 그대로 활용한다.

```
Condition Image → VAE Encoder → Latent Representation (z_c)
↓
Base ControlNet + LoRA 입력
```



#### 3.3.1 기존 ControlNet의 한계
- ControlNet은 condition image를 feature로 변환하기 위해  
  **임의 초기화된 CNN**을 사용.  
- 학습 초반에는 유의미한 피처를 추출하지 못해,  
  **수렴이 매우 느리고 불안정함**.  
- “Sudden Convergence” 현상 발생:  
  → 학습이 일정 단계까지 전혀 진행되지 않다가,  
    갑자기 condition을 강하게 반영하며 폭발적으로 수렴함.

> 💡 이 현상은 조건 임베딩 네트워크가  
> 학습 초기에 “무의미한 latent 공간”을 만들어내기 때문임.



#### 3.3.2 Pretrained VAE Encoder의 도입
CtrLoRA는 이를 해결하기 위해 **Stable Diffusion의 VAE Encoder**를 condition embedding network로 채택하였다.  
즉, condition image는 VAE Encoder를 통해 즉시 **latent representation**으로 변환된다.

\[
z_c = \text{VAE}_{enc}(c)
\]

- **VAE Encoder**는 원래 이미지를 latent space로 압축하도록 학습되어 있으므로,  
  이미 강력한 시각 표현(visual representation)을 보유.  
- Base ControlNet은 Stable Diffusion의 Encoder 구조를 복제하여 초기화하기 때문에,  
  두 네트워크의 입력 공간이 **정확히 정렬(aligned)** 되어 있음.  
- 이로 인해 condition feature가 **별도 학습 없이도 즉시 의미 공간에 매핑**됨.

| 항목 | 기존 ControlNet | CtrLoRA |
|------|------------------|----------|
| Condition Encoder | Random CNN | Pretrained VAE (SD Encoder) |
| 초기 수렴 속도 | 느림 / 불안정 | 빠름 / 안정적 |
| Sudden Convergence | 존재 | 제거됨 |
| 학습 안정성 | 낮음 | 매우 높음 |



#### 3.3.3 효과 및 장점
- **빠른 수렴**: 학습 초기부터 의미 있는 condition feature를 전달함.  
- **훈련 안정성 향상**: gradient 폭주나 loss 진동 현상 감소.  
- **정확한 feature alignment**: VAE latent space와 ControlNet 입력 공간이 일치.  
- **추가 학습 불필요**: VAE는 이미 SD에서 사전 학습된 모듈이므로, 재학습 없이 바로 활용 가능.

> 💡 결과적으로, CtrLoRA는 condition embedding을  
> “pretrained latent encoder → immediate alignment” 형태로 단순화하여  
> 학습 효율과 안정성을 동시에 확보했다.

---

### 3.4 Inference (Multi-Conditional Generation)

CtrLoRA의 추론(inference)은 **Base ControlNet**과 **여러 조건별 LoRA 모듈**을  
동시에 조합하여 이미지를 생성하는 과정이다.  
Stable Diffusion의 denoising 과정 위에서 동작하며,  
각 LoRA의 출력을 **가중합(weighted sum)** 하여 최종 제어 신호로 사용한다.

```
Condition 1 (Segmentation) ┐
Condition 2 (Lighting) ├──> 각 LoRA → Feature Map
Condition 3 (Normal) ┘
↓
Base ControlNet (shared)
↓
UNet Denoiser → Diffusion Sampling
↓
VAE Decoder → 최종 이미지

```


#### 3.4.1 Multi-Conditional Feature Aggregation

여러 LoRA를 동시에 적용할 때,  
각 LoRA는 동일한 Base ControlNet의 feature map에 대해  
조건별 잔차(residual)를 생성하고 이를 합산한다.

\[
C_{\theta, \Psi}(z, c) = C_{\theta}(z) + \sum_{i=1}^{N} w_i \cdot L_{\psi_i}(z, c_i)
\]

- \( C_{\theta} \): Base ControlNet  
- \( L_{\psi_i} \): i번째 condition의 LoRA  
- \( w_i \): 해당 조건의 가중치(weight, 기본값 1.0)  
- \( z \): latent representation  
- \( c_i \): 각 condition image의 embedding  

> 💡 이 구조 덕분에, 여러 조건(예: segmentation + lighting + pose)을  
> 별도 네트워크 병합 없이 단일 forward pass로 통합 가능하다.



#### 3.4.2 Denoising Process

CtrLoRA는 Stable Diffusion의 denoising 과정과 동일하게 작동한다.  
Base ControlNet과 LoRA의 출력을 UNet에 주입하여  
latent 공간에서 노이즈를 점진적으로 제거한다.

\[
\epsilon_\theta(x_t, c) = D(E(x_t), C_{\theta, \Psi}(z, c))
\]

- \( x_t \): 노이즈가 추가된 latent 이미지  
- \( E, D \): Stable Diffusion의 Encoder/Decoder  
- \( C_{\theta, \Psi} \): Base ControlNet + LoRA 조합  
- 출력: 다음 timestep으로 전달될 노이즈 예측값  

Sampling은 일반적으로 **DDIM (50 steps)** 또는 **DPM-Solver**를 사용하며,  
Classifier-Free Guidance Scale은 **7.5** 전후로 설정한다.



#### 3.4.3 Conditional Strength & Guidance

각 LoRA의 기여도는 가중치 \( w_i \)로 조절 가능하며,  
이 값을 높일수록 해당 조건의 영향력이 커진다.

| Condition | Weight (예시) | 결과 |
|------------|----------------|------|
| Segmentation | 1.0 | 구조적 형태 유지 |
| Lighting | 0.5 | 색상 및 음영만 부분 반영 |
| Normal | 0.3 | 표면 방향감만 약하게 반영 |

> 💡 “가중치 조절”은 ControlNet의 strength 개념과 유사하며,  
> 여러 조건의 밸런스를 직접 제어할 수 있다.  



#### 3.4.4 Inference Pipeline Summary
1. **Condition Encoding**: 모든 condition image → VAE Encoder → latent embedding 생성  
2. **LoRA Aggregation**: 각 LoRA의 출력 weighted sum → Base ControlNet feature에 주입  
3. **UNet Denoising**: Stable Diffusion의 latent space에서 noise 제거  
4. **Decoding**: 최종 latent → VAE Decoder → 고해상도 이미지 복원  

---

> 💡 결과

