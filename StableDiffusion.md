# 🧠 Stable Diffusion 정리

---

## 1️⃣ 개요 (Overview)

**Stable Diffusion = Multimodal Generative AI**  
```
Input
├── Text Prompt (자연어)
├── Image (참조 이미지, 마스크 등)
└── Other Modalities (Depth, Segmentation, Pose 등)

Process
├── Text Encoding → CLIP Transformer
├── Image Encoding → VAE Encoder → Latent Space z
├── Denoising & Refinement → UNet + Diffusion Process
└── Cross-Attention → 텍스트 의미와 시각 정보 결합

Output
└── VAE Decoder → 고해상도 이미지 생성 (512×512 이상)
```

---

## 2️⃣ 기본 원리 (Fundamental Principle)

Stable Diffusion은 다음 세 단계를 거칩니다:

| 단계 | 역할 | 핵심 기술 |
|------|------|-----------|
| 1️⃣ Embedding | 텍스트나 이미지를 벡터로 변환 | **CLIP Text Encoder / Image Encoder** |
| 2️⃣ Image Generation | 텍스트 벡터를 이미지 벡터로 매핑 | **Neural Network (UNet)** |
| 3️⃣ Refinement | 노이즈 제거 과정을 통해 이미지를 선명하게 | **Diffusion Model (Denoising)** |

---

### 🔹 2.1 Embedding

- **텍스트 임베딩(Text Embedding)**  
  사용자가 입력한 문장은 **CLIP(Contrastive Language–Image Pretraining)** 모델의 Transformer를 통해  
  의미적 벡터 공간으로 변환됩니다.  
  이 벡터는 모델이 이해하는 ‘개념적 표현(semantic embedding)’이 됩니다.

- **이미지 임베딩(Image Embedding)**  
  학습 중에는 **VAE(Variational Autoencoder)**의 인코더가 이미지 데이터를 **잠재 공간(latent space)**으로 압축합니다.  
  즉, 인간이 보는 픽셀 공간이 아니라, 모델이 이해할 수 있는 저차원 벡터 공간으로 변환됩니다.

> 💡 "인간이 볼 수 있는 것 → 컴퓨터가 이해할 수 있는 것(vector)"  
> embedding 과정을 통해 텍스트와 이미지가 공통 공간에서 매핑됨.

---

### 🔹 2.2 Image Generation (Mapping & Interpolation)

- **Neural Network**를 통해 Text Vector ↔ Image Vector 사이의 매핑 관계를 학습합니다.  
- 두 벡터의 연산이 “의미적 연산”으로 작동합니다.
  예를 들어:  
  `vector("penguin") + vector("clown") ≈ "penguin dressed as a clown"`
- 이런 성질을 통해 **이미지 합성(image interpolation)**이 가능해집니다.
- 출력층은 **sigmoid**를 사용하여 0~1 사이의 값을 가지며, bias 보정을 통해 시각적 품질을 개선합니다.

---

### 🔹 2.3 Refine Image (Diffusion Process)

- 초기에는 흐릿하고 노이즈가 많은 이미지를 생성합니다.  
- Diffusion Model은 이 이미지를 점진적으로 복원(denoising)하며, 최종적으로 선명한 이미지를 얻습니다.

#### Diffusion 학습 과정
1. 원본 이미지에 **가우시안 노이즈**를 단계적으로 추가 (forward process).  
2. 모델이 이 노이즈를 제거하도록 학습 (reverse process).  
3. 훈련 후에는 역과정을 통해 새로운 이미지를 생성.

---

## 3️⃣ 세부 작동 원리 (Detailed Architecture)

Stable Diffusion은 **Latent Diffusion Model (LDM)** 구조를 기반으로 동작합니다.

---

### 🔸 3.1 전체 구조 개요

'''
Text Prompt → CLIP Text Encoder → Text Embedding
↓
Image → VAE Encoder → Latent Representation z
↓
Noise 추가 → UNet (Denoising)
↑
Cross-Attention으로 Text 정보 결합
↓
VAE Decoder → 최종 이미지
'''

- Diffusion 과정은 **latent space**에서 수행되어 **연산량을 대폭 절감**합니다.
- 이 latent space는 VAE Encoder가 생성한 압축된 표현으로,  
  픽셀 공간보다 약 8~16배 작은 차원을 가집니다.

---

### 🔸 3.2 주요 구성 요소

| 구성요소 | 역할 | 설명 |
|-----------|------|------|
| **CLIP Text Encoder** | Text Embedding | Transformer 기반, 텍스트를 의미적 벡터로 변환 |
| **VAE (Autoencoder)** | Image Embedding/Decoding | 이미지를 latent space로 압축 및 복원 |
| **UNet (Denoiser)** | Diffusion Backbone | 노이즈 제거 및 latent 복원 |
| **Cross-Attention** | Text–Image 융합 | 텍스트의 의미를 latent feature에 주입 |
| **Scheduler (DDIM/DPM)** | Sampling 제어 | 노이즈 제거 단계를 조정하여 품질·속도 제어 |

---

### 🔸 3.3 Cross-Attention 작동 원리

- 각 UNet 레이어에서, **텍스트 임베딩**과 **latent feature map** 사이에 Cross-Attention 수행  
  → 텍스트의 의미적 정보가 이미지 생성에 반영됨  
- Attention 수식:
  \[
  \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
  \]
  여기서:
  - \( Q \): UNet의 feature map  
  - \( K, V \): Text embedding  

이 과정을 통해 “prompt를 반영한 이미지 생성”이 이루어집니다.

---

## 4️⃣ Stable Diffusion의 학습 과정

1. **데이터 준비**: 이미지–텍스트 쌍(예: LAION-5B)을 이용  
2. **VAE 학습**: 픽셀 이미지를 latent 공간으로 인코딩  
3. **Diffusion 학습**: latent 공간에서 노이즈 제거 모델(UNet) 훈련  
4. **CLIP 텍스트 결합**: 텍스트–이미지 상관 관계를 학습  
5. **샘플링(Sampling)**: 역 diffusion으로 latent 복원 후 VAE로 디코딩

---

## 5️⃣ 결과 및 특징

| 항목 | 내용 |
|------|------|
| **속도** | Latent Space에서 연산 → Pixel Diffusion보다 4~10배 빠름 |
| **품질** | FID, IS 등 주요 지표에서 GAN 대비 우수 |
| **유연성** | 텍스트, 이미지, depth, segmentation 등 다양한 condition 지원 |
| **확장성** | ControlNet, LoRA, DreamBooth 등 다양한 확장 구조와 결합 가능 |

---

## 6️⃣ Stable Diffusion의 의의

- **고해상도 이미지 합성의 민주화**: 누구나 GPU 한두 장으로 고품질 이미지 생성 가능  
- **범용성**: Text-to-Image뿐 아니라 Super-Resolution, Inpainting, Image-to-Image까지 확장  
- **효율성**: Pixel-space Diffusion 대비 수백 배의 효율 향상

---

## 7️⃣ Stable Diffusion의 확장 모델

| 확장 기술 | 핵심 아이디어 |
|------------|----------------|
| **ControlNet** | 외부 조건(Depth, Edge, Pose 등)을 제어 입력으로 사용 |
| **LoRA** | 가벼운 fine-tuning 기법 (저랭크 적응) |
| **DreamBooth** | 개인 데이터 기반 커스터마이즈 학습 |
| **StreamDiffusion** | 실시간 inference를 위한 latent streaming 구조 |

---

## 8️⃣ 한계점 (Limitations)

- Sequential Sampling으로 인해 GAN보다 여전히 느림  
- Autoencoder 품질이 전체 결과에 영향  
- Fine detail accuracy 제한  
- 텍스트 prompt 해석의 모호성 존재



