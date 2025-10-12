# Stable Diffusion 정리

<details>
<summary>목차</summary>

1. [Overview](#1-overview)  
2. [Fundamental Principle](#2-fundamental-principle)  
 2.1 [Embedding](#21-embedding)  
 2.2 [Image Generation (Mapping & Interpolation)](#22-image-generation-mapping--interpolation)  
 2.3 [Refine Image (Diffusion Process)](#23-refine-image-diffusion-process)  
3. [Detailed Architecture](#3-detailed-architecture)  
 3.1 [CLIP (Text Encoder)](#31-clip-text-encoder)  
  3.1.1 [Why CLIP?](#311-why-clip)  
  3.1.2 [입력/토크나이징/출력](#312-입력토크나이징출력)  
  3.1.3 [Cross-Attention로의 주입](#313-cross-attention로의-주입)  
  3.1.4 [Classifier-Free Guidance (CFG)와 Negative Prompt](#314-classifier-free-guidance-cfg와-negative-prompt)  
 3.2 [VAE](#32-vae)  
 3.3 [UNET](#33-unet)  
 3.4 [Scheduler](#34-scheduler)  
4. [Stable Diffusion의 학습 과정](#4-stable-diffusion의-학습-과정)  
5. [결과 및 특징](#5-결과-및-특징)  
6. [Stable Diffusion의 의의](#6-stable-diffusion의-의의)  
7. [Stable Diffusion의 확장 모델](#7-stable-diffusion의-확장-모델)

</details>
---

## 1. Overview

**Stable Diffusion = Multimodal Generative AI**  
```
Input
├── Text Prompt
├── Image
└── Other Modalities (sementic map, depth map, representations ...)

Process
├── Text Encoding → CLIP Transformer
├── Image Encoding → VAE Encoder → Latent Space z
├── Denoising & Refinement → UNet + Diffusion Process
└── Cross-Attention → 텍스트 의미와 시각 정보 결합

Output
└── VAE Decoder → 고해상도 이미지 생성 (512×512 이상)
```

- Latent Space 기반 Diffusion → 효율적 연산

- 멀티모달 입력 지원 → 언어+시각 융합

- 텍스트로부터 의미적·시각적 일관성 있는 이미지 생성

---

## 2. Fundamental Principle

pixel space 대신 **latent space** 에서 denoising 과정을 수행하는 효율적 확산 모델.

| 단계 | 역할 | 핵심 기술 |
|------|------|-----------|
| 1. Embedding | 텍스트나 이미지를 벡터로 변환 | **CLIP Text Encoder / Image Encoder** |
| 2. Image Generation | 텍스트 벡터를 이미지 벡터로 매핑 | **Neural Network (UNet)** |
| 3. Refinement | 노이즈 제거 과정을 통해 이미지를 선명하게 | **Diffusion Model (Denoising)** |

---

### 2.1 Embedding

- **Text Embedding**  
  사용자가 입력한 문장은 **CLIP(Contrastive Language–Image Pretraining)** 모델의 Transformer를 통해  
  의미적 벡터 공간으로 변환된다.  
  이 벡터는 모델이 이해하는 ‘semantic embedding’이 된다.

- **Image Embedding**  
  학습 중에는 **VAE(Variational Autoencoder)**의 Encoder가 이미지 데이터를 **latent space**로 압축한다.  
  즉, 인간이 보는 pixel space가 아니라, 모델이 이해할 수 있는 저차원 벡터 공간으로 변환된다.

> 💡 "Human-perceivable → Machine-interpretable (vector)"  
> Embedding 과정을 통해 텍스트와 이미지가 공통 공간에서 매핑된다.

---

### 2.2 Image Generation (Mapping & Interpolation)

- **Neural Network**를 통해 Text Vector ↔ Image Vector 사이의 매핑 관계를 학습한다.  
- 두 벡터의 연산이 “semantic operation”으로 작동한다.  
  예를 들어:  
  `vector("penguin") + vector("clown") ≈ "penguin dressed as a clown"`
- 이러한 성질을 통해 **image interpolation**이 가능하다.  
- 출력층은 **sigmoid**를 사용하여 0~1 사이의 값을 가지며, bias 보정을 통해 시각적 품질을 개선한다.

---

### 2.3 Refine Image (Diffusion Process)

- 초기에는 흐릿하고 노이즈가 많은 이미지를 생성한다.  
- **Diffusion Model**은 이 이미지를 점진적으로 Denoising하며, 최종적으로 선명한 이미지를 얻는다.

#### Diffusion Training Process
1. 원본 이미지에 **Gaussian noise**를 단계적으로 추가 (forward process).  
2. 모델이 이 노이즈를 제거하도록 학습 (reverse process).  
3. 학습이 완료된 후에는 reverse process를 통해 새로운 이미지를 생성한다.

---

## 3. Detailed Architecture

Stable Diffusion은 **Latent Diffusion Model (LDM)** 구조를 기반으로 동작


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

- Diffusion 과정은 **latent space**에서 수행되어 **연산량을 절감**함
- latent space는 VAE Encoder가 생성한 압축된 표현으로, 픽셀 공간보다 약 8~16배 작은 차원을 가짐

---

### 3.1 CLIP (Text Encoder)

사용자 프롬프트를 벡터화한 `text embedding`을 만들어 UNet의 Cross-Attention에 공급    
→ 언어 의미를 시각 피처에 주입해 **텍스트 조건부 생성**을 가능하게 한다.


#### 3.1.1 Why CLIP?
- **Multimodal alignment**: 텍스트–이미지 쌍 기반의 contrastive 학습으로 두 모달리티를 같은 semantic space에 정렬
- **Generalization**: 거대 웹 스케일 데이터로 학습되어 **스타일·콘셉트** 전반에 강한 일반화
- **Lightweight at inference**: 인퍼런스에서는 텍스트만 인코딩하므로 비용이 낮음(embedding 캐시 가능)

> 참고: Stable Diffusion 1.x/SDXL은 CLIP 계열을 사용함 (후속 계열/다른 파이프라인은 T5 등 다른 텍스트 인코더를 쓰기도 함. 여기서는 CLIP 사용을 기준으로 설명.)


#### 3.1.2 입력/토크나이징/출력

**Tokenization**: BPE 기반 토크나이저 사용. 일반적으로 **max length = 77 tokens** (SD 1.x/SDXL)
  - Tokenization은 사용자가 입력한 문장을 **컴퓨터가 이해할 수 있는 단위(토큰)**로 분할하는 과정
    Stable Diffusion의 CLIP은 단어를 직접 이해하지 않고,  
    각 단어를 **indexed embedding**로 변환해야만 처리할 수 있다.  

  - CLIP은 **BPE(Byte Pair Encoding)** 기반 토크나이저를 사용한다.  
    - 긴 단어를 자주 등장하는 부분(subword) 단위로 분할해 어휘 수를 줄이면서 의미 보존.  
    - ex) “strawberries” → “straw”, “ber”, “ries”  
    - 이런 방식으로 **새로운 단어**나 **철자 변형(예: stylized, plural)** 도 안정적으로 인코딩 가능

- 입력 문장은 다음 순서로 처리된다
  1. **문장 정규화**: 소문자화, 불필요한 공백 제거 등 전처리.  
  2. **Tokenization (BPE)**: 문장을 subword 단위로 분할하고 토큰 ID 시퀀스로 변환.  
  3. **[BOS]/[EOS]/[PAD]** 토큰 추가: 문장의 시작/끝/빈 자리 표시용.  
  4. **길이 제한 적용**: Stable Diffusion의 CLIP은 **최대 77 tokens**까지만 사용.  
     → 초과 부분은 **truncate**되어 무시됨.  
     → 따라서 중요한 단어는 **문장 앞부분**에 배치하는 것이 중요.
 
  ex)Prompt: "A cute penguin wearing a clown costume"
    → [BOS] a, cute, penguin, wearing, a, clown, costume, [EOS], [PAD]...
    → token IDs: [49406, 320, 12345, 2891, 320, 7842, 9121, 49407, 0, ...]

출력구조 : 
  - **Per-token hidden states**
  - shape: `(B, T, D)`  
    - `B`: batch size  
    - `T`: token 수 (≈77)  
    - `D`: embedding dimension (ViT-L/14의 경우 768)  
  - 각 토큰이 의미 벡터로 변환되어 UNet의 **Cross-Attention layer**에서 K/V(Key/Value)로 사용됨.  
  - 즉, 각 단어가 이미지 생성 과정에서 **어떤 시각적 요소로 반영될지**를 결정.
  
  - **Pooled embedding (CLS/mean)**
  - shape: `(B, D)`  
  - 문장 전체의 전역적 의미(global meaning)를 나타내는 벡터.  
  - SDXL 등에서는 **전체 스타일·분위기** 제어 신호로 사용.

  

#### 3.1.3 Cross-Attention로의 주입
UNet의 각 블록에서 **Q/K/V**를 다음과 같이 구성:
- `Q = W_Q * F` : UNet의 feature map(공간 해상도 H×W를 펼친 시퀀스)에서 추출  
- `K = W_K * E_text`, `V = W_V * E_text` : CLIP의 per-token embedding에서 추출  
- Attention:
  \[
  \mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
  \]
- 직관: UNet 피처가 텍스트 토큰(단어/구)의 의미를 **선택적으로 참조**해 시각 세부 묘사에 반영


#### 3.1.4 Classifier-Free Guidance (CFG)와 Negative Prompt
- **Unconditional branch**: 빈 문자열 `""`(또는 negative prompt)로 얻은 embedding을 별도로 계산
- **Guidance**:
  \[
  \hat{\epsilon} = \epsilon_{\text{uncond}} + s \cdot \big(\epsilon_{\text{cond}} - \epsilon_{\text{uncond}}\big)
  \]
  - \( s \): guidance scale (권장 범위 예: **5–9**; 너무 크면 포즈/구도 왜곡, 너무 작으면 조건 약화)
- **Negative prompt**: 원치 않는 개념을 명시(예: “blurry, low quality”) → **uncond** 대신 **neg** 임베딩 사용

---

### 3.2 VAE


---

### 3.3 UNET

---

### 3.4 Scheduler


---

## 4. Stable Diffusion의 학습 과정

1. **데이터 준비**: 이미지–텍스트 쌍(예: LAION-5B)을 이용  
2. **VAE 학습**: 픽셀 이미지를 latent 공간으로 인코딩  
3. **Diffusion 학습**: latent 공간에서 노이즈 제거 모델(UNet) 훈련  
4. **CLIP 텍스트 결합**: 텍스트–이미지 상관 관계를 학습  
5. **샘플링(Sampling)**: 역 diffusion으로 latent 복원 후 VAE로 디코딩

---

## 5. 결과 및 특징

| 항목 | 내용 |
|------|------|
| **속도** | Latent Space에서 연산 → Pixel Diffusion보다 4~10배 빠름 |
| **품질** | FID, IS 등 주요 지표에서 GAN 대비 우수 |
| **유연성** | 텍스트, 이미지, depth, segmentation 등 다양한 condition 지원 |
| **확장성** | ControlNet, LoRA, DreamBooth 등 다양한 확장 구조와 결합 가능 |

---

## 6. Stable Diffusion의 의의

- **고해상도 이미지 합성의 민주화**: 누구나 GPU 한두 장으로 고품질 이미지 생성 가능  
- **범용성**: Text-to-Image뿐 아니라 Super-Resolution, Inpainting, Image-to-Image까지 확장  
- **효율성**: Pixel-space Diffusion 대비 수백 배의 효율 향상

---

## 7. Stable Diffusion의 확장 모델

| 확장 기술 | 핵심 아이디어 |
|------------|----------------|
| **ControlNet** | 외부 조건(Depth, Edge, Pose 등)을 제어 입력으로 사용 |
| **LoRA** | 가벼운 fine-tuning 기법 (저랭크 적응) |
| **DreamBooth** | 개인 데이터 기반 커스터마이즈 학습 |


---

## 8. 한계점 (Limitations)

- Sequential Sampling으로 인해 GAN보다 여전히 느림  
- Autoencoder 품질이 전체 결과에 영향  
- Fine detail accuracy 제한  
- 텍스트 prompt 해석의 모호성 존재



