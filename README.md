# Streaming Diffusion for Real-Time Dental Image Generation
**Goal:** 실시간 Dental 영상 생성에서 구조적 일관성과 디테일 유지 개선  
**Keywords:** Stable Diffusion, StreamDiffusion, ctrLoRA, ControlNet++, Weighted LoRA

<img src="images/pipeline_overview.png" width="700"/>

---

## Overview
이 프로젝트는 Stable Diffusion 기반 dental image generation의 **real-time 성능 저하** 및  
**세부 디테일 손실** 문제를 해결하기 위해 진행되었습니다.

- **Stable Diffusion** → 기본 구조 학습  
- **Stream Diffusion** → 실시간 추론 구조 적용  
- **ControlNet / LoRA / ctrLoRA** → 조건 기반 제어 실험  
- **Weighted LoRA + ControlNet++** → multi-condition interference 문제 해결
