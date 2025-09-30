# Real-Time Dental Image Generation with Stable Diffusion

## 📌 Description
This project implements a **real-time Stable Diffusion-based system** for dental image generation, focusing on **enhancing structural details of teeth**.  
By integrating **StreamDiffusion**, **ControlNet**, and **segmentation-guided LoRA**, the model restores fine details while maintaining **real-time inference speed**.  

**Key Features:**
- Real-time image generation pipeline
- Enhanced structural detail in dental imagery
- Integration of StreamDiffusion + ControlNet + Segmentation-guided LoRA
- Potential applications: dental imaging, interactive medical simulation

---

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/username/dental-image-generation.git
cd dental-image-generation

# Create environment
conda create -n dental-ai python=3.10 -y
conda activate dental-ai

# Install requirements
pip install -r requirements.txt
```
---

## 🚀 Usage
```bash
# Run inference
python inference.py \
  --input ./examples/input.png \
  --output ./results/output.png \
  --config ./configs/streamdiffusion_dental.yaml \
  --ckpt ./ckpts/dental_lora.ckpt
```

