# DIV2K_SR_chalge
Final project - ESPCN Super Resolution models and ablation study
# DIV2K Super Resolution Challenge

### Deep Learning and its Applications to Signal and Image Processing and Analysis
**Course:** 361.2.1120  
**Final Project**  
# Authors:   Or Shraga , Gal Apple
---

## 📌 **Project Overview**

This project tackles **single-image super-resolution** using the DIV2K dataset. We compare:

1. **Model A (Vanilla ESPCN)** – Standard Efficient Sub-Pixel Convolutional Network.
2. **Model B (Modified ESPCN-Residual)** – Adds local residual blocks + a global skip connection to focus on high-frequency details.
3. **Ablation Study (NoGlobal)** – Tests Model B **without** the global skip to evaluate its importance.

---

## 🎯 **Objective**

Enhance LR images by **×4 upscaling** to produce HR images with:

- **Sharper textures**
- **Higher PSNR, SSIM**
- **Lower FID** (better perceptual quality)

---

## 🗃 **Dataset**

- **DIV2K**: 800 training + 100 validation images.
- Images converted from **RGB to YCbCr**.
- **Only Y channel** used for training/validation.
- **Preprocessing:**
  - **Training:** Random 384×384 crops → downscaled ×4 → 96×96 LR patches.
  - **Validation/Test:** Center crop + downscale.
  - **Test:** CbCr upsampled by bicubic interpolation.

---

## 🏗 **Models**

### **Model A: Vanilla ESPCN**
- 3 conv layers + PixelShuffle.
- Learns full HR mapping directly.

### **Model B: ESPCN + Residuals**
- Head Conv → **4 local residual blocks** → Tail Conv → PixelShuffle.
- **Global skip connection** adds bicubic upsample to learned residual.
- Focuses on **predicting high-frequency details only**.

### **Ablation (NoGlobal)**
- Same as Model B but **removes the global skip**, forcing the network to learn the full HR mapping alone.

---

## 🧪 **Evaluation Metrics**

| Metric | Description |
|--|--|
| **PSNR** | Measures pixel-level fidelity (higher is better) |
| **SSIM** | Measures structural similarity (higher is better) |
| **FID** | Measures perceptual realism via InceptionNet features (lower is better) |

Calculated using the **`piq` library**.

---

## ⚙️ **Hyperparameters**

- **Scale factor:** 4
- **Batch size:** 16 (train), 4 (val/test)
- **Optimizer:** Adam (lr=1e-4)
- **Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)
- **Loss:** MSE on Y channel
- **Seeds:** [42, 123, 999]

---

## 📈 **Results (Example, Seed=42)**

| Model | PSNR | SSIM | FID |
|--|--|--|--|
| **Vanilla ESPCN** | ~25.4 dB | 0.862 | 122 |
| **ESPCN-Residual** | ~26.2 dB | 0.876 | 102 |
| **Ablation (NoGlobal)** | ~25.6 dB | 0.865 | 117 |

✅ **Residual model outperforms Vanilla and Ablation** in all metrics.

---

## 🔬 **Key Insights**

✔ **Global skip connection** accelerates convergence and boosts PSNR by ~0.7 dB.  
✔ **Ablation study confirms** removing it reduces performance, proving its value.

---

## 💻 **How to Use**

1. **Place `kaggle.json`** in the project directory.  
2. Modify:
   ```python
   project_path = '/content/drive/My Drive/DIV2K_SR_chalge'


3.

to match your local or Colab environment.

Run:

python main.py
place your kaggle.json in the root folder.
4. Adjust model_type in run_model() to 'base', 'residual', or 'no_global'.

🗂 Code Structure
main.py – Pipeline entry point, runs experiments

train_utils.py – Training and evaluation utilities (e.g. run_model, run_model_multiple_seeds)

model.py – Contains ESPCN, ESPCN-Residual, and Ablation model classes

data.py – Data loading and preprocessing logic

eval.py – Evaluation metrics calculation (PSNR, SSIM, FID)

plot.py – Visualization and plotting utilities

utils.py – General utility functions (e.g. set_seed)

requirements.txt – Project dependencies list

README.md – Project description, usage instructions, and results

✅ All file paths are relative for portability.

🧠 Future Work
Train with SSIM or combined PSNR+SSIM loss

Extend to scale factors ×8, ×16

Integrate with GhostNet for embedded real-time deployment.

📚 References
ESPCN Paper https://arxiv.org/abs/1609.05158

DIV2K Dataset https://www.kaggle.com/datasets/joe1995/div2k-dataset

ESPCN GitHub https://github.com/Lornatang/ESPCN-PyTorch
