# FLARE: Fingerprint Enhancement Modules - UNetEnh & PriorEnh

This repository contains the implementation of the **fingerprint enhancement modules** proposed in the FLARE framework, specifically:

- **UNetEnh**: A U-Net-based fingerprint enhancement network designed to improve ridge clarity.
- **PriorEnh**: A prior-guided enhancement network that leverages ridge prior maps to enhance robustness under varying fingerprint qualities.

These modules are part of the [FLARE](https://github.com/Yu-Yy/FLARE) framework for fingerprint recognition using fixed-length dense descriptors.


## 🔍 Overview

### 🔹 UNetEnh

- A standard U-Net variant for direct fingerprint image enhancement.

### 🔹 PriorEnh
- Incorporates an auxiliary **ridge prior latent codebook** extracted from high-quality rolled and plain fingerprints.
---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
````

### 2. Load a pretrained model

Download the [UNetEnh](https://cloud.tsinghua.edu.cn/f/2abf5af2c0064c24a262/?dl=1) model and place it in the `pretrained_model/unetenh` directory. Then, download both the [PriorEnh](https://cloud.tsinghua.edu.cn/f/968696b8cf8a4a9da82c/?dl=1) model and the [Prior](https://cloud.tsinghua.edu.cn/f/772b10a628ef4505a8a2/?dl=1) model, and place them in `pretrained_model/priorenh` directory.

### 3. Run enhancement

### 🔹 UNetEnh
```bash
python deploy_unetenh.py -f /path/to/image
```
or 
```bash
python deploy_unetenh.py -f /path/to/image -e
```

### 🔹 PriorEnh
```bash
python deploy_priorenh.py -f /path/to/image
```
or 
```bash
python deploy_unetenh.py -f /path/to/image -e
```

The -e flag enables an optional contrast enhancement step prior to processing.


## 📄 Citation

If you use these modules in your research, please cite:

```
@article{pan2025flare,
  title={Fixed-Length Dense Fingerprint Representation},
  author={Pan, Zhiyu and Guan, Xiongjun and Duan, Yongjie and Feng, Jianjiang and Zhou, Jie},
  journal={arXiv preprint arXiv:2505.03597},
  year={2025},
  url={https://arxiv.org/abs/2505.03597}
}
```

---

## 📬 Contact
For any questions or feedback, feel free to open an issue or contact [Zhiyu Pan](pzy20@mails.tsinghua.edu.cn).


---

## ⚠️ License & Usage Notice

This repository is released **for academic research and educational purposes only**.
**Commercial use is strictly prohibited.**

