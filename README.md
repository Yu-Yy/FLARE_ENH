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

Download the [UNetEnh](https://drive.google.com/file/d/1U0uP8XoxWc90IPlEGnt2KIe0ATAA2VKl/view?usp=drive_link) model and place it in the `pretrained_model/unetenh` directory. Then, download both the [PriorEnh](https://drive.google.com/file/d/1h3JD6ZhS_TUaCmBhINqKZ0Pb1ad-FRao/view?usp=drive_link) model and the [Prior](https://drive.google.com/file/d/14c0A4qRo_lrqa83e-_UpBvu79qEGkK5q/view?usp=drive_link) model, and place them in `pretrained_model/priorenh` directory.

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
python deploy_priorenh.py -f /path/to/image -e
```

`-w` sets the weight of the ridge-prior features (default 0.5).

The -e flag enables an optional contrast enhancement step prior to processing.


## 📄 Citation

If you use these modules in your research, please cite:

```
@ARTICLE{pan2025flare,
  author={Pan, Zhiyu and Guan, Xiongjun and Duan, Yongjie and Feng, Jianjiang and Zhou, Jie},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Fixed-Length Dense Fingerprint Representation With Alignment and Robust Enhancement}, 
  year={2026},
  volume={21},
  pages={1751-1765},
}
```

---

## 📬 Contact
For any questions or feedback, feel free to open an issue or contact [Zhiyu Pan](pzy20@mails.tsinghua.edu.cn).


---

## ⚠️ License & Usage Notice

This repository is released **for academic research and educational purposes only**.
**Commercial use is strictly prohibited.**

