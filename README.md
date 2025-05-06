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

### 🔹 PriorEnh
```bash
python deploy_priorenh.py -f /path/to/image
```


## 📄 Citation

If you use these modules in your research, please cite:

```
@article{Pan2025FLARE,
  title={FLARE: Fixed-Length Dense Descriptor with Fingerprint Enhancement and Alignment for Effective Fingerprint Matching},
  author={Zhiyu Pan and [Other authors]},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025}
}
```

---

## 📬 Contact
For any questions or feedback, feel free to open an issue or contact [Zhiyu Pan](pzy20@mails.tsinghua.edu.cn).




