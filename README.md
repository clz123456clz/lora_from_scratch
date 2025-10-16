# 🚀 Reproducing LoRA from Scratch

> **A hands-on journey into low-rank adaptation and PyTorch parameterization**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-≥1.13-EE4C2C?logo=pytorch)
![Notebook](https://img.shields.io/badge/Format-Jupyter%20Notebook-9cf?logo=jupyter)

---

## 📘 Overview

This repository documents a small but insightful project on **understanding and implementing LoRA (Low-Rank Adaptation)** from scratch.  
It bridges theory and practice by exploring how **low-rank matrix decomposition** can lead to efficient model fine-tuning with minimal parameter updates.

---

## 🧩 Implemented Notebooks

### 1️⃣ `svd.ipynb`
- Provides an **intuitive demonstration of Singular Value Decomposition (SVD)**.  
- Shows how a low-rank approximation can represent nearly the same information as a full-rank matrix, but with **much fewer parameters**.

### 2️⃣ `lora_from_scratch.ipynb`
- Implements **LoRA applied to a fully connected (Linear) layer** in PyTorch.  
- Demonstrates that fine-tuning only a small number of low-rank parameters can **improve recognition accuracy on the digit ‘9’**, compared to full-parameter training.  
- Includes visualization and comparison of performance between the **base model** and the **LoRA-augmented model**.

---

## 🧠 Key Learnings

- Gained a deeper understanding of **low-rank decomposition** as the mathematical foundation of LoRA.  
- Learned how to use **`torch.nn.utils.parametrize`** to dynamically modify model parameters — enabling **external parameter injection or transformations**  
  without altering the model’s original architecture (`__init__` and `forward`).  
- Experienced how **parameter-efficient fine-tuning (PEFT)** techniques can achieve meaningful improvements with minimal additional compute.

---

## 🛠️ Environment

| Dependency | Version |
|-------------|----------|
| Python | 3.8+ |
| PyTorch | ≥ 1.13 |
| Jupyter | ≥ 6.0 |

You can install dependencies via:

```bash
pip install torch torchvision jupyter