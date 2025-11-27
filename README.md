# 🧠 Extreme Multi-label Learning with Explainability and Adversarial Evaluation using MACH

## 📌 Project Overview

This project explores **Extreme Multi-label Learning (XML)** using the **MACH (Merged-Averaged Classifiers via Hashing)** algorithm, and enhances it by incorporating:

- ✅ **Model Explainability** using Captum
- ✅ **Adversarial Robustness Evaluation** using FGSM

We preserve the original scalability and performance of MACH while making it more **interpretable** and **robust** — addressing key aspects of **dependable machine learning**.

---

## 🔍 Background

### What is XML?
In XML, a sample may be associated with **multiple relevant labels** drawn from a **very large set** (e.g., up to hundreds of thousands). This makes conventional classification methods infeasible due to:

- Memory limitations
- Training/inference time
- Poor generalization

### What is MACH?
**MACH** solves this by using **hash-based label compression**:
- It hashes the large label space into smaller buckets using `R` hash functions.
- Trains a classifier for each hashed label space (with `B` buckets).
- At inference time, it **averages** the outputs across all classifiers to recover scores for the original label set.

> ✅ MACH is fast, scalable, and memory-efficient — ideal for extreme-scale problems.

---

## 🧪 Our Modifications

### 1. 🔬 Model Explainability
We integrated **Captum**, a PyTorch library for model interpretability. Specifically:

- Used **Integrated Gradients** to identify which input features contribute most to specific label predictions.
- Enabled per-sample introspection for multi-label outputs.

This allows users to **trust and understand MACH’s predictions** — especially critical in high-stakes domains (healthcare, legal, etc.).

### 2. 🛡️ Adversarial Evaluation
We performed **adversarial robustness testing** using the **Fast Gradient Sign Method (FGSM)**:

- Applied small, targeted perturbations to test inputs.
- Measured the drop in performance under adversarial settings.
- Evaluated using standard XML metrics (Precision@k, NDCG@k).

> ⚠️ We did not apply adversarial training to preserve MACH's performance.

---

## 📊 Results

### 🔹 Evaluation Metrics
We report **Precision@k** and **NDCG@k** for both clean and adversarial inputs:
#### 📉 Chart: Precision@k and NDCG@k (Clean vs Adversarial)
![Performance Drop](./image.png)

> 🔍 Even simple FGSM attacks noticeably reduce ranking quality, showing the importance of robustness evaluation in XML.

---

## 🛠️ Installation & Setup

### 🧾 Requirements
- Python 3.8+
- PyTorch
- Captum
- NumPy
- Matplotlib
- scikit-learn

### 📦 Install dependencies
```bash
pip install -r requirements.txt


run training
python train.py --dataset wiki10 --hash_functions 3 --buckets 100
📈 Run Explainability (Captum)
python explain.py --model_path saved_model.pth --sample_id 42
Run Adversarial Evaluation
python adversarial_eval.py --eps 0.01 --attack fgsm

video

