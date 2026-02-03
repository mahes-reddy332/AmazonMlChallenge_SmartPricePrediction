#  Amazon ML Challenge 2025  
## The Multi-Modal Pricing Journey  
### From Raw Data to a Top 1% Global Rank

This repository documents my end-to-end journey in building a high-performance multi-modal machine learning pipeline for product price prediction in the Amazon ML Challenge 2025.

By combining computer vision, large language models, and advanced ensemble learning, this project achieved a **Top 1% global ranking** and demonstrated scalable AI system design under real-world constraints.

---

##  Achievement Highlights

🏅 **Global Rank:** #819 (Top 1%)  
📊 **Final Score (SMAPE):** 49.23  
🎯 **Validation Accuracy:** ~80.3% (InceptionResNetV2)  
📦 **Training Samples:** 75,000+ products  

---

## 📖 The Story Behind the Model

### Phase 1: Understanding the Problem

The dataset consisted of product images and rich catalog descriptions.

Early analysis revealed a critical insight:

> Product price is influenced not only by specifications, but also by visual quality, branding, and perceived value.

A purely text-based model would miss this “visual signal.”  
A purely vision-based model would ignore structured metadata.

So the solution had to be **multi-modal**.

---

### Phase 2: Decoding Product Descriptions (LLM-Powered NLP)

Traditional NLP methods failed to capture subtle semantic differences like:

- “Premium” vs “Budget”  
- “Pack of 3” vs “Single Unit”  
- Flavor and variant descriptions  

#### Approach

- Deployed **Mistral-7B-v0.3** for semantic enrichment  
- Built a hybrid feature pipeline:
  - Regex-based extraction (weights, quantities, pack size)
  - LLM-generated structured JSON features
  - Transformer embeddings

#### Output



text_train_emb.npy


A dense semantic representation of the entire product catalog.

---

### Phase 3: Seeing the Product (Computer Vision)

Images revealed product quality, packaging, and branding cues.

#### Benchmarking

Tested multiple architectures:

- VGG16  
- Xception  
- ResNet  
- EfficientNet  
- InceptionResNetV2  

#### Best Performers

| Model              | Accuracy |
|--------------------|----------|
| InceptionResNetV2  | 80.3%    |
| EfficientNetB2     | 80.1%    |

#### Optimization

To avoid Colab RAM crashes:

- Precomputed embeddings
- Saved to disk


img_train_emb.npy


This enabled fast experimentation without recomputation.

---

### Phase 4: The Fusion Challenge

Combining:

- 768-dim text vectors  
- 2048-dim image vectors  

Created extremely large feature matrices.

#### Key Problems

❌ Memory overflow  
❌ Sample misalignment  
❌ Slow training  

#### Solutions

✅ Chunked training (15k rows per batch)  
✅ Strict sample_id alignment  
✅ Zero-vector padding for missing data  
✅ Incremental scaling  

Final fused representation:


X_train_scaled.npy


This phase transformed a fragile pipeline into a stable system.

---

### Phase 5: Modeling & Optimization

With multi-modal features ready, focus shifted to learning algorithms.

#### Ensemble Benchmarking

- XGBoost (GPU)  
- LightGBM  
- CatBoost  

#### Hyperparameter Tuning

Used **Optuna** for Bayesian optimization:

- Learning rate
- Depth
- Regularization
- Subsampling

15+ automated trials per model.

<img width="1425" height="1035" alt="Screenshot 2026-02-04 024743" src="https://github.com/user-attachments/assets/131a867b-27a8-45c0-9ffa-952b9a84f919" />
<img width="945" height="843" alt="Screenshot 2026-02-04 021504" src="https://github.com/user-attachments/assets/a73c5cb6-b2a1-41ed-be33-ccbfc2182444" />
<img width="1149" height="576" alt="Screenshot 2026-02-04 021110" src="https://github.com/user-attachments/assets/8e5ee9b2-0505-4c5b-8575-79e1c8b00d1f" />
<img width="956" height="895" alt="Screenshot 2026-02-04 015802" src="https://github.com/user-attachments/assets/a1075065-e347-4200-9a3a-8e53209ece87" />


#### Objective

Optimized specifically for:


SMAPE (Symmetric Mean Absolute Percentage Error)


Final local OOF SMAPE:

> 57.22%

Before leaderboard submission.

---

##  System Architecture
Text Data ──► LLM + Regex ──► Text Embeddings
Image Data ─► CNN Models ───► Image Embeddings
↓
Feature Fusion & Scaling
↓
GBM Ensemble (XGB + LGBM + CB)
↓
Price Prediction




---

## 🛠️ Technology Stack

| Layer         | Tools & Frameworks |
|---------------|--------------------|
| NLP / LLM     | Mistral-7B, Transformers |
| Vision        | InceptionResNetV2, EfficientNetB2 |
| ML Models     | XGBoost (GPU), LightGBM, CatBoost |
| Optimization  | Optuna |
| Data          | NumPy, Pandas |
| Infra         | Google Colab (A100/RTX), AWS API |

---

## 📂 Repository Structure


├── FeatureExtraction.ipynb # LLM + Regex pipeline
├── FinalMl2.ipynb # GPU XGBoost training
├── FinalPhaseAmazonML.ipynb # Fusion + Optuna + Submission
├── dataset/ # Raw catalog data
├── *.npy # Precomputed embeddings




---

## 📈 Key Results

✅ Integrated text, vision, and rule-based features  
✅ Built memory-safe multi-modal pipeline  
✅ Ranked Top 1% globally  
✅ Achieved strong generalization  
✅ Enabled fast experimentation  

---

## ⚙️ How to Run

### 1. Clone Repository
```bash
git clone https://github.com/mahes-reddy332/Amazon-ML-Challenge-2025.git
cd Amazon-ML-Challenge-2025
```


2. Install Dependencies
```bash
git clone https://github.com/mahes-reddy332/Amazon-ML-Challenge-2025.git
cd Amazon-ML-Challenge-2025
```
3. Feature Extraction
```bash
jupyter notebook FeatureExtraction.ipynb
```

4. Train Models
```bash
jupyter notebook FinalMl2.ipynb
```
5. Final Fusion & Submission
```bash
jupyter notebook FinalPhaseAmazonML.ipynb
```

 Impact & Learnings

This project strengthened my skills in:

✔ Multi-modal ML system design
✔ Large-scale feature engineering
✔ LLM integration
✔ GPU optimization
✔ Memory management
✔ Competition strategy

It reflects my ability to convert raw data into production-ready AI pipelines.


Future Work

Transformer-based multi-modal fusion

End-to-end deep fusion networks

Automated feature selection

Real-time inference pipeline

Model explainability (SHAP)
