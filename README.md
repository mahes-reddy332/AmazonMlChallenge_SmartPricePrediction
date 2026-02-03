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
