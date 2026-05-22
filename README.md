# 📊 Predict, Explain, and Act: An End-to-End Explainable AI System for Profit-Driven Customer Retention

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-red.svg)](https://shap.readthedocs.io/)
[![Conference](https://img.shields.io/badge/Paper-IEEE%20ICISCT%202026-blueviolet.svg)](#)

> **Official Code Repository for the research paper:** *"Predict, Explain, and Act: An End-to-End Explainable AI System for Profit-Driven Customer Retention"* accepted at the 4th International Conference on Information Science and Communication Technology (ICISCT) 2026.

![Project Interface](image-3.png) ## 📋 Table of Contents
1. [Project Overview & Research Gap](#-project-overview--research-gap)
2. [Key Innovations](#-key-innovations)
3. [Dataset Description](#-dataset-description)
4. [Methodology & Architecture](#-methodology--architecture)
5. [Model Performance & Statistical Validation](#-model-performance--statistical-validation)
6. [Explainability (SHAP) & Recommendations](#-explainability-shap--recommendations)
7. [Profit-Driven ROI Analysis](#-profit-driven-roi-analysis)
8. [Installation & Usage](#-installation--usage)
9. [Citation](#-citation)

---

## 🎯 Project Overview & Research Gap

Customer churn drains telecommunication revenues at an alarming rate. While modern machine learning models achieve high accuracy in predicting churn, they fundamentally suffer from a **"black-box"** problem. They identify *who* will leave, but fail to explain *why*, leaving business managers without actionable retention strategies.

**This project bridges the gap between raw predictive analytics and prescriptive business value.** It provides a fully integrated desktop application that:
1. **Predicts** churn using statistically validated ensemble models.
2. **Explains** the exact features driving the prediction using game-theoretic SHAP values.
3. **Acts** by transforming those explanations into targeted, profit-maximizing retention strategies.

---

## 🚀 Key Innovations

This repository implements five distinct technical pipelines that set it apart from standard churn prediction projects:

* **Leakage-Proof SMOTE:** Synthetic Minority Oversampling Technique (SMOTE) is strictly isolated *within* the 5-fold cross-validation folds to prevent data leakage and artificial accuracy inflation.
* **Statistical Model Selection:** Performance superiority is proven not by chance, but via the **Wilcoxon Signed-Rank Test**.
* **Global & Local Explainability:** Utilizes TreeSHAP to decompose complex XGBoost decisions into human-readable insights.
* **Two-Tier Recommendation Engine:** Automatically triggers specific business interventions based on SHAP feature impacts.
* **Financial ROI Calculator:** Employs a specific cost-benefit matrix to quantify the exact monetary value of the AI deployment.

---

## 📊 Dataset Description

The system utilizes the **IBM Telco Customer Churn** dataset (7,043 records, 19 predictors). 

| Category | Features |
| :--- | :--- |
| **Demographics** | Gender, SeniorCitizen, Partner, Dependents |
| **Account** | Tenure, Contract, PaperlessBilling, PaymentMethod |
| **Services** | PhoneService, InternetService, OnlineSecurity, TechSupport, etc. |
| **Financials** | MonthlyCharges, TotalCharges |
| **Target Variable** | Churn (Yes = 1, No = 0) *(Class Imbalance: 26.5% Churners)* |

---

## ⚙️ Methodology & Architecture

The system pipeline is designed to be statistically sound and highly scalable:

1. **Preprocessing:** Deterministic missing value imputation, Label Encoding for tree efficiency, and Standard Scaling.
2. **Cross-Validation:** 5-Fold Stratified CV ensures robust evaluation across data splits.
3. **Algorithms Compared:** XGBoost, Gradient Boosting, Random Forest, Logistic Regression, and Gaussian Naive Bayes.
4. **Integration:** A modular Python backend integrated with a Tkinter frontend for seamless user interaction.

---

## 📈 Model Performance & Statistical Validation

**XGBoost** was selected as the optimal algorithm, prioritizing **Recall** to maximize the identification of at-risk customers, aligning with business financial goals.

| Model | Accuracy | Precision | Recall | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **68.4%** | **45.2%** | **90.6%** | **84.31%** |
| Gradient Boosting | 77.3% | 55.6% | 71.4% | 84.15% |
| Random Forest | 75.4% | 52.4% | 77.8% | 84.17% |

*Note: The precision-recall trade-off was deliberate. XGBoost's aggressive classification of churners yields the highest financial return based on the cost-benefit matrix.*

**Statistical Validation (Wilcoxon Signed-Rank Test):**
The superiority of XGBoost over Gradient Boosting was statistically validated with a significance level of **p-value = 0.041** (α = 0.05).

---

## 🔍 Explainability (SHAP) & Recommendations

### TreeSHAP Integration
The system calculates local feature impacts $(\phi)$ for every single customer.

![SHAP Summary Plot](image-fcc667.png) ### Action Recommendation Engine
The UI translates SHAP values into a two-tier strategy:

**Tier 1: Risk-Level Actions**
* **HIGH (≥70%):** Immediate attention, personal account manager, 20-30% personalized discounts.
* **MODERATE (40-69%):** Active engagement, feedback surveys, limited-time offers.
* **LOW (<40%):** Standard loyalty programs.

**Tier 2: Factor-Specific Interventions (Examples)**
* *If SHAP flags Month-to-Month Contract:* System recommends an annual contract transition with a 15% loyalty discount.
* *If SHAP flags No Tech Support:* System assigns a dedicated technical support representative.

---

## 💰 Profit-Driven ROI Analysis

The true value of this AI system is measured in dollars. Using a cost framework where retaining a churner saves customer lifetime value (CLV = $450) and a false positive costs a campaign fee ($50):

$$ROI = \sum CLV_{saved} - \sum Costs_{campaign}$$

By successfully identifying and targeting the right customers, the XGBoost-powered framework generates an estimated net savings of **$132,000**, proving its economic feasibility.

---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* pip

### Setup Instructions
```bash
# Clone the repository
git clone [https://github.com/SamamaKarim092/Predict-Explain-and-Act-An-End-to-End-Explainable-AI-System-for-Profit-Driven-Customer-Retention.git](https://github.com/SamamaKarim092/Predict-Explain-and-Act-An-End-to-End-Explainable-AI-System-for-Profit-Driven-Customer-Retention.git)
cd Predict-Explain-and-Act

# Install required dependencies
pip install -r requirements.txt

