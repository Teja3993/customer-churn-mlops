# 📡 Telco Customer Churn: SLM-Enhanced MLOps Engine

An end-to-end production-grade MLOps project that predicts customer churn using a **Stacking Meta-Model** and **NLP-enhanced feature engineering**.



## 🚀 Business Case & Impact
Retaining customers is 5x cheaper than acquiring new ones. This project identifies high-risk churners by analyzing not just their billing data, but also the **emotional sentiment** of their feedback.

### Key Technical Achievements:
* **SLM Feature Engineering**: Leveraged a quantized **Phi-3-mini** Small Language Model to extract "Primary Complaints" from raw customer text.
* **Stacking Meta-Model**: Achieved a **0.988 PR-AUC** by ensembling CatBoost, LightGBM, and XGBoost with a Logistic Regression meta-classifier.
* **Interpretability**: Used **SHAP** values to prove that Sentiment Score and SLM-extracted complaints are the strongest predictors of churn.

## 📊 Model Performance Highlights
| Metric | Score |
| :--- | :--- |
| **PR-AUC (Primary Metric)** | **0.9880** |
| **Recall (Catching Churners)** | **99.52%** |
| **ROC-AUC** | **0.9943** |



## 🛠️ Project Architecture
1. **`notebooks/`**: Exploratory Data Analysis, Cloud-based training on Kaggle GPUs, and SHAP evaluation.
2. **`models/`**: Serialized `.joblib` artifacts and performance visualizations.
3. **`src/`**: Modular Python scripts for preprocessing and training pipelines.
4. **`app.py`**: A live **Streamlit** dashboard for real-time churn probability calculation.

## 💻 How to Run the App
1. Clone the repository: `git clone https://github.com/Teja3993/customer-churn-mlops.git`
2. Install dependencies: `pip install streamlit joblib textblob xgboost lightgbm catboost scikit-learn`
3. Launch the dashboard: `streamlit run app.py`

---
**Author**: Teja Karri | M.Tech Data Science GITAM Vizag 2025-2027 | BTech EE NIT Rourkela '23