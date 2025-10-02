# ❤️ Heart Disease Prediction — Full Machine Learning Pipeline

This project implements a **comprehensive machine learning pipeline** on the **UCI Heart Disease dataset**.  
It includes **data preprocessing, PCA, feature selection, supervised & unsupervised learning, hyperparameter tuning, and deployment** via Streamlit.

---

## 📊 Project Overview

- **Dataset:** [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Goal:** Predict whether a patient has heart disease based on clinical features.
- **Language:** Python  
- **Frameworks:** scikit-learn, pandas, Streamlit  
- **Deployment:** Local web app with Streamlit (Ngrok optional for public link)

---

## 🧠 ML Pipeline Steps

### 1️⃣ Data Preprocessing (`01_data_preprocessing.ipynb`)
- Loaded UCI Heart Disease dataset  
- Cleaned and renamed columns  
- Encoded categorical features (e.g. `sex`, `cp`, `thal`)  
- Handled missing values  
- Scaled numerical features with `StandardScaler`

### 2️⃣ PCA Analysis (`02_pca_analysis.ipynb`)
- Performed PCA to analyze variance  
- Retained 95% variance  
- Visualized PC1 vs PC2 scatter plot

### 3️⃣ Feature Selection (`03_feature_selection.ipynb`)
- Used:
  - **RandomForest feature importance**
  - **Recursive Feature Elimination (RFE)**
  - **Chi-Square test**
- Combined selected features into final dataset

### 4️⃣ Supervised Models (`04_supervised_learning.ipynb`)
- Trained multiple models:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine (SVM)
- Evaluated with Accuracy, Precision, Recall, F1-score, ROC-AUC  
- Saved best model (`RandomForestClassifier`)

### 5️⃣ Unsupervised Learning (`05_unsupervised_learning.ipynb`)
- Applied **K-Means Clustering** and **Hierarchical Clustering**
- Determined optimal `k` using elbow method
- Compared clusters with actual target labels

### 6️⃣ Hyperparameter Tuning (`06_hyperparameter_tuning.ipynb`)
- Tuned `RandomForestClassifier` using **RandomizedSearchCV**
- Selected best hyperparameters
- Saved tuned model: `final_model_tuned.pkl`

---

## 🚀 Deployment (Streamlit App)

### 📁 App location:
