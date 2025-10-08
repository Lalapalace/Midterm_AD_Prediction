# Machine Learning-Based Detection of Alzheimer’s Disease from Handwriting Features  

## 📌 Introduction  
Alzheimer’s Disease (AD) is a progressive disorder that impacts memory, cognition, and motor function. Handwriting analysis has emerged as a promising biomarker since features like pressure, speed, and tremor capture both cognitive and motor impairments. However, real clinical datasets are often small, limiting robust model training.  
This project combines **synthetic data generation** and **machine learning (ML)** to improve AD detection from handwriting features.  

## 📌 Project Aim  
This project explores the use of **machine learning models** to detect Alzheimer’s Disease (AD) using handwriting features as non-invasive biomarkers. The study leverages the **DARWIN dataset**, applies **synthetic data augmentation** via Tabular Variational Autoencoder (TVAE), and compares 13 machine learning algorithms to evaluate predictive performance.  

## ⚙️ Methodology  
### Dataset and Augmentation  
- **Dataset**: DARWIN dataset (174 participants, 452 handwriting features).  
- **Synthetic Data**: Generated 700 samples using **Tabular Variational Autoencoder (TVAE)**.  
- **Final Dataset**: 874 samples (real + synthetic).  
- **Noise Simulation**: Artificial missing values (~1%) added to simulate clinical data variability.  

### Preprocessing  
- Missing values handled with **median imputation**.  
- Features standardized with **StandardScaler**.  
- Target variable encoded: **Healthy (H)** vs **AD-positive (P)**.  

### Model Training  
- Evaluated **13 ML models**:  
  Logistic Regression, LDA, Naïve Bayes, Decision Tree, Random Forest, ExtraTrees, Gradient Boosting, KNN, SVM, MLP, XGBoost, LightGBM.  
- **Hyperparameter tuning** with `RandomizedSearchCV` and **5-fold cross-validation**.  
- Metrics: **Macro F1-score** and **Recall** (prioritized to reduce false negatives).  

## 📊 Results  
- **Top-performing models (F1 > 0.85):**  
  - Logistic Regression: 0.879  
  - XGBoost: 0.878  
  - MLP: 0.876  
  - LightGBM: 0.862  
  - SVM: 0.849  
- **Tree-based ensembles** (Random Forest, ExtraTrees, Gradient Boosting): ~0.83–0.84  
- **Lowest performance**: Decision Tree (0.738)  
- **Key Insight**: Simpler models like Logistic Regression and LDA achieved high recall, consistent with related AD handwriting studies.  

## 📚 References  
- F. Fontanella, *DARWIN Dataset*, UCI Machine Learning Repository, 2022. [Link](https://archive.ics.uci.edu/dataset/732/darwin)  
- D. P. Kingma and M. Welling, *Auto-Encoding Variational Bayes*, arXiv:1312.6114, 2013. [Link](https://arxiv.org/abs/1312.6114)  
- J. Yuan et al., *Handwriting markers for the onset of Alzheimer’s disease*, Frontiers in Aging Neuroscience, 2023. [Link](https://www.frontiersin.org/articles/10.3389/fnagi.2023.1117250/full)  
- H. Qi et al., *A study of auxiliary screening for Alzheimer’s disease based on handwriting characteristics*, Frontiers in Aging Neuroscience, 2023.  

## 🖥️ How to Run the Code  

## 📦 Dependencies  

Install the required dependencies using pip:  

```bash
pip install pandas numpy matplotlib seaborn scikit-learn sdv xgboost lightgbm


## 📦 Dependencies  

```bash
python==3.10
pandas
numpy
matplotlib
seaborn
scikit-learn
sdv
xgboost
lightgbm
