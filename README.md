Machine Learning-Based Detection of Alzheimer’s Disease from Handwriting Features
Introduction

Alzheimer’s Disease (AD) is a progressive neurodegenerative disorder that affects memory, cognition, and motor skills. Early detection is critical to slowing disease progression and improving patient outcomes. Handwriting analysis has gained attention as a potential biomarker, since kinematic features such as pressure, speed, and tremor reflect both cognitive and motor impairments in AD patients. However, the limited size of clinical handwriting datasets often reduces the reliability of predictive models. To address this challenge, synthetic data generation and machine learning (ML) methods were applied to enhance model performance and generalization.

Methodology
Dataset and Augmentation

The DARWIN dataset was used, containing handwriting samples from 174 participants with 452 spatiotemporal features.

A Tabular Variational Autoencoder (TVAE) was trained to generate 700 synthetic samples. Together with the original data, the extended dataset included 874 instances.

Artificial missing values were introduced to mimic real-world data conditions.

Preprocessing

Median imputation was applied to handle missing values.

All features were standardized using StandardScaler.

The target variable was encoded as binary: Healthy vs AD-positive.

Model Training

Thirteen machine learning models were tested, including Logistic Regression, Linear Discriminant Analysis, Naïve Bayes, Decision Tree, Random Forest, ExtraTrees, Gradient Boosting, Support Vector Machine, K-Nearest Neighbors, Multi-Layer Perceptron, XGBoost, and LightGBM.

Hyperparameters were tuned with RandomizedSearchCV and 5-fold cross-validation.

Macro F1-score and Recall were prioritized, since minimizing false negatives is more important in a clinical context.

Results

Logistic Regression with L1 regularization confirmed that there was no data leakage, and only a few features showed high predictive power.

Best-performing models achieved F1-scores above 0.85:

Logistic Regression: 0.879

XGBoost: 0.878

MLP: 0.876

LightGBM: 0.862

SVM: 0.849

Tree-based ensembles (Random Forest, ExtraTrees, Gradient Boosting) performed consistently in the 0.83–0.84 range.

Decision Tree alone had the lowest score at 0.738, indicating that more complex models are required to capture handwriting variability.

Heatmaps and confusion matrices showed that errors were evenly distributed, with simpler models like Logistic Regression and LDA achieving higher recall, consistent with findings from related AD handwriting studies.

References

F. Fontanella, "DARWIN Dataset," UCI Machine Learning Repository, 2022. Available: https://archive.ics.uci.edu/dataset/732/darwin

D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," arXiv:1312.6114, 2013. Available: https://arxiv.org/abs/1312.6114

J. Yuan et al., "Handwriting markers for the onset of Alzheimer’s disease," Frontiers in Aging Neuroscience, vol. 15, 2023. Available: https://www.frontiersin.org/articles/10.3389/fnagi.2023.1117250/full

H. Qi et al., "A study of auxiliary screening for Alzheimer’s disease based on handwriting characteristics," Frontiers in Aging Neuroscience, vol. 15, 2023.
