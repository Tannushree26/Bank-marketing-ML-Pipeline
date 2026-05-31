# DG4AML – Applied Machine Learning: Bank Marketing Subscription Prediction

An end-to-end machine learning pipeline built on the **UCI Bank Marketing dataset** to predict whether a customer subscribes to a term deposit following a marketing call.

Submitted as individual coursework for **DG4AML Applied Machine Learning** at Aston University (2025/26).

---

## Project Overview

The core task is a **binary classification** problem, but the pipeline integrates all four ML paradigms covered on the module — EDA & preprocessing, clustering, regression, and classification — in a principled, leakage-free sequence.

Two engineered features are derived **solely from pre-call information** to avoid data leakage:
1. **Customer segment label** — produced by unsupervised clustering
2. **Predicted call duration** — estimated by a regression model trained on pre-call features and cluster labels

The true call duration is deliberately excluded from the classification model.

---

## Pipeline Structure

### 1. Exploratory Data Analysis & Preprocessing
- Dataset overview, missing value analysis, unknown value handling
- Target class imbalance identified (~88% "no", ~11.7% "yes")
- Outlier detection via IQR; capping applied to `balance`, `campaign`, `previous`
- Two distinct preprocessing approaches compared:
  - **Approach 1:** StandardScaler + One-Hot Encoding
  - **Approach 2 (selected):** MinMax Scaling + Frequency Encoding — chosen for its compact, distance-friendly feature space, well-suited to clustering

### 2. Clustering (Unsupervised)
- **K-Means** vs **BIRCH** evaluated across k = 2–10
- Metrics: Silhouette Score, Davies-Bouldin, Calinski-Harabasz, stability (pairwise ARI)
- **K-Means with k=4 selected** — provided more granular, marketing-actionable segments despite slightly lower silhouette than BIRCH (k=2)
- Four interpretable segments identified: General Base, Repeated Contact, High Debt, High-Value Previously Engaged

### 3. Regression
- Target: `duration` (pre-call prediction)
- **Linear Regression** (baseline) vs **Random Forest Regressor**
- 5-fold cross-validation on log-transformed target
- **Random Forest selected** — lower CV RMSE (0.8929 vs 0.9063), higher R² (0.0634 vs 0.0353)

### 4. Classification
- Features include cluster label and predicted call duration as engineered inputs
- **Logistic Regression** (with `class_weight='balanced'`) vs **Random Forest Classifier**
- GridSearchCV tuning with StratifiedKFold (5-fold), optimising PR-AUC
- **Random Forest selected** — ROC-AUC 0.79, PR-AUC 0.40, better precision-recall trade-off for minority class

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| pandas, numpy | Data manipulation |
| scikit-learn | ML models, preprocessing, evaluation |
| matplotlib, seaborn | Visualisation |
| Jupyter Notebook | Reproducible pipeline |

---

## Key Design Decisions

- `duration` excluded from classification features to prevent **target leakage**
- `poutcome` dropped as it is a post-call variable (leakage risk)
- Cluster labels and predicted duration added as **engineered features** flowing through the pipeline
- Each pipeline stage is independently scoreable — later stages do not depend on earlier stages working perfectly
- Written justification (~1,500 words) embedded directly in notebook cells at point of relevance

---

## References

- Breiman, L., 2001. *Random Forests.* Machine Learning, 45(1), pp.5–32.
- Han, J., Kamber, M. and Pei, J., 2011. *Data Mining: Concepts and Techniques.* 3rd ed. Morgan Kaufmann.
- He, H. and Garcia, E.A., 2009. Learning from imbalanced data. *IEEE TKDE*, 21(9), pp.1263–1284.
- James, G. et al., 2021. *An Introduction to Statistical Learning.* 2nd ed. Springer.
- MacQueen, J., 1967. Some methods for classification and analysis of multivariate observations. *5th Berkeley Symposium.*
- Rousseeuw, P.J., 1987. Silhouettes: a graphical aid to cluster analysis. *JCAM*, 20, pp.53–65.
- Tan, P.N. et al., 2019. *Introduction to Data Mining.* 2nd ed. Pearson.

---

## Notes

- The dataset (`bank-full-updated.csv`) is not included in this repository (sourced from Blackboard/UCI).
- AI tools were used only for study support (idea generation, summarising notes) in line with Aston University's assessment regulations. All code and written justifications are the author's own work.
