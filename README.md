# Banking Customer Churn Prediction: Statistical Inference & Ensemble Optimization

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-success)
![Optuna](https://img.shields.io/badge/Optuna-Bayesian_TPE-blueviolet)
![Status](https://img.shields.io/badge/Study-Undergraduate_Thesis-darkred)

## Abstract
Customer attrition (churn) is a primary source of revenue loss in retail banking and credit card portfolio management. Developed as an **Undergraduate Graduation Thesis in Statistics at Marmara University**, this repository presents an end-to-end statistical and machine learning framework to model, explain, and predict credit card customer churn across $N = 10,127$ accounts. 

Combining non-parametric outlier winsorization, domain-specific ratio engineering, parametric/non-parametric hypothesis testing (**One-Way ANOVA $F$-test** and **Pearson's $\chi^2$ test**), and **Bayesian hyperparameter optimization via Optuna (Tree-structured Parzen Estimator)**, the proposed **LightGBM** architecture achieves a **Cross-Validated ROC-AUC of 0.9936** and **97.0% Hold-Out Test Accuracy**.

---

## 1. Dataset & Class Distribution
The dataset (`BankChurners`) comprises **10,127 credit card holders** characterized by 19 demographic, account-level, and transactional predictors:
* **Retained Customers (`Attrition_Flag = 1`):** $8,500$ ($83.93\%$)
* **Attrited / Churned Customers (`Attrition_Flag = 0`):** $1,627$ ($16.07\%$)

Given the $83.9\% : 16.1\%$ class imbalance, accuracy alone is insufficient. Model selection and hyperparameter tuning prioritize **ROC-AUC**, **Macro-Averaged F1-Score**, and **Stratified $K$-Fold Cross-Validation**.

---

## 2. Statistical Methodology & Preprocessing Pipeline

### 2.1 Non-Parametric Outlier Winsorization
Several financial predictors (`Credit_Limit`, `Avg_Open_To_Buy`, and quarterly ratio changes) exhibit strong positive skewness. Rather than discarding observations and inducing sample selection bias, extreme values are winsorized using robust inter-percentile fencing ($Q_{0.05}$ and $Q_{0.95}$):

$$\text{IQR}_{0.05, 0.95} = Q_{0.95} - Q_{0.05}, \quad [\text{Lower}, \text{Upper}] = \left[ Q_{0.05} - 1.5 \times \text{IQR}_{0.05, 0.95}, \; Q_{0.95} + 1.5 \times \text{IQR}_{0.05, 0.95} \right]$$

### 2.2 Domain-Specific Feature Engineering
To capture customer lifecycle velocity and credit utilization intensity, five interaction terms are constructed:
1. **`TenureYears`**: $\text{Months\_on\_book} / 12$
2. **`CreditLimitPerYear`**: $\text{Credit\_Limit} / \text{TenureYears}$
3. **`InactiveRate`**: $\text{Months\_Inactive\_12\_mon} / \text{Months\_on\_book}$
4. **`DependentsAgeRatio`**: $\text{Dependent\_count} / \text{Customer\_Age}$
5. **`BalancePerMonth`**: $\text{Total\_Revolving\_Bal} / \text{Months\_on\_book}$

### 2.3 Statistical Feature Selection ($\chi^2$ & One-Way ANOVA)
* **Numerical Predictors (ANOVA $F$-Test):** Behavioral activity metrics dominate discriminative power, led by `Total_Trans_Ct` ($F = 1,625.24$), `Total_Ct_Chng_Q4_Q1` ($F = 995.36$), `Total_Revolving_Bal` ($F = 753.75$), and `BalancePerMonth` ($F = 656.60$). Static demographic variables (`Customer_Age`, `Months_on_book`) show statistically negligible main effects ($F < 3.0$).
* **Categorical Predictors (Pearson's $\chi^2$):** Service contact frequency (`Contacts_Count_12_mon`, where 6 contacts yields a $100\%$ attrition rate) and inactivity duration (`Months_Inactive_12_mon` $\ge 3$) represent the strongest categorical risk factors.

---

## 3. Model Benchmarking (10-Fold Cross-Validation)
Eight classification algorithms were benchmarked using 10-Fold Cross-Validation across 54 encoded features:

| Model | Test Accuracy | 10-Fold CV Accuracy | 10-Fold CV ROC-AUC | 10-Fold CV Precision | 10-Fold CV Recall | 10-Fold CV F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.9625** | **0.9505** | **0.9753** | **0.9647** | **0.9769** | **0.9707** |
| **XGBoost** | 0.9600 | 0.9499 | 0.9749 | 0.9651 | 0.9758 | 0.9704 |
| **Gradient Boosting** | 0.9571 | 0.9414 | 0.9662 | 0.9572 | 0.9739 | 0.9654 |
| **Random Forest** | 0.9427 | 0.9338 | 0.9553 | 0.9449 | 0.9791 | 0.9614 |
| **CART (Decision Tree)** | 0.9299 | 0.9063 | 0.8470 | 0.9529 | 0.9344 | 0.9431 |
| **Logistic Regression** | 0.9072 | 0.8993 | 0.9230 | 0.9254 | 0.9593 | 0.9405 |
| **Support Vector Machine (SVM)** | 0.8850 | 0.8881 | 0.9079 | 0.8926 | 0.9866 | 0.9369 |
| **K-Nearest Neighbors (KNN)** | 0.8445 | 0.8373 | 0.6704 | 0.8582 | 0.9656 | 0.9087 |

---

## 4. Bayesian Hyperparameter Optimization (Optuna TPE)
Using Optuna's **Tree-structured Parzen Estimator (TPE)** with `MedianPruner` across 100 trials (5-Fold Stratified CV), the joint model and hyperparameter space converged on **LightGBM**:

* **Optimal Hyperparameters:** `n_estimators = 498`, `learning_rate = 0.08563`, `num_leaves = 12`, `random_state = 42`
* **Best Stratified CV ROC-AUC:** **0.9936**

### Hold-Out Test Set Performance ($N_{\text{test}} = 2,026$)
| Class / Metric | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Attrited Customer (`0`)** | 0.92 | 0.88 | 0.90 | 327 |
| **Retained Customer (`1`)** | 0.98 | 0.99 | 0.98 | 1,699 |
| **Macro Average** | **0.95** | **0.93** | **0.94** | **2,026** |
| **Weighted Average / Accuracy** | **0.97** | **0.97** | **0.97** | **2,026** |

---

## 5. Repository Structure & Reproducibility
```text
├── Banking_Churn_Prediction_Thesis.ipynb   # Full statistical analysis, EDA, ANOVA/Chi2 & Optuna TPE pipeline
├── BankChurners1.csv                       # Credit card portfolio dataset (N = 10,127)
├── pipeline_clean.pkl                      # Serialized scikit-learn ColumnTransformer + LightGBM pipeline
└── README.md                               # Project documentation & methodology summary
