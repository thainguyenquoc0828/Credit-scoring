# Credit Scoring with WOE, LightGBM & SHAP Explainability

## 📑 Table of Contents

* [📌 Overview](#-overview)
* [📂 Dataset](#-dataset)
* [🔄 Workflow](#-workflow)
* [📊 Model Evaluation](#-model-evaluation)
* [📤 Kaggle Submission](#-kaggle-submission)
* [🔍 Interpretability & Insights](#-interpretability--insights)

  * [Individual Predictions](#individual-predictions)
  * [Global Feature Importance](#global-feature-importance)
* [📎 References](#-references)

---

## 📌 Overview

This project implements a complete **credit risk prediction pipeline** using the Kaggle *Give Me Some Credit* dataset.

The objective is to predict whether a borrower will experience serious financial distress within the next two years. The notebook combines leakage-safe preprocessing, **Weight of Evidence (WOE)** transformation, hyperparameter optimization, gradient boosting models, and **SHAP-based explainability**.

The final model selected for submission is **LightGBM**, which achieved the best validation AUC among the evaluated models.

---

## 📂 Dataset

* **Source**: Kaggle’s *Give Me Some Credit* dataset
* **Task**: Binary classification
* **Target variable**: `SeriousDlqin2yrs`

  * `1`: borrower experienced serious financial distress within two years
  * `0`: borrower did not experience serious financial distress
* **Features**: Financial and demographic variables, including:

  * Revolving utilization of unsecured credit lines
  * Age
  * Debt ratio
  * Monthly income
  * Number of open credit lines and loans
  * Delinquency history
  * Number of dependents

---

## 🔄 Workflow

### 1. Leakage-Safe Data Splitting

The dataset is split into training and validation sets before any preprocessing step:

```python
train_test_split(..., stratify=y)
```

This ensures that all preprocessing statistics are learned only from the training set and then applied to the validation and test sets.

---

### 2. Data Preprocessing

The preprocessing pipeline includes:

* Removing features with more than **30% missing values**, based only on the training set
* Capping outliers at the **99th percentile**, using thresholds calculated from the training set
* Median imputation for missing values
* Removing highly collinear variables using **VIF > 10**
* Applying the same fitted preprocessing steps to validation and test data

In the current experiment:

* No features were removed due to missing rate > 30%
* No features were removed due to VIF > 10

---

### 3. WOE Transformation

After preprocessing, the features are transformed using **Weight of Evidence (WOE)** binning with `scorecardpy`.

WOE transformation is commonly used in credit scoring because it:

* Converts raw numerical variables into risk-oriented representations
* Improves interpretability
* Works well with traditional scorecard models such as Logistic Regression
* Provides stable transformed features for tree-based models such as XGBoost and LightGBM

The WOE bins are fitted only on the training set and then applied to the validation and test sets.

---

### 4. Model Training and Hyperparameter Optimization

Three models are trained and compared:

1. **Logistic Regression**
2. **XGBoost**
3. **LightGBM**

Hyperparameter optimization is performed using **Hyperopt** with Tree-structured Parzen Estimator search.

For XGBoost and LightGBM, early stopping is used with validation AUC as the monitoring metric.

Class imbalance is handled using:

* `class_weight="balanced"` for Logistic Regression
* `scale_pos_weight` for XGBoost and LightGBM

---

## 📊 Model Evaluation

The models are evaluated on the validation set using:

* **AUC-ROC**
* **F1-score**
* **Recall**

| Model               |      AUC | F1-score |   Recall |
| ------------------- | -------: | -------: | -------: |
| LightGBM            | 0.863189 | 0.328759 | 0.788529 |
| XGBoost             | 0.862993 | 0.329778 | 0.786534 |
| Logistic Regression | 0.860437 | 0.327624 | 0.787032 |

LightGBM achieved the highest validation AUC and was selected as the final model.

However, the performance gap between LightGBM and XGBoost is very small, indicating that both gradient boosting models perform similarly on this dataset. Logistic Regression also remains competitive after WOE transformation, showing the effectiveness of WOE-based credit scoring features.

---

## 📤 Kaggle Submission

The final Kaggle submission is generated using the selected **LightGBM** model.

The same preprocessing pipeline fitted on the training data is applied to the Kaggle test set:

1. Keep selected columns
2. Cap outliers using training-set 99th percentile values
3. Impute missing values using the training-fitted median imputer
4. Keep VIF-selected variables
5. Apply WOE bins fitted on the training set
6. Predict probabilities using the final LightGBM model

The submission file follows Kaggle’s required format:

```text
Id,Probability
1,0.586418
2,0.426647
3,0.139034
...
```

The generated submission file contains:

```text
101503 rows × 2 columns
```

---

## 🔍 Interpretability & Insights

Model interpretability is performed using **SHAP values** on the final LightGBM model.

Since the model is trained on WOE-transformed features, the SHAP explanations should be interpreted as the contribution of each **WOE feature**, not the original raw feature value.

For example:

```text
RevolvingUtilizationOfUnsecuredLines_woe
```

represents the WOE-transformed version of:

```text
RevolvingUtilizationOfUnsecuredLines
```

---

### Individual Predictions

#### Client 10683

<p align="center">
  <img src="Client 10683.png" width="800" alt="Client 10683 SHAP Force Plot">
</p>

For this client, features such as `RevolvingUtilizationOfUnsecuredLines_woe` and `age_woe` reduce the predicted risk, while `DebtRatio_woe` has a minor positive contribution. The final prediction is below the base value, indicating a relatively low probability of serious financial distress.

---

#### Client 8652

<p align="center">
  <img src="Client 8652.png" width="800" alt="Client 8652 SHAP Force Plot">
</p>

For this client, features such as `age_woe`, `DebtRatio_woe`, and `RevolvingUtilizationOfUnsecuredLines_woe` contribute to increasing the predicted risk. The final prediction is above the base value, suggesting a higher-than-average probability of serious financial distress.

---

### Global Feature Importance

#### SHAP Summary Plot

<p align="center">
  <img src="SHAP 1.png" width="800" alt="SHAP Summary Plot">
</p>

The SHAP summary plot shows that `RevolvingUtilizationOfUnsecuredLines_woe` is the most influential feature in the LightGBM model.

Other important predictors include:

* `NumberOfTime30-59DaysPastDueNotWorse_woe`
* `NumberOfTimes90DaysLate_woe`
* `DebtRatio_woe`
* `age_woe`

In general, delinquency-related variables and high credit utilization contribute strongly to increasing predicted default risk, while age-related effects tend to reduce risk for certain WOE bins.

---

## 📎 References

* **Dataset**: [Kaggle - Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
* **scorecardpy Documentation**: https://github.com/ShichenXie/scorecardpy
* **LightGBM Documentation**: https://lightgbm.readthedocs.io/
* **XGBoost Documentation**: https://xgboost.readthedocs.io/
* **SHAP Documentation**: https://shap.readthedocs.io/
* **Hyperopt Documentation**: http://hyperopt.github.io/hyperopt/
