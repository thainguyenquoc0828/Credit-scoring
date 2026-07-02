# Credit Scoring with WOE & Logistic Regression

## 📑 Table of Contents

* [📌 Overview](#-overview)
* [📂 Dataset](#-dataset)
* [🔎 Exploratory Data Analysis](#-exploratory-data-analysis)
* [🔄 Preprocessing Pipeline](#-preprocessing-pipeline)
* [🤖 Model Training](#-model-training)
* [📊 Model Evaluation](#-model-evaluation)
* [📤 Kaggle Submission](#-kaggle-submission)
* [📎 References](#-references)

---

## 📌 Overview

This project builds a **credit scoring pipeline** using the Kaggle *Give Me Some Credit* dataset.

The objective is to predict whether a borrower will experience serious financial distress within the next two years. The project follows a traditional credit risk modeling workflow, including exploratory data analysis, leakage-safe preprocessing, **Weight of Evidence (WOE)** transformation, feature selection using **Information Value** and **VIF**, and final model training with **Logistic Regression**.

The final model is designed to be interpretable and suitable for credit scoring applications.

---

## 📂 Dataset

* **Source**: Kaggle’s *Give Me Some Credit* dataset

* **Task**: Binary classification

* **Target variable**: `SeriousDlqin2yrs`

  * `1`: borrower experienced serious financial distress within two years
  * `0`: borrower did not experience serious financial distress

* **Main features**:

  * `RevolvingUtilizationOfUnsecuredLines`
  * `age`
  * `NumberOfTime30-59DaysPastDueNotWorse`
  * `DebtRatio`
  * `MonthlyIncome`
  * `NumberOfOpenCreditLinesAndLoans`
  * `NumberOfTimes90DaysLate`
  * `NumberRealEstateLoansOrLines`
  * `NumberOfTime60-89DaysPastDueNotWorse`
  * `NumberOfDependents`

---

## 🔎 Exploratory Data Analysis

The notebook `exploratory-data-analysis.ipynb` investigates data quality, feature distributions, outliers, missing values, target imbalance, and relationships between predictors and the target.

Main findings:

* The dataset is highly imbalanced, with only about **6.7%** of borrowers having target value `1`.

* Missing values appear mainly in:

  * `MonthlyIncome`
  * `NumberOfDependents`

* Several variables contain abnormal or extreme values:

  * `age = 0`
  * very large `DebtRatio`
  * very large `RevolvingUtilizationOfUnsecuredLines`
  * delinquency variables with abnormal values such as `96` and `98`

* Most numerical variables are right-skewed.

* `age` has an approximately normal distribution compared with other variables.

* `RevolvingUtilizationOfUnsecuredLines` and `age` show clear separation between target classes.

* Delinquency-related variables are strongly related to default risk.

* The delinquency variables are highly correlated with each other, which may cause multicollinearity in Logistic Regression.

---

## 🔄 Preprocessing Pipeline

The preprocessing logic is implemented in `preprocessing.py`.

The pipeline contains three custom transformers:

```python
preprocess_pipeline = Pipeline([
    ("cleaner", CreditDataCleaner()),
    ("woe_iv", WOEIVTransformer(
        target="SeriousDlqin2yrs",
        iv_threshold=0.02,
        bin_num_limit=5
    )),
    ("vif", VIFSelector(threshold=5.0)),
])
```

### 1. CreditDataCleaner

This transformer handles abnormal values and missing values.

Steps:

* Replace abnormal values `96` and `98` in delinquency variables with the median calculated from the training set.
* Impute missing `NumberOfDependents` using the training-set median.
* Replace invalid `age = 0` with the median valid age from the training set.

Affected delinquency variables:

```python
NumberOfTime30-59DaysPastDueNotWorse
NumberOfTimes90DaysLate
NumberOfTime60-89DaysPastDueNotWorse
```

---

### 2. WOEIVTransformer

This transformer applies **Weight of Evidence** binning using `scorecardpy`.

Main steps:

* Fit WOE bins on the training data only.
* Calculate **Information Value** for each variable.
* Keep variables with:

```python
IV >= 0.02
```

* Transform raw features into WOE-transformed features.

Example transformed feature name:

```text
age_woe
```

WOE transformation is useful in credit scoring because it creates monotonic, risk-oriented feature representations and improves interpretability for Logistic Regression.

---

### 3. VIFSelector

This transformer removes multicollinearity among WOE-transformed features.

Main steps:

* Calculate **Variance Inflation Factor** for all transformed variables.
* Iteratively remove the variable with the highest VIF.
* Stop when all remaining variables satisfy:

```python
VIF <= 5.0
```

---

## 🤖 Model Training

The model training workflow is implemented in `training.ipynb`.

The training process includes:

1. Load training data from:

```python
data/cs-training.csv
```

2. Remove duplicate rows.

3. Split the data into training and validation sets:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

4. Build a full modeling pipeline:

```python
full_pipeline = Pipeline([
    ("preprocess", preprocess_pipeline),
    ("clf", LogisticRegression(max_iter=1000))
])
```

5. Tune Logistic Regression hyperparameters using 5-fold Stratified Cross-Validation.

Grid search space:

```python
param_grid = {
    "clf__C": [0.01, 0.1, 1.0, 10.0],
    "clf__penalty": ["l1"],
    "clf__solver": ["liblinear"],
}
```

6. Select the best model using validation **ROC-AUC**.

---

## 📊 Model Evaluation

The main evaluation metric is:

```text
ROC-AUC
```

After hyperparameter tuning, the best Logistic Regression pipeline is evaluated on the validation set:

```python
valid_auc = roc_auc_score(y_valid, y_pred_proba)
```

The project also fits a final `statsmodels.Logit` model on the transformed features to generate a statistical summary table, including:

* coefficient β
* standard error
* z-value
* p-value
* odds ratio
* 95% confidence interval

Before fitting the `statsmodels` model, two WOE features are manually removed:

```python
drop_cols = ["MonthlyIncome_woe", "NumberOfDependents_woe"]
```

The final statistics table is used to inspect the direction, magnitude, and significance of each predictor.

---

## 📤 Kaggle Submission

The final submission is generated in `training.ipynb`.

The test data is loaded from:

```python
data/cs-test.csv
```

The fitted preprocessing pipeline is applied to the test set, then the final Logistic Regression model predicts default probabilities.

Submission format:

```text
Id,Probability
1,0.123456
2,0.067891
3,0.245678
...
```

The output file is saved as:

```python
submission.csv
```

---

## 📁 Project Structure

```text
.
├── data/
│   ├── cs-training.csv
│   └── cs-test.csv
├── exploratory-data-analysis.ipynb
├── preprocessing.py
├── training.ipynb
├── submission.csv
└── README.md
```

---

## ⚙️ Installation

Install the required libraries:

```bash
pip install numpy pandas scikit-learn statsmodels scorecardpy scipy matplotlib
```

---

## ▶️ How to Run

Run exploratory analysis:

```bash
jupyter notebook exploratory-data-analysis.ipynb
```

Run preprocessing, model training, evaluation, and submission generation:

```bash
jupyter notebook training.ipynb
```

Make sure the input files are placed in the `data/` directory:

```text
data/cs-training.csv
data/cs-test.csv
```

---

## 📎 References

* **Dataset**: [Kaggle - Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
* **scorecardpy Documentation**: https://github.com/ShichenXie/scorecardpy
* **scikit-learn Documentation**: https://scikit-learn.org/
* **statsmodels Documentation**: https://www.statsmodels.org/
