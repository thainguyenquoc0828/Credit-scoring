import numpy as np
import pandas as pd
import scorecardpy as sc

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

# Cấu hình chung
train_data = pd.read_csv("data/cs-training.csv", index_col=0)
test_data = pd.read_csv("data/cs-test.csv", index_col=0)

target = "SeriousDlqin2yrs"
random_state = 42
test_size = 0.2

iv_threshold = 0.02
bin_num_limit = 5
vif_threshold = 5.0

# Loại bỏ dữ liệu trùng lặp và chia train/valid
print("Shape before drop duplicates:", train_data.shape)
train_data = train_data.drop_duplicates()
print("Shape after drop duplicates:", train_data.shape)

X = train_data.drop(columns=[target])
y = train_data[target]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
)

X_test = test_data.drop(columns=[target], errors="ignore").copy()

print("\nTrain target distribution:")
print(y_train.value_counts(normalize=True))

print("\nValid target distribution:")
print(y_valid.value_counts(normalize=True))

class CreditDataCleaner(BaseEstimator, TransformerMixin):
    """
    Transformer xử lý dữ liệu bất thường và missing values.

    Các bước:
    1. Thay giá trị 96, 98 ở các biến trễ nợ bằng median học từ train.
    2. Điền missing NumberOfDependents bằng median học từ train.
    3. Thay age = 0 bằng median age học từ train.
    """

    def __init__(self):
        self.delay_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse",
        ]

    def fit(self, X, y=None):
        X = X.copy()

        self.delay_medians_ = {}

        for col in self.delay_cols:
            if col in X.columns:
                median_value = X.loc[~X[col].isin([96, 98]), col].median()
                self.delay_medians_[col] = median_value

        if "NumberOfDependents" in X.columns:
            self.median_dependents_ = X["NumberOfDependents"].median()
        else:
            self.median_dependents_ = None

        if "age" in X.columns:
            self.median_age_ = X.loc[X["age"] != 0, "age"].median()
        else:
            self.median_age_ = None

        return self

    def transform(self, X):
        X = X.copy()

        for col, median_value in self.delay_medians_.items():
            if col in X.columns:
                X[col] = X[col].replace([96, 98], median_value)

        if "NumberOfDependents" in X.columns and self.median_dependents_ is not None:
            X["NumberOfDependents"] = X["NumberOfDependents"].fillna(
                self.median_dependents_
            )

        if "age" in X.columns and self.median_age_ is not None:
            X.loc[X["age"] == 0, "age"] = self.median_age_

        return X

class WOEIVTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer thực hiện:
    1. WOE binning bằng scorecardpy.
    2. Lọc biến theo Information Value.
    3. Transform dữ liệu raw thành dữ liệu WOE.
    """

    def __init__(self, target, iv_threshold=0.02, bin_num_limit=5):
        self.target = target
        self.iv_threshold = iv_threshold
        self.bin_num_limit = bin_num_limit

    def fit(self, X, y):
        X = X.copy()

        y = pd.Series(y, name=self.target).reset_index(drop=True)

        train_woe_input = pd.concat(
            [
                X.reset_index(drop=True),
                y
            ],
            axis=1
        )

        self.bins_ = sc.woebin(
            train_woe_input,
            y=self.target,
            bin_num_limit=self.bin_num_limit
        )

        self.iv_table_ = []

        for var, bin_df in self.bins_.items():
            total_iv = bin_df["total_iv"].iloc[0]
            self.iv_table_.append({
                "variable": var,
                "iv": total_iv
            })

        self.iv_table_ = pd.DataFrame(self.iv_table_).sort_values(
            "iv",
            ascending=False
        ).reset_index(drop=True)

        self.filtered_bins_ = {
            var: df for var, df in self.bins_.items()
            if df["total_iv"].iloc[0] >= self.iv_threshold
        }

        self.selected_variables_ = list(self.filtered_bins_.keys())

        print("\n========== WOE / IV Selection ==========")
        print(f"Số biến ban đầu: {len(self.bins_)}")
        print(f"Số biến sau khi lọc IV >= {self.iv_threshold}: {len(self.filtered_bins_)}")
        print("Danh sách biến được giữ lại:")
        print(self.selected_variables_)

        print("\nIV table:")
        print(self.iv_table_)

        return self

    def transform(self, X):
        X = X.copy()

        X_woe = sc.woebin_ply(
            X,
            self.filtered_bins_
        )

        woe_cols = [col for col in X_woe.columns if col.endswith("_woe")]

        X_woe = X_woe[woe_cols]

        return X_woe

class VIFSelector(BaseEstimator, TransformerMixin):
    """
    Transformer loại bỏ dần các biến có VIF > threshold.
    """

    def __init__(self, threshold=5.0):
        self.threshold = threshold

    def calculate_vif(self, X):
        X_clean = X.copy()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.dropna()

        X_const = sm.add_constant(X_clean)

        vif_data = pd.DataFrame()
        vif_data["variable"] = X_const.columns

        vif_data["vif"] = [
            variance_inflation_factor(X_const.values, i)
            for i in range(X_const.shape[1])
        ]

        vif_data = vif_data[vif_data["variable"] != "const"]

        vif_data = vif_data.sort_values(
            "vif",
            ascending=False
        ).reset_index(drop=True)

        return vif_data

    def fit(self, X, y=None):
        X_current = X.copy()

        self.removed_features_ = []

        while True:
            vif_df = self.calculate_vif(X_current)

            max_vif = vif_df["vif"].max()

            if max_vif <= self.threshold:
                break

            feature_to_drop = vif_df.iloc[0]["variable"]

            self.removed_features_.append(
                (feature_to_drop, max_vif)
            )

            X_current = X_current.drop(columns=[feature_to_drop])

        self.selected_features_ = X_current.columns.tolist()
        self.final_vif_ = self.calculate_vif(X_current)

        print("\n========== VIF Selection ==========")
        print("Các biến bị loại do VIF:")
        print(self.removed_features_)

        print("\nVIF cuối cùng:")
        print(self.final_vif_)

        return self

    def transform(self, X):
        X = X.copy()
        return X[self.selected_features_]

preprocess_pipeline = Pipeline([
    (
        "cleaner",
        CreditDataCleaner()
    ),
    (
        "woe_iv",
        WOEIVTransformer(
            target=target,
            iv_threshold=iv_threshold,
            bin_num_limit=bin_num_limit
        )
    ),
    (
        "vif",
        VIFSelector(
            threshold=vif_threshold
        )
    )
])