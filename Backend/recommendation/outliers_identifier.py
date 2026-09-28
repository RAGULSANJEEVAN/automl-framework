from scipy.stats import shapiro
import pandas as pd

def identify_outliers(self, df):

    data_size = self.analysis["dataset_info"]["num_rows"]
    numerical_columns = self.analysis["feature_info"]["numerical_features"]

    outlier_info = {}

    if len(numerical_columns) == 0:
        return outlier_info

    def z_score_outliers(column):

        mean = df[column].mean()

        std = df[column].std()

        if std == 0:
            return {
                "method": "Z-Score",
                "mean": float(mean),
                "std": float(std),
                "outlier_count": 0,
                "percentage": 0.0
            }

        z_scores = (df[column] - mean) / std

        count = int(((z_scores < -3) | (z_scores > 3)).sum())
        percentage = round((count / len(df)) * 100, 2)

        return {
            "method": "Z-Score",
            "mean": float(mean),
            "std": float(std),
            "outlier_count": count,
            "percentage": percentage
        }

    def iqr_outliers(column):

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        count = int(
            df[
                (df[column] < lower_bound) |
                (df[column] > upper_bound)
            ].shape[0]
        )

        percentage = round((count / len(df)) * 100, 2)

        return {
            "method": "IQR",
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": count,
            "percentage": percentage
        }

    for column in numerical_columns:

        if data_size <= 5000:

            try:
                _, p = shapiro(df[column].dropna())
                distribution = "normal" if p > 0.05 else "not_normal"
            except Exception:
                distribution = "not_normal"

        else:

            skewness = self.analysis["statistics"]["skewness"].get(column, 0)

            distribution = "normal" if abs(skewness) < 0.5 else "not_normal"

        if distribution == "normal":

            data = z_score_outliers(column)

        else:

            data = iqr_outliers(column)

        outlier_info[column] = {
            "distribution": distribution,
            "data": data,
            "count": data["outlier_count"],
            "percentage": data["percentage"]
        }

    return outlier_info