import pandas as pd
from helper.missing_value_helper import drop_column, mean_imputation, median_imputation, mode_imputation, forward_imputation, knn_imputation, iterative_imputation
def missing_values_identifier(self, df):

    missing_values = self.analysis["quality_info"]["missing_values"]
    data_size = self.analysis["dataset_info"]["num_rows"]

    missing_info = {}

    for column, missing_count in missing_values.items():

        if missing_count == 0:
            continue

        missing_percent = round(
            (missing_count / len(df)) * 100,
            2
        )

        if missing_percent <= 30:

            if pd.api.types.is_numeric_dtype(df[column]):

                skewness = self.analysis["statistics"]["skewness"].get(column, 0)

                if abs(skewness) < 0.5:

                    action = "Mean Imputation"
                    mean_imputation(column, df)

                else:

                    action = "Median Imputation"
                    median_imputation(column, df)

            elif pd.api.types.is_datetime64_any_dtype(
                df[column]):

                action = "Forward Fill"
                forward_imputation(column, df)

            else:

                action = "Mode Imputation"
                mode_imputation(column, df)

        elif missing_percent <= 60:

            if pd.api.types.is_numeric_dtype(df[column]):

                if data_size <= 5000:

                    action = "KNN Imputation"
                    knn_imputation(df)

                else:

                    action = "Iterative Imputation"
                    iterative_imputation(df)

            else:

                action = "Mode Imputation"
                mode_imputation(column, df)

        else:

            action = "Drop Column"
            drop_column(column, df)

        missing_info[column] = {

            "missing_value": True,

            "missing_count": int(missing_count),

            "missing_percentage": float(missing_percent),

            "recommended_action": action
        }

    if len(missing_info) == 0:

        return {
            "missing_value": False,

            "status": "No Missing Values Found",

            "recommendation": None

        }, df

    return missing_info, df
