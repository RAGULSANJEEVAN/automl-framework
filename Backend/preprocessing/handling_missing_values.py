from helper.missing_value_helper import (
    drop_column,
    mean_imputation,
    median_imputation,
    mode_imputation,
    forward_imputation,
    knn_imputation,
    iterative_imputation
)


def handel_missing_value(self, df):

    missing_info = self.recommender["preprocessing"]["handling_missing_values"]

    if missing_info.get("missing_value") is False:

        handel_missing_value_info = {

            "column" : "All Column",

            "action" : "No Action",

            "Na_values"  : 0
        }

        return df, handel_missing_value_info

    for column, details in missing_info.items():

        action = details.get("recommended_action")

        if action == "Drop Column":

            df = drop_column(column, df)

        elif action == "Mean Imputation":

            df = mean_imputation(column, df)

        elif action == "Median Imputation":

            df = median_imputation(column, df)

        elif action == "Forward Fill":

            df = forward_imputation(column, df)

        elif action == "Mode Imputation":

            df = mode_imputation(column, df)

        elif action == "KNN Imputation":

            df = knn_imputation(df)

        elif action == "Iterative Imputation":

            df = iterative_imputation(df)

        else:

            print(f"No preprocessing action for '{column}'.")

        if column not in df.columns:
            print(f"'{column}' not found.")
            print(df.columns.tolist())
        else:
            handel_missing_value_info = {
                "column": column,
                "action": action,
                "Na_values": df[column].isnull().sum()
            }

    return df, handel_missing_value_info