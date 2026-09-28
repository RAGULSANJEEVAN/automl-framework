import pandas as pd

def identify_encoding(self, df):

    categorical_columns = self.analysis["feature_info"]["categorical_features"]

    target_column = self.analysis["problem_info"].get("target_column")
    
    problem_type = self.analysis["problem_info"]["problem_type"]

    incorrect_dataType_column = set(
        self.recommendation["preprocessing"]["data_cleaning"].get(
        "incorrect_dataType_column", []
        )
    )

    categorical_columns = [
        col for col in categorical_columns
        if col not in incorrect_dataType_column
    ]

    encoding_info = {}

    if len(categorical_columns) == 0:
        return encoding_info

    for column in categorical_columns:

        if column == target_column:

            encoding_info[column] = {
                "encoding": "Label Encoding",

                "reason": "Target Column"
            }
            continue

        if (
            column.lower() == "id"
            or column.lower().endswith("_id")
            or "s.no" in column.lower()
        ):

            encoding_info[column] = {
                "encoding": "No Encoding",

                "reason": "Identifier Column"
            }

            continue

        unique_count = df[column].nunique(dropna=True)

        if unique_count <= 15:

            encoding = "One-Hot Encoding"

        elif unique_count <= 100:

            encoding = "Frequency Encoding"

        else:

            if problem_type in ["supervised_learning"]:

                encoding = "Target Encoding"

            else:

                encoding = "Hash Encoding"

        encoding_info[column] = {

            "encoding": encoding,   

            "unique_count": int(unique_count)
        }

    return encoding_info