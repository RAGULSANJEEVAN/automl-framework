from helper.encoding_values_helper import (
    one_hot_encoding,
    frequency_encoding,
    target_encoding,
    hash_encoding,
    label_encoding
)

def encoding_cat_data(self, df):

    encoding_info = self.recommender["preprocessing"]["encoding"]

    target_column = self.analysis["problem_info"].get("target_column")

    encoding_summary = {}

    if not encoding_info:
        return df, encoding_summary

    for column, details in encoding_info.items():

        if column not in df.columns:

            print(f"Skipping '{column}' because it is not present.")
            continue

        method = details.get("encoding")

        old_columns = set(df.columns)

        if method == "One-Hot Encoding":

            df = one_hot_encoding(column, df)

        elif method == "Frequency Encoding":

            df = frequency_encoding(column, df)

        elif method == "Target Encoding":

            df = target_encoding(column, df, target_column)

        elif method == "Hash Encoding":

            df = hash_encoding(column, df)

        elif method == "Label Encoding":

            df = label_encoding(column, df)

        else:

            print(f"No encoding applied for {column}")
            continue

        new_columns = set(df.columns)

        created_columns = list(new_columns - old_columns)

        # If the encoding modifies the same column
        if len(created_columns) == 0:
            created_columns = [column]

        encoding_summary[column] = {

            "method": method,

            "original_column": column,

            "new_columns": sorted(created_columns)
        }

    return df, encoding_summary