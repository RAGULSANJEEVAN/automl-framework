import pandas as pd
from preprocessing.handling_missing_values import handel_missing_value
from preprocessing.handling_categorical_value import encoding_cat_data
from preprocessing.handling_outliers_value import handel_outlier_value
from sklearn.model_selection import train_test_split
class preprocessingEngine():

    def __init__(self, analysis, recommender, df):

        self.analysis = analysis

        self.recommender = recommender

        self.df_preprocessing = df

        self.problem_type = self.analysis["problem_info"]["problem_type"]

        self.proprocessed_data = {

            "dataTypeCorrection" : {},

            "missing_value_info" : {},

            "duplicated_count" : "",

            "encoding_info" : {},

            "removed_columns" : {}
        }

    def dataTypeCorrection(self):

        incorrect_columns = self.recommender["preprocessing"]["data_cleaning"]["incorrect_dataType_column"]

        dataTypeCorrection_info = []

        if not incorrect_columns:

            self.proprocessed_data["dataTypeCorrection"] = {

                "column" : "All column",

                "dataType" : "All Columns has correct DataTypes"

            }

            return self.df_preprocessing

        else:

            for column in incorrect_columns:
            
                self.df_preprocessing[column] = pd.to_datetime(self.df_preprocessing[column])

                dataTypeCorrection_info.append({

                    "column" : column,

                    "dataType" : str(self.df_preprocessing[column].dtype)

                })

        self.proprocessed_data["dataTypeCorrection"] = dataTypeCorrection_info

        return self.df_preprocessing

    def missing_values_handling(self):

        self.df_preprocessing, self.handel_missing_value_info = handel_missing_value(self,self.df_preprocessing)

        self.proprocessed_data["missing_value_info"] = self.handel_missing_value_info

        # return self.df_preprocessing

    def drop_duplicates(self):

        duplicate_date = self.analysis["quality_info"]["duplicate_rows"]

        if duplicate_date > 0:

            self.df_preprocessing = self.df_preprocessing.drop_duplicates()

        self.proprocessed_data["duplicated_count"] = int(self.df_preprocessing.duplicated().sum())
            
        return self.df_preprocessing

    def handling_outliers(self):

        self.df_preprocessing = handel_outlier_value(self,self.df_preprocessing)

    def encoding_data(self):

        self.df_preprocessing, self.encoding_summary = encoding_cat_data(self,self.df_preprocessing)

        self.proprocessed_data["encoding_info"] = self.encoding_summary

    def remove_unwanted_column(self):

        id_columns = None

        for column in self.df_preprocessing:

            if (column.lower() == "id" or column.lower().endswith("_id") or "s.no" in column.lower()): 

                id_columns = column

                self.df_preprocessing = self.df_preprocessing.drop(column, axis=1)

        unwanted_col = (self.df_preprocessing.select_dtypes(exclude=['int', 'float']).columns).tolist()

        if len(unwanted_col) > 0:

            self.df_preprocessing = self.df_preprocessing.drop(columns=unwanted_col)

        removed_columns = {

            "id_columns" : id_columns,

            "unwanted_col" : unwanted_col
        }

        self.proprocessed_data["removed_columns"] = removed_columns

        return self.df_preprocessing

    def split_data(self):

        target_column = self.analysis["problem_info"].get("target_column")

        if  self.problem_type == "supervised_learning":

            X = self.df_preprocessing.drop(target_column, axis=1)

            y = self.df_preprocessing[target_column]

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )
            )

            return X_train, X_test, y_train, y_test
        else:

            return self.df_preprocessing

    def proprocessor(self):

        self.dataTypeCorrection()

        self.missing_values_handling()

        self.drop_duplicates()

        self.handling_outliers()

        self.encoding_data()

        self.remove_unwanted_column()

        if self.problem_type == "supervised_learning":

            return self.proprocessed_data, self.split_data()

        else:

            return self.proprocessed_data, self.df_preprocessing