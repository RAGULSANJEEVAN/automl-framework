import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.linear_model import (LinearRegression, LogisticRegression)
from sklearn.cluster import (KMeans)
import json
class DatasetAnalyzer:

    def __init__(self, df, target):

        self.df = df
        self.target = target

        self.analysis = {

            "dataset_info": {},

            "problem_info": {},

            "feature_info": {},

            "quality_info": {},

            "statistics": {

                "summary": {},

                "correlation": {},

                "skewness": {},

                "variance": {}

            },
            "characteristics" :{

                "constant_columns" : {},

                "cardinality" : {}
            },
            "baseline_model": {}

        }
    def make_json_safe(self, obj):

        if isinstance(obj, dict):
            return {k: self.make_json_safe(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self.make_json_safe(v) for v in obj]

        elif isinstance(obj, tuple):
            return tuple(self.make_json_safe(v) for v in obj)

        elif isinstance(obj, pd.DataFrame):
            return self.make_json_safe(obj.to_dict(orient="index"))

        elif isinstance(obj, pd.Series):
            return self.make_json_safe(obj.to_dict())

        elif isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.dtype):
            return str(obj)

        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        elif pd.isna(obj):
            return None

        return obj

    def dataset_info(self):

        info = {
            
            "num_rows": int(self.df.shape[0]),
            
            "num_columns": int(self.df.shape[1]),
            
            "column_names": self.df.columns.tolist(),
            
            "data_types": {
                col: str(dtype)
                for col, dtype in self.df.dtypes.items()
            }
        }

        self.analysis["dataset_info"] = info

    def problem_info(self):
        
        if self.target is None:
            problem_type = "unsupervised_learning"
            data = None
            dataType = None
            data_unique_values = None
        else:
            problem_type = "supervised_learning"
            data = self.df[self.target]
            dataType = str(data.dtype)
            data_unique_values = int(data.nunique())
        # elif data.nunique()<=10:

        #     problem_type = "classification"

        # elif data.dtype == "object":

        #     problem_type = "classification"
        
        # else:
        #     problem_type = "regression"

        info = {
            "problem_type" : problem_type,

            "target_column" : self.target,

            "target_type" : dataType,

            "target_unique_values" : data_unique_values,
        }

        self.analysis["problem_info"] = info

    def feature_info(self):

        numerical_feature = self.df.select_dtypes(include = ["int64", "float64"]).columns.tolist()

        categorical_features = self.df.select_dtypes(include = ["object","category"]).columns.tolist()

        boolean_features = self.df.select_dtypes(include = ["bool"]).columns.tolist()

        datetime_features = self.df.select_dtypes(include = ["datetime64"]).columns.tolist()

        info = {
            "numerical_features": numerical_feature,
            "categorical_features": categorical_features,
            "boolean_features": boolean_features,
            "datetime_features": datetime_features
        }

        self.analysis["feature_info"] = info

    def quality_info(self):

        missing_value = self.df.isnull().sum().to_dict()

        missing_percentage = (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict()

        duplicate_rows = int(self.df.duplicated().sum())

        info = {
            "missing_values": missing_value,
            "missing_percentage": missing_percentage,
            "duplicate_rows": duplicate_rows
        }

        self.analysis["quality_info"] = self.make_json_safe(info)
    
    def summary_statistics(self):

        summary = (self.df.describe(include="all").T.replace([np.inf, -np.inf], np.nan))

        self.analysis["statistics"]["summary"] = self.make_json_safe(summary)

    def correlation_analysis(self):

        corr = self.df.corr(numeric_only=True).replace([np.inf, -np.inf], np.nan)

        self.analysis["statistics"]["correlation"] = self.make_json_safe(corr)

    def skewness_analysis(self):

        skew = self.df.skew(
            numeric_only=True
        )

        self.analysis["statistics"]["skewness"] = self.make_json_safe(skew)

    def variance_analysis(self):

        variance = self.df.var(
            numeric_only=True
        )

        self.analysis["statistics"]["variance"] = self.make_json_safe(variance)
    
    def constant_columns(self):

        constant = []

        for col in self.df.columns:

            if self.df[col].nunique() == 1:

                constant.append(col)

        self.analysis["characteristics"]["constant_columns"] = constant
    
    def cardinality(self):
        high = []
        encoding_info = {}

        categorical = self.df.select_dtypes(
        include=["object","category"]
        ).columns

        for col in categorical:
            unique_count = int((self.df[col].nunique()))

            encoding_info[col] = {
                "unique_count": unique_count
            }

        self.analysis["characteristics"]["cardinality"] = encoding_info    
    
    def baseline_model(self):

        recommended_model = {}

        problem_type = self.analysis["problem_info"]["problem_type"]

        target_type = self.analysis["problem_info"]["target_type"]

        if problem_type == "supervised_learning" and target_type == "object":

            recommended_model = "Logistic Regression"

            model_name = LogisticRegression()

            method = "classification"

        elif problem_type == "supervised_learning" and target_type in ["int64", "float64"]:

            if self.df[self.target].nunique()<=10:

                recommended_model = "Logistic Regression"

                model_name = LogisticRegression()

                method = "classification"

            else:

                recommended_model = "Linear Regression"

                model_name = LinearRegression()

                method = "regression"
        
        else:

            recommended_model = "KMeans Clustering"

            model_name = KMeans(
                    random_state=42,
                    n_init=10
                )

            method = "clustering"

        recommended_model = {

            "recommended_model": recommended_model,

            "model_name" : model_name,

            "method": method
        }

        self.analysis["baseline_model"] = recommended_model
           

    def analyze(self):

        self.dataset_info()

        self.problem_info()

        self.feature_info()

        self.quality_info()

        self.summary_statistics()

        self.correlation_analysis()

        self.skewness_analysis()

        self.variance_analysis()
        
        self.constant_columns()

        self.cardinality()

        self.baseline_model()

        return self.analysis      



