import pandas as pd
from recommendation.outliers_identifier import identify_outliers
from recommendation.encoding_identifier import identify_encoding
from recommendation.missing_value_drop_na import missing_values_identifier
from recommendation.candidate_model_identifier import identifier_candidate_model
from recommendation.evaluation_metrics_identifier import identifier_evaluation_metrics

class RecommendationEngine:
    
    def __init__(self, analysis, df):

        self.analysis = analysis

        self.df_copy = df.copy()

        self.df_recommendation = df.copy()

        self.recommendation = {

            "preprocessing": {

                "data_cleaning": {},

                "handling_missing_values": {},

                "outliers": {},

                "encoding": {}
            },

            "baseline_model": "",

            "candidate_models": {},

            "evaluation_metrics": {},

            "cross_validation": True,

            "hyperparameter_tuning": True

        }
  
    def dataType_identifier(self):

        dataType = self.analysis["dataset_info"]["data_types"]

        incorrect_data_types = []

        for column, dtype in dataType.items():

            if "date" in column.lower() or "time" in column.lower():

                if not pd.api.types.is_datetime64_any_dtype(self.df_recommendation[column]):

                    incorrect_data_types.append(column)

        self.recommendation["preprocessing"]["data_cleaning"]["incorrect_dataType_column"] = incorrect_data_types
    
    def missing_values_identifier(self):

        missing_values,self.df_recommendation = missing_values_identifier(self,self.df_recommendation)

        self.recommendation["preprocessing"]["handling_missing_values"] = missing_values

    def outliers_identifier(self):

        outliers = identify_outliers(self, self.df_recommendation)

        self.recommendation["preprocessing"]["outliers"] = outliers

    def encoding_identifier(self):

        encoding_info = identify_encoding(self, self.df_recommendation)

        self.recommendation["preprocessing"]["encoding"] = encoding_info

    def baseline_model_identifier(self):

        baseline_model = self.analysis["baseline_model"]["recommended_model"]

        self.recommendation["baseline_model"] = baseline_model

    def candidate_identifier(self):

        candidate_model = identifier_candidate_model(self, self.df_recommendation)

        self.recommendation["candidate_models"] = candidate_model

    def evaluation_metrics_identifier(self):

        evaluation_metrics = identifier_evaluation_metrics(self)

        self.recommendation["evaluation_metrics"] = evaluation_metrics
    
    def recommend(self):
            
        self.dataType_identifier()

        self.missing_values_identifier()

        self.outliers_identifier()

        self.encoding_identifier()

        self.baseline_model_identifier()

        self.candidate_identifier() 

        self.evaluation_metrics_identifier()

        return self.recommendation
