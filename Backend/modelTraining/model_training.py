from modelTraining.feature_selection import supervised_learning_feature_selection
from modelTraining.feature_selection import unsupervised_learning_feature_selection
from modelTraining.feature_scaling import handling_supervised_feature_scaling
from modelTraining.feature_scaling import handling_unsupervised_feature_scaling
from sklearn.linear_model import (LinearRegression, LogisticRegression)
from sklearn.cluster import (KMeans)
from modelTraining.base_model_evaluation import base_model_evaluation
from modelTraining.candidate_model_training import training_candidate_model
from modelTraining.hyper_parameter_training import training_hyper_parameter
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
SCORING = {

    "R² Score": "r2",

    "RMSE": "neg_root_mean_squared_error",

    "MSE": "neg_mean_squared_error",

    "MAE": "neg_mean_absolute_error",

    "Accuracy": "accuracy",

    "Precision": "precision_weighted",

    "Recall": "recall_weighted",

    "F1 Score": "f1_weighted",

    "ROC-AUC": "roc_auc_ovr",

    "Silhouette Score": "silhouette_score",

    "Calinski-Harabasz Score": "calinski_harabasz_score",

    "Davies-Bouldin Score": "davies_bouldin_score"

}

class modelTrainingEngine:

    def serialize(self, obj):

        if isinstance(obj, dict):
            return {k: self.serialize(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self.serialize(v) for v in obj]

        elif isinstance(obj, tuple):
            return tuple(self.serialize(v) for v in obj)

        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")

        elif isinstance(obj, pd.Series):
            return obj.to_dict()

        elif isinstance(obj, pd.Index):
            return obj.tolist()

        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        elif isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif hasattr(obj, "get_params"):   # sklearn estimator
            return obj.__class__.__name__

        return obj

    def __init__(self, analysis, recommender, data):

        self.analysis = analysis

        self.recommender = recommender

        self.data = data

        self.problem_type = self.analysis["problem_info"]["problem_type"]

        self.model_training = {

            "feature_selection" : {},

            "evaluating_base_model" : {},

            "candidate_model_training" : {},

            "result_evaluation" : {},

            "cross_validation_score" : {},

            "hyper_parameter_info" : {}
        }

    def feature_scaling(self):
        
        if self.problem_type == "supervised_learning":
                
            self.df_model_training = handling_supervised_feature_scaling(self, self.data)

        elif self.problem_type == "unsupervised_learning":

            self.df_model_training = handling_unsupervised_feature_scaling(self, self.data)

    def feature_selection(self):

        if self.problem_type == "supervised_learning":
                
            self.df_model_training, self.feature_selection_info = supervised_learning_feature_selection(self, self.df_model_training)

        elif self.problem_type == "unsupervised_learning":

            self.df_model_training, self.feature_selection_info = unsupervised_learning_feature_selection(self, self.df_model_training)

        self.model_training["feature_selection"] = self.feature_selection_info

    def baseline_model_training(self):

        baseline_info = self.analysis["baseline_model"]

        method = baseline_info["method"]

        baseline_model = baseline_info["model_name"]

        if method == "clustering":

            X = self.df_model_training

            if X is not None:
                print("Shape:", X.shape)

            labels = baseline_model.fit_predict(X)

            self.splitData = X, labels

            return self.splitData

        elif method in ("classification", "regression"):
            
            X_train, X_test, y_train, y_test = self.df_model_training

            baseline_model.fit(X_train, y_train)

            y_pred = baseline_model.predict(X_test)

            self.splitData = (X_train, X_test, y_train, y_test, y_pred)

            return self.splitData

    def evaluating_base_model(self):
        
        self.evaluation_data = base_model_evaluation(self, self.splitData)

        self.model_training["evaluating_base_model"] = self.evaluation_data

    def candidate_model_training(self):

        self.candidate_model_evaluation = training_candidate_model(self, self.df_model_training)

        self.model_training["candidate_model_training"] = self.candidate_model_evaluation

    def result_evaluation(self):

        self.evaluation_data.extend(self.candidate_model_evaluation)

        self.final_evaluation_matrics = pd.DataFrame(self.evaluation_data)

        baseline_info = self.analysis["baseline_model"]

        method = baseline_info["method"]

        if method == "regression":

            self.sort_value = "R² Score"

        elif method == "classification":

            self.sort_value = "Precision"

        elif method == "clustering":

            self.sort_value = "Silhouette Score"

        else:

             self.sort_value = None

             return

        self.final_evaluation_matrics = self.final_evaluation_matrics.sort_values(self.sort_value, ascending=False)  

        self.best_model_name = self.final_evaluation_matrics.iloc[0]["model"]

        evaluation = self.final_evaluation_matrics.to_dict(orient="records")
        
        for row in evaluation:
            
            if "model" in row and hasattr(row["model"], "get_params"):
                
                row["model"] = row["model"].__class__.__name__
                
        result_evaluation_info = {

            "final_evaluation_matrics": evaluation,

            "best_model_name": self.best_model_name.__class__.__name__
        }

        self.model_training["result_evaluation"] = result_evaluation_info

    def cross_validation_score(self):

        cross_validation_info = {}

        self.scoring = SCORING[self.sort_value]

        if self.problem_type == "supervised_learning":

            X_train, X_test, y_train, y_test, y_pred = self.splitData

            self.cv_scores = cross_val_score(

                self.best_model_name,

                X_train,

                y_train,

                cv=5,

                scoring=self.scoring,

                n_jobs=-1

            )

            print(self.cv_scores)

            print("Mean CV Score:", self.cv_scores.mean())
        
            cross_validation_info = {

                "cv_scores" : self.cv_scores,

                "mean cv_scores" : self.cv_scores.mean()

            }
        
        self.model_training["cross_validation_score"] = cross_validation_info

    def hyper_parameter_handling(self):

        hyper_parameter_info = {}

        self.hyper_parameter_evaluation, self.cluster_data = training_hyper_parameter(self,self.splitData,self.best_model_name,self.scoring)

        print("hyper_parameter_evaluation",self.hyper_parameter_evaluation)

        # if self.cluster_data is not None and len(self.cluster_data) > 0:

            # if self.problem_type == "supervised_learning":

            #     # print("self.cluster_data", self.cluster_data.head(10))

            # else:
            
            #     print("self.cluster_data", self.cluster_data["cluster_size"])

        hyper_parameter_info = {

            "hyper_parameter_evaluation" : self.hyper_parameter_evaluation,

            "cluster_data" : self.cluster_data
        }

        self.model_training["hyper_parameter_info"] = hyper_parameter_info

    def modelTraining(self):

        self.feature_scaling()

        self.feature_selection()

        self.baseline_model_training()

        self.evaluating_base_model()

        self.candidate_model_training()

        self.result_evaluation()
        
        self.cross_validation_score()

        self.hyper_parameter_handling()

        return self.serialize(self.model_training)

        # return self.baseline_model_training()
        
        # if self.problem_type == "supervised_learning":
 
        #     return self.split_data()

        # else:

        #     return self.df_model_training


