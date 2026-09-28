from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


def identifier_evaluation_metrics(self):

    method = self.analysis["baseline_model"]["method"]

    imbalance = self.recommendation["candidate_models"].get(
        "imbalance",
        None
    )

    if method == "regression":

        return {

            "R2": {
                "display_name": "R² Score",
                "function": "r2_score",
                "greater_is_better": True
            },

            "RMSE": {
                "display_name": "RMSE",
                "function": "rmse",
                "greater_is_better": False
            },

            "MAE": {
                "display_name": "MAE",
                "function": "mean_absolute_error",
                "greater_is_better": False
            },

            "MSE": {
                "display_name": "MSE",
                "function": "mean_squared_error",
                "greater_is_better": False
            }

        }

    elif method == "classification":

        if imbalance == "Severe":

            return {

                "Precision": {
                    "display_name": "Precision",
                    "function": "precision_score",
                    "greater_is_better": True
                },

                "Recall": {
                    "display_name": "Recall",
                    "function": "recall_score",
                    "greater_is_better": True
                },

                "F1": {
                    "display_name": "F1 Score",
                    "function": "f1_score",
                    "greater_is_better": True
                },

                "ROC_AUC": {
                    "display_name": "ROC-AUC",
                    "function": "roc_auc_score",
                    "greater_is_better": True
                }

            }

        return {

            "Accuracy": {
                "display_name": "Accuracy",
                "function": "accuracy_score",
                "greater_is_better": True
            },

            "Precision": {
                "display_name": "Precision",
                "function": "precision_score",
                "greater_is_better": True
            },

            "Recall": {
                "display_name": "Recall",
                "function": "recall_score",
                "greater_is_better": True
            },

            "F1": {
                "display_name": "F1 Score",
                "function": "f1_score",
                "greater_is_better": True
            }

        }

    elif method == "clustering":

        return {

            "Silhouette": {
                "display_name": "Silhouette Score",
                "function": "silhouette_score",
                "greater_is_better": True
            },

            "Calinski": {
                "display_name": "Calinski-Harabasz Score",
                "function": "calinski_harabasz_score",
                "greater_is_better": True
            },

            "Davies": {
                "display_name": "Davies-Bouldin Score",
                "function": "davies_bouldin_score",
                "greater_is_better": False
            }

        }

    elif method == "anomaly_detection":

        return {

            "ROC_AUC": {
                "display_name": "ROC-AUC",
                "function": "roc_auc_score",
                "greater_is_better": True
            },

            "Precision": {
                "display_name": "Precision",
                "function": "precision_score",
                "greater_is_better": True
            },

            "Recall": {
                "display_name": "Recall",
                "function": "recall_score",
                "greater_is_better": True
            },

            "F1": {
                "display_name": "F1 Score",
                "function": "f1_score",
                "greater_is_better": True
            }

        }

    elif method == "dimensionality_reduction":

        return {

            "ExplainedVariance": {
                "display_name": "Explained Variance Ratio",
                "function": "explained_variance_ratio",
                "greater_is_better": True
            },

            "ReconstructionError": {
                "display_name": "Reconstruction Error",
                "function": "reconstruction_error",
                "greater_is_better": False
            }

        }

    return {}