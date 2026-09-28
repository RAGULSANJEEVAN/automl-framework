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

METRIC_FUNCTIONS = {

    "r2_score": r2_score,

    "mean_squared_error": mean_squared_error,

    "mean_absolute_error": mean_absolute_error,

    "accuracy_score": accuracy_score,

    "precision_score": precision_score,

    "recall_score": recall_score,

    "f1_score": f1_score,

    "roc_auc_score": roc_auc_score,

    "silhouette_score": silhouette_score,

    "calinski_harabasz_score": calinski_harabasz_score,

    "davies_bouldin_score": davies_bouldin_score,

    "rmse": lambda y_true, y_pred: mean_squared_error(
        y_true,
        y_pred,
    )
}

def base_model_evaluation(self, splitData):

    self.problem_type = self.analysis["problem_info"]["problem_type"]

    baseline_info = self.analysis["baseline_model"]

    baseline_model_name = baseline_info["recommended_model"]

    baseline_model_method = baseline_info["model_name"]

    model_result = {
        
        "model_name": baseline_model_name,

        "model": baseline_model_method,
        
        }

    if self.problem_type == "supervised_learning":

        X_train, X_test, y_train, y_test, y_pred = splitData

    else:

        X, labels = splitData

    results = []

    metrics = self.recommender["evaluation_metrics"]

    if self.problem_type == "supervised_learning":

        for key, metric in metrics.items():

            n_classes = len(set(y_test))

            if n_classes == 2:
                average = "binary"
            else:
                average = "weighted"

            function = METRIC_FUNCTIONS[metric["function"]]

            metric_name = metric["display_name"]

            if metric_name in ["Precision","Recall","F1 Score"]:
                
                score = function(
                    y_test,
                    y_pred,
                    average=average
                )

            else:

                score = function(
                    y_test,
                    y_pred
                )

            model_result[metric["display_name"]] = score

    else:

        for key, metric in metrics.items():

            function = METRIC_FUNCTIONS[metric["function"]]

            score = function(X, labels)

            model_result[metric["display_name"]] = score

    
    results.append(model_result)

    return results


    

