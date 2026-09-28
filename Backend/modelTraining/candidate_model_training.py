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

import numpy as np

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

    "rmse": lambda y_true, y_pred: np.sqrt(
        mean_squared_error(y_true, y_pred)
    )
}
def training_candidate_model(self, splitData):

    problem_type = self.analysis["problem_info"]["problem_type"]

    method = self.analysis["baseline_model"]["method"]

    candidate_models = self.recommender["candidate_models"]

    metrics = self.recommender["evaluation_metrics"]

    results = []

    for model_name, model in candidate_models.items():

        if problem_type == "supervised_learning":

            X_train, X_test, y_train, y_test = splitData

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            model_result = {

                "model_name": model_name,

                "model": model,

            }

            for metric_key, metric in metrics.items():

                metric_function = METRIC_FUNCTIONS[metric["function"]]

                metric_name = metric["function"]

                try:

                    if metric_name in [
                        "precision_score",
                        "recall_score",
                        "f1_score"
                    ]:

                        score = metric_function(
                            y_test,
                            y_pred,
                            average="weighted"
                        )

                    elif metric_name == "roc_auc_score":

                        score = metric_function(
                            y_test,
                            model.predict_proba(X_test),
                            multi_class="ovr"
                        )

                    else:

                        score = metric_function(
                            y_test,
                            y_pred
                        )

                except Exception:

                    score = None

                model_result[metric["display_name"]] = score

        else:

            X = splitData

            labels = model.fit_predict(X)

            model_result = {

                "model_name": model_name,

                "model": model,

            }

            for metric_key, metric in metrics.items():

                metric_function = METRIC_FUNCTIONS[metric["function"]]

                metric_name = metric["function"]

                try:

                    score = metric_function(
                        X,
                        labels
                    )

                except Exception:

                    score = None

                model_result[metric["display_name"]] = score

        results.append(model_result)

    return results