from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    BaggingRegressor
)

from sklearn.svm import (
    LinearSVC,
    SVR
)

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsRegressor

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    OPTICS,
    MiniBatchKMeans
)

from sklearn.mixture import GaussianMixture

# External Libraries
from xgboost import XGBRegressor, XGBClassifier
# from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
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
import time
from joblib import Parallel, delayed
param_grids = {
    LogisticRegression: {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"]
    },

    DecisionTreeClassifier: {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    },

    RandomForestClassifier: {
        "n_estimators": [100, 300],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    },

    # GradientBoostingClassifier: {
    #     "n_estimators": [100, 200],
    #     "learning_rate": [0.01, 0.05, 0.1],
    #     "max_depth": [3, 5],
    #     "subsample": [0.8, 1.0]
    # },

    # AdaBoostClassifier: {
    #     "n_estimators": [50, 100, 200],
    #     "learning_rate": [0.01, 0.1, 1]
    # },

    # KNeighborsClassifier: {
    #     "n_neighbors": [3, 5, 7, 9],
    #     "weights": ["uniform", "distance"],
    #     "p": [1, 2]
    # },

    # SVC: {
    #     "kernel": ["linear", "rbf"],
    #     "C": [0.1, 1, 10, 100],
    #     "gamma": ["scale", "auto"]
    # },

    GaussianNB: {
        "var_smoothing": [1e-12, 1e-10, 1e-9, 1e-8]
    },
    LinearRegression: {},

    Ridge: {
        "alpha": [0.01, 0.1, 1, 10, 100]
    },

    Lasso: {
        "alpha": [0.001, 0.01, 0.1, 1, 10]
    },

    ElasticNet: {
        "alpha": [0.001, 0.01, 0.1, 1],
        "l1_ratio": [0.2, 0.5, 0.8]
    },

    DecisionTreeRegressor: {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },

    RandomForestRegressor: {
        "n_estimators": [100, 300],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    },

    ExtraTreesRegressor: {
        "n_estimators": [100, 300],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },

    # GradientBoostingRegressor: {
    #     "n_estimators": [100, 200],
    #     "learning_rate": [0.01, 0.05, 0.1],
    #     "max_depth": [3, 5],
    #     "subsample": [0.8, 1.0]
    # },

    # KNeighborsRegressor: {
    #     "n_neighbors": [3, 5, 7, 9],
    #     "weights": ["uniform", "distance"],
    #     "p": [1, 2]
    # },

    SVR: {
        "kernel": ["linear", "rbf"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto"],
        "epsilon": [0.01, 0.1, 0.5]
    },
    KMeans: {
        "n_clusters": [2, 3, 4, 5, 6, 8, 10],
        "init": ["k-means++", "random"],
        "n_init": [10, 20],
        "max_iter": [300, 500]
    },

    MiniBatchKMeans: {
        "n_clusters": [2, 3, 4, 5, 6, 8, 10],
        "batch_size": [100, 256, 512],
        "n_init": [10, 20],
        "max_iter": [100, 300]
    },

    AgglomerativeClustering: {
        "n_clusters": [2, 3, 4, 5, 6, 8],
        "linkage": ["ward", "complete", "average", "single"],
        "metric": ["euclidean", "manhattan", "cosine"]
    },

    Birch: {
        "n_clusters": [None, 2, 3, 5, 8],
        "threshold": [0.3, 0.5, 0.7],
        "branching_factor": [25, 50, 100]
    },

    DBSCAN: {
        "eps": [0.2, 0.5, 1.0, 2.0],
        "min_samples": [3, 5, 10],
        "metric": ["euclidean", "manhattan"]
    },

    OPTICS: {
        "min_samples": [5, 10, 20],
        "max_eps": [1.0, 2.0, np.inf],
        "metric": ["minkowski", "euclidean", "manhattan"]
    },

    GaussianMixture: {
        "n_components": [2, 3, 4, 5, 6],
        "covariance_type": ["full", "tied", "diag", "spherical"],
        "max_iter": [100, 200],
        "reg_covar": [1e-6, 1e-5, 1e-4]
    },

    XGBClassifier: {
        "n_estimators": [50,100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    },

    XGBRegressor: {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    },

    # LGBMClassifier: {
    #     "n_estimators": [100, 200, 300],
    #     "learning_rate": [0.01, 0.05, 0.1],
    #     "num_leaves": [31, 50, 100],
    #     "max_depth": [-1, 5, 10],
    #     "subsample": [0.8, 1.0]
    # },

    # LGBMRegressor: {
    #     "n_estimators": [100, 200, 300],
    #     "learning_rate": [0.01, 0.05, 0.1],
    #     "num_leaves": [31, 50, 100],
    #     "max_depth": [-1, 5, 10],
    #     "subsample": [0.8, 1.0]
    # },

    CatBoostClassifier: {
        "iterations": [100, 200, 500],
        "learning_rate": [0.05, 0.1],
        "depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7],
        "verbose": False
    },

    CatBoostRegressor: {
        "iterations": [100, 200, 500],
        "learning_rate": [0.05, 0.1],
        "depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7],
        "verbose": False
    }, 

    BalancedRandomForestClassifier: {
        "n_estimators": [100, 300],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }
}

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

def sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None so json.dumps doesn't choke."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, pd.Series):
        return sanitize_for_json(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return sanitize_for_json(obj.to_dict(orient="records"))
    return obj

def training_hyper_parameter(self, splitData, best_model_name, score):

    self.problem_type = self.analysis["problem_info"]["problem_type"]
    self.metrics = self.recommender["evaluation_metrics"]
    baseline_info = self.analysis["baseline_model"]
    self.method = baseline_info["method"]

    if self.problem_type == "supervised_learning":
        self.hyper_parameter_metrics = supervised_training_hyper_parameter(
            self, splitData, best_model_name, score
        )
        return self.hyper_parameter_metrics
    else:
        self.hyper_parameter_metrics = unsupervised_learning_training_hyper_parameter(
            self, splitData, best_model_name, score
        )
        return self.hyper_parameter_metrics


def supervised_training_hyper_parameter(self, splitData, best_model_name, score):

    X_train, X_test, y_train, y_test, y_pred = splitData

    model_type = type(best_model_name)
    param_grid = param_grids.get(model_type, {})

    if not param_grid:
        return None, None

    grid = RandomizedSearchCV(
        estimator=best_model_name,
        param_distributions=param_grid,   # FIX: was param_grid=, wrong kwarg name
        cv=2,                             # FIX: 3 -> 2, cuts fit count by ~33%
        n_iter=15,                        # FIX: 20 -> 15, fewer candidates
        scoring=score,
        n_jobs=-1,
        random_state=42
    )
    grid.fit(X_train, y_train)

    self.hyper_parameter_metrics = {
        "parameters": grid.best_params_,
        "score": grid.best_score_,
        "best_model": grid.best_estimator_
    }

    best_model = grid.best_estimator_
    prediction = best_model.predict(X_test)

    for metric_key, metric_cfg in self.metrics.items():

        metric_function = METRIC_FUNCTIONS[metric_cfg["function"]]
        metric_name = metric_cfg["function"]

        try:
            if metric_name in ["precision_score", "recall_score", "f1_score"]:
                score_val = metric_function(y_test, prediction, average="weighted")

            elif metric_name == "roc_auc_score":
                # FIX: was calling model.predict_proba(prediction) — wrong.
                # roc_auc needs probabilities of X_test, not predictions of predict()
                if hasattr(best_model, "predict_proba"):
                    proba = best_model.predict_proba(X_test)
                    score_val = metric_function(y_test, proba, multi_class="ovr")
                else:
                    score_val = None

            else:
                score_val = metric_function(y_test, prediction)

        except Exception:
            score_val = None

        self.hyper_parameter_metrics[metric_cfg["display_name"]] = score_val

    comparison = pd.DataFrame({
        "Actual": y_test,
        "Prediction": prediction
    })

    if self.method == "regression":
        comparison["Error"] = comparison["Actual"] - comparison["Prediction"]
        comparison["Absolute Error"] = comparison["Error"].abs()
        comparison = comparison.sort_values("Absolute Error", ascending=False)  # FIX: wasn't reassigned before
    else:
        comparison["Correct"] = comparison["Actual"] == comparison["Prediction"]

    return sanitize_for_json(self.hyper_parameter_metrics), sanitize_for_json(comparison)


def unsupervised_learning_training_hyper_parameter(self, splitData, best_model, score):

    print("unsupervised_learning_training_hyper_parameter")

    X, labels = splitData
    model_type = type(best_model)
    param_grid = param_grids.get(model_type, {})

    if not param_grid:
        return None, None

    # Convert to numpy for fast fancy-indexing during silhouette subsampling
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)

    def evaluate_params(params, sample_size=2000):
        try:
            model = model_type(**params)
            fit_labels = model.fit_predict(X_arr)

            n_clusters = len(set(fit_labels))
            if -1 in fit_labels:
                n_clusters -= 1
            if n_clusters < 2:
                return None

            # FIX: silhouette_score is O(n^2) — subsample large datasets
            if len(X_arr) > sample_size:
                idx = np.random.RandomState(42).choice(len(X_arr), sample_size, replace=False)
                sil = silhouette_score(X_arr[idx], fit_labels[idx])
            else:
                sil = silhouette_score(X_arr, fit_labels)

            return (sil, params, model, fit_labels)
        except Exception:
            return None

    start = time.time()
    # FIX: run the grid in parallel instead of a serial for-loop
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_params)(params) for params in ParameterGrid(param_grid)
    )

    results = [r for r in results if r is not None]

    if not results:
        print("No valid parameter combination produced >=2 clusters.")
        return None, None

    best_score, best_params, tuned_model, pred_labels = max(results, key=lambda r: r[0])

    self.hyper_parameter_metrics = {
        "parameters": best_params,
        "score": best_score,
        "best_model": tuned_model
    }

    # FIX: AgglomerativeClustering/DBSCAN/OPTICS/Birch have no .predict() —
    # reuse the labels already produced by fit_predict instead of calling .predict again
    if hasattr(tuned_model, "predict") and model_type not in (
        AgglomerativeClustering, DBSCAN, OPTICS
    ):
        try:
            pred_labels = tuned_model.predict(X_arr)
        except Exception:
            pass  # fall back to fit_predict labels already stored

    for metric_key, metric_cfg in self.metrics.items():

        metric_function = METRIC_FUNCTIONS[metric_cfg["function"]]

        try:
            score_val = metric_function(X_arr, pred_labels)
        except Exception:
            score_val = None

        self.hyper_parameter_metrics[metric_cfg["display_name"]] = score_val

    # FIX: moved out of the metrics loop — was being recomputed once per metric
    self.df_model_training["cluster"] = pred_labels

    cluster_summary = self.df_model_training.groupby("cluster").agg(
        ["mean", "median", "min", "max"]
    )
    cluster_summary.columns = [
        f"{c[0]}_{c[1]}" if isinstance(c, tuple) else str(c)
        for c in cluster_summary.columns
    ]
    cluster_summary = cluster_summary.to_dict(orient="records")

    cluster_size = self.df_model_training["cluster"].value_counts().sort_index()  # FIX: removed duplicate self-assignment

    cluster_data = {
        "cluster_summary": cluster_summary,
        "cluster_size": cluster_size
    }

    return sanitize_for_json(self.hyper_parameter_metrics), sanitize_for_json(cluster_data)