import pandas as pd
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


def identifier_candidate_model(self, df):

    target_column = self.analysis["problem_info"]["target_column"]

    problem_type = self.analysis["problem_info"]["problem_type"]

    data_size = self.analysis["dataset_info"]["num_rows"]

    method = self.analysis["baseline_model"]["method"]

    def identify_linearity():

        if method != "regression":
            return None

        if target_column is None:
            return None

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        if len(numeric_cols) == 0:
            return None

        correlation = df[numeric_cols].corrwith(df[target_column]).abs()

        max_corr = correlation.max()

        return True if max_corr >= 0.7 else False

    def identify_noise():

        outliers = self.recommendation["preprocessing"]["outliers"]

        if len(outliers) == 0:
            return "Low"

        total = 0

        for col in outliers:

            total += outliers[col]["percentage"]

        avg_noise = total / len(outliers)

        if avg_noise < 5:
            return "Low"

        elif avg_noise < 15:
            return "Medium"

        else:
            return "High"

    def identify_dimension():

        rows, cols = df.shape

        ratio = cols / rows

        if cols > 100:
            return True

        if ratio > 0.1:
            return True

        return False


    def identify_imbalance():

        if method != "classification":

            return None

        counts = df[target_column].value_counts(normalize=True)

        minority = counts.min()

        majority = counts.max()

        if minority == 0:

            return "Severe"

        ratio = majority / minority

        if ratio < 1.5:

            return "Balanced"

        elif ratio < 3:

            return "Moderate"

        elif ratio == 0:

            return "Severe"

        else:

            return "Severe"

    candidate_model_info = {

        "problem_type": problem_type,

        "method" : method,

        "linearity": identify_linearity(),

        "dataset_size": data_size,

        "noise_level": identify_noise(),

        "high_dimension": identify_dimension(),

        "class_imbalance": identify_imbalance()
    }

    return candidate_model_selector(self, candidate_model_info)

def candidate_model_selector(self, analysis):

    method = analysis["method"].lower()

    linearity = analysis["linearity"]

    dataset_size = analysis["dataset_size"]

    noise = analysis["noise_level"]

    high_dimension = analysis["high_dimension"]

    imbalance = analysis["class_imbalance"]

    candidate_models = {}

    if method == "regression":

        if linearity:

            if high_dimension:

                candidate_models = {

                    "Ridge": Ridge(),

                    "Lasso": Lasso(),

                    "ElasticNet": ElasticNet()

                }

            else:

                candidate_models = {

                    "Linear Regression": LinearRegression(),

                    "Ridge": Ridge(),

                    "Lasso": Lasso()

                }

        else:

            if dataset_size < 20000:

                candidate_models = {

                    "Decision Tree": DecisionTreeRegressor(random_state=42),

                    "Random Forest": RandomForestRegressor(random_state=42),

                    "Extra Trees": ExtraTreesRegressor(random_state=42)

                }

            elif dataset_size < 100000:

                candidate_models = {

                    "Random Forest": RandomForestRegressor(random_state=42),

                    "Extra Trees": ExtraTreesRegressor(random_state=42),

                    "Bagging": BaggingRegressor(random_state=42)

                }

            else:

                candidate_models = {

                    "XGBoost": XGBRegressor(),

                    # "LightGBM": LGBMRegressor(),

                    "CatBoost": CatBoostRegressor()

                }

        if noise == "High":

            candidate_models.update({

                "Random Forest": RandomForestRegressor(),

                "Extra Trees": ExtraTreesRegressor(),

                "CatBoost": CatBoostRegressor()

            })

    elif method == "classification":

        if linearity:

            candidate_models = {

                "Logistic Regression": LogisticRegression(),

                "Linear SVC": LinearSVC(),

                "Gaussian NB": GaussianNB()

            }

        else:

            if dataset_size < 10000:

                candidate_models = {

                    "Decision Tree": DecisionTreeClassifier(),

                    "Random Forest": RandomForestClassifier(),

                    "Extra Trees": ExtraTreesClassifier()

                }

            else:

                candidate_models = {

                    "XGBoost": XGBClassifier(),

                    # "LightGBM": LGBMClassifier(),

                    "CatBoost": CatBoostClassifier()

                }

        if imbalance in ["Moderate", "Severe"]:

            candidate_models.update({

                "Balanced Random Forest": BalancedRandomForestClassifier(),

                "XGBoost": XGBClassifier(),

                "CatBoost": CatBoostClassifier()

            })

    elif method == "clustering":

        if dataset_size < 10000:

            candidate_models = {

                "KMeans": KMeans(
                    random_state=42,
                    n_init=10
                ),

                "Agglomerative Clustering": AgglomerativeClustering(),

                "Gaussian Mixture": GaussianMixture()

            }

        elif dataset_size < 100000:

            candidate_models = {

                "MiniBatch KMeans": MiniBatchKMeans(),

                "Birch": Birch(),

                "KMeans": KMeans(
                    random_state=42,
                    n_init=10
                )

            }

        else:

            candidate_models = {

                "MiniBatch KMeans": MiniBatchKMeans(
                    random_state=42,
                    batch_size=2048
                ),

                "Birch": Birch(),

                "DBSCAN": DBSCAN()

            }

        if noise == "High":

            candidate_models.update({

                "DBSCAN": DBSCAN(),

                "OPTICS": OPTICS()

            })

    elif method == "anomaly_detection":

        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.svm import OneClassSVM

        candidate_models = {

            "Isolation Forest": IsolationForest(),

            "Local Outlier Factor": LocalOutlierFactor(
                n_neighbors=20
            ),

            "One-Class SVM": OneClassSVM()

        }

    elif method == "dimensionality_reduction":

        from sklearn.decomposition import (
            PCA,
            KernelPCA,
            TruncatedSVD,
            FastICA
        )

        candidate_models = {

            "PCA": PCA(
                random_state=42
            ),

            "Kernel PCA": KernelPCA(),

            "Truncated SVD": TruncatedSVD(
                random_state=42
            ),

            "Fast ICA": FastICA(
                random_state=42
            )

        }

    return candidate_models