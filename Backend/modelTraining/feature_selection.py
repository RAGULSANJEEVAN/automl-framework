from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import pandas as pd
def supervised_learning_feature_selection(self, splitData):

    feature_selection_info = {}

    method = self.analysis["baseline_model"]["method"]

    X_train, X_test, y_train, y_test = splitData

    num_X_col = X_train.select_dtypes(include = ["int64","float64"]).columns

    X_train_count = len(num_X_col)

    important_feature_count = int(X_train_count*0.7)

    if method == "classification":

        estimator = RandomForestClassifier(random_state=42)
    else:

        estimator = RandomForestRegressor(random_state=42)

    selector = RFE( estimator=estimator, n_features_to_select=important_feature_count)

    selector.fit(X_train,y_train)

    selected_features = X_train.columns[
        selector.support_
    ]
    X_train = X_train[selected_features]

    X_test = X_test[selected_features]

    feature_selection_info = {

        "important_features_count" : important_feature_count,

        "imporrtant_features" : selected_features,

    }
    

    return (X_train, X_test, y_train, y_test), feature_selection_info

def unsupervised_learning_feature_selection(self, splitData):

    feature_selection_info = {}

    X_scaled = splitData

    if X_scaled.shape[1] > 10:

        X_pca = apply_pca(X_scaled)

        X_pca = pd.DataFrame(
            X_pca,
            columns=[f"PC{i+1}" for i in range(X_pca.shape[1])],
            index=X_scaled.index
        )

        feature_selection_info = {
            
        "important_features_count" : X_pca.shape[1],

        "imporrtant_features" : X_pca.columns,
        }

        return X_pca, feature_selection_info

    else:

        return X_scaled, feature_selection_info

def apply_pca(X_scaled):

    pca = PCA(n_components=0.95)

    X_pca = pca.fit_transform(X_scaled)

    return X_pca

    