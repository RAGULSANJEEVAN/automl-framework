from sklearn.preprocessing import StandardScaler
import pandas as pd
def handling_supervised_feature_scaling(self, splitData):

    X_train, X_test, y_train, y_test = splitData

    train_columns = X_train.columns

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns= train_columns, index=y_train.index)

    X_test = pd.DataFrame(X_test, columns= train_columns, index=y_test.index)

    return X_train, X_test, y_train, y_test

def handling_unsupervised_feature_scaling(self, splitData):

    X = splitData

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X_scaled
