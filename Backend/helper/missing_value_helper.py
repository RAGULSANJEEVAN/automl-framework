from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def drop_column(column, df):

    df = df.drop(columns=[column])

    return df


def mean_imputation(column, df):

    df[column] = df[column].fillna(df[column].mean())

    return df


def median_imputation(column, df):

    df[column] = df[column].fillna(df[column].median())

    return df


def mode_imputation(column, df):

    if not df[column].mode().empty:

        df[column] = df[column].fillna(df[column].mode()[0])

    return df


def forward_imputation(column, df):

    df[column] = df[column].ffill()

    return df


def knn_imputation(df):

    numeric_cols = df.select_dtypes(include="number").columns

    imputer = KNNImputer(n_neighbors=5)

    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df


def iterative_imputation(df):

    numeric_cols = df.select_dtypes(include="number").columns

    imputer = IterativeImputer(random_state=42)

    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df