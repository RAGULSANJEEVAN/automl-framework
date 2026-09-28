from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler
import numpy as np
from scipy.stats import zscore

def winsorization(column, df):

    df[column] = winsorize(
        df[column],
        limits = [0.05, 0.05]
    )

    return df

def robustScaler(column, df):

    scaler = RobustScaler()

    df[column] = scaler.fit_transform(df[column])

    return df

def logTransform(column, df):

    df[column] = np.log1p(df[column])
    
    return df

def removeRow(column, df, action, data):

    if action == "IQR":

        lower = data.get("lower_bound")

        upper = data.get("upper_bound")

        df = df[(df[column] >= lower) & (df[column] <= upper)]

        return df

    else:

        z = zscore(df[column])

        df = df[abs(z) < 3]

        return df



