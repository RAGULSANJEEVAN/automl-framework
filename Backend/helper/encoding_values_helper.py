from sklearn.preprocessing import LabelEncoder
import pandas as pd
from category_encoders import HashingEncoder

def one_hot_encoding(column, df):

    df = pd.get_dummies(
        df,
        columns= [column],
        dtype = int
    )

    return df

def frequency_encoding(column, df):

    freq = df[column].value_counts()

    df[column] = df[column].map(freq)

    return df

def target_encoding(column, df, target_column):

    target = df.groupby(column)[target_column].mean()

    df[column] = df[column].map(target)

    return df

def hash_encoding(column, df):

    encoder = HashingEncoder(cols=[column])

    df = encoder.fit_transform(df)

    return df

def label_encoding(column, df):

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    return df