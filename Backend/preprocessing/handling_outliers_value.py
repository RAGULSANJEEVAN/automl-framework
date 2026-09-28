from helper.outlier_handling_helper import (removeRow, winsorization, logTransform, robustScaler)
def handel_outlier_value(self, df):

    outliers_info = self.recommender["preprocessing"]["outliers"]

    if not outliers_info:

        return df

    for column , details in outliers_info.items():

        distribution = details.get("distribution")

        count = details.get("count")

        percentage = details.get("percentage")

        if percentage == 0:

            return df

        elif percentage <= 5:

            data = details.get("data",{})

            action = data.get("method")

            df = removeRow(column, df, action, data)

        elif percentage <15:

            df = winsorization(column, df)

        elif percentage < 30 and distribution == "not_normal":

            df = logTransform(column, df)

        elif percentage < 30 and distribution == "normal":

            df = robustScaler(column, df)

        elif percentage > 30:

            df = robustScaler(column, df)

        else:

            print("No Outliers to handle")

    return df