def inspect_outliers(dataframe, column, whisker_width=1.5):
    q1 = dataframe[column].quantile(0.25)
    # mediam = dataframe[column].quantile(0.5)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - whisker_width * iqr
    upper_bound = q3 + whisker_width * iqr

    return dataframe [
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    ]