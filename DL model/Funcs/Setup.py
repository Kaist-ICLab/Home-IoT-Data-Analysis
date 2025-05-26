import pandas as pd
import cloudpickle
from datetime import datetime
import numpy as np
import pytz
from sklearn.feature_selection import VarianceThreshold

# def binarize_by_user(data, threshold = 2, personalized = False, uids = None):
#     if personalized:
#         tmp = pd.DataFrame(dict(zip(['label', 'uid'], [data, uids])), index=None)
#         mean_values = tmp.groupby('uid')['label'].mean().to_dict()
#         def binarize(row):
#             uid = row['uid']
#             value = row['label']
#             mean_value = mean_values[uid]
#             if value >= mean_value:
#                 return 1
#             else:
#                 return 0
#         data = tmp.apply(binarize, axis=1)
#     else:
#         data = (data > threshold).astype(int)
#     return data

def binarize_by_user(df):
    labels = ['phq2_result', 'gad2_result', 'stress_result', 'posNeg_result', 'arousal_result']
    
    for label in labels:
        mean_values = df.groupby('uid')[label].mean().to_dict()
        
        def binarize(row, label=label):
            uid = row['uid']
            value = row[label]
            mean_value = mean_values[uid]
            return 1 if value > mean_value else 0
        
        df[f'{label}_binary'] = df.apply(binarize, axis=1)
    return df


# def binarize_by_range(df):
#     df['phq2_result_binary'] = df['phq2_result'].apply(lambda x: 0 if x == 0 else 1)
#     df['gad2_result_binary'] = df['gad2_result'].apply(lambda x: 0 if x == 0 else 1)
#     df['stress_result_binary'] = df['stress_result'].apply(lambda x: 0 if x <= 2 else 1)
#     df['posNeg_result_binary'] = df['posNeg_result'].apply(lambda x: 0 if x <= 2 else 1)
#     df['arousal_result_binary'] = df['arousal_result'].apply(lambda x: 0 if x <= 2 else 1)

#     return df

def binarize_by_range(df):
    df['phq2_result_binary'] = df['phq2_result'].apply(lambda x: 0 if x < 2 else 1)
    df['gad2_result_binary'] = df['gad2_result'].apply(lambda x: 0 if x < 2 else 1)
    df['stress_result_binary'] = df['stress_result'].apply(lambda x: 0 if x <= 2 else 1)
    df['posNeg_result_binary'] = df['posNeg_result'].apply(lambda x: 0 if x <= 2 else 1)
    df['arousal_result_binary'] = df['arousal_result'].apply(lambda x: 0 if x <= 2 else 1)
#    df['phq4_result_binary'] = df['phq4_result'].apply(lambda x: 0 if x < 2 else 1)
    return df


def load(path: str):
    with open(path, mode='rb') as f:
        return cloudpickle.load(f)


def to_unix_timestamp(time_value):
    if isinstance(time_value, str):
        dt = datetime.strptime(time_value, '%Y-%m-%d %H:%M:%S.%f%z')
    elif isinstance(time_value, pd.Timestamp):
        dt = time_value.to_pydatetime()
    else:
        raise ValueError("Unsupported data type")

    dt_utc = dt.astimezone(pytz.utc)

    return int(dt_utc.timestamp() * 1000)



def filter_high_correlation(data, threshold=0.6):
    """
    Removes features that have a pairwise correlation higher than the specified threshold.
    
    Args:
        data (pd.DataFrame): The input dataset with features.
        threshold (float): The correlation threshold to identify highly correlated features.
        
    Returns:
        pd.DataFrame: The reduced dataset with high correlation features removed.
        list: The list of features that were dropped.
    """
    # Calculate the correlation matrix
    corr_matrix = data.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return to_drop


def filter_zero_variance(data):
    """
    Removes features with zero variance.
    
    Args:
        data (pd.DataFrame): The input dataset with features.
        
    Returns:
        pd.DataFrame: The reduced dataset with zero variance features removed.
    """
    selector = VarianceThreshold()
    reduced_data = selector.fit_transform(data)
    return pd.DataFrame(reduced_data, columns=data.columns[selector.get_support()])