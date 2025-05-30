from Funcs.Setup import *
from sklearn.impute import KNNImputer

import random
import os
import numpy as np
import pandas as pd
import torch

import warnings
import json
warnings.filterwarnings(action='ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # True

# Load Dataset

def user_demographic(demo):
    demo = pd.get_dummies(demo, columns=['Gender'])
    selected_columns = ['UID', 'Extroversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness', 'PHQ-9', 'GAD-7', 'PSS', 'GHQ', 'Gender_F', 'Gender_M']
    demo = demo[selected_columns]
    demo = demo.rename(columns={'UID': 'uid'})  # rename UID to uid
    return demo

def get_part_of_day(hour):
    if 5 <= hour < 12:
        return 1 #Morning
    elif 12 <= hour < 17:
        return 2 #Afternoon
    elif 17 <= hour < 21:
        return 3 #Evening
    else:
        return 4 #Night

def load_aqara_data_updated(file_name, hour, yesterday):
    aqara_data = pd.read_csv(file_name, index_col=None)
    suffix = f'_{hour*60}min'
    filtered_columns = [col for col in aqara_data.columns if col.endswith(suffix)]
    # Select the filtered columns along with any essential columns like 'uid', 'startTime'
    essential_columns = ['uid', 'startTime']
    aqara_features = aqara_data[essential_columns + filtered_columns]
    if yesterday == True:
        
        aqara_features = aqara_features.replace(np.nan, 0)
        aqara_features = aqara_features.replace('[None]', np.nan)
    # Identify aqara columns only (after filtering by hour)
    aqara_columns_only = [col for col in aqara_features.columns if col.startswith('aqara_')]

    for col in aqara_columns_only:
        # Convert the column to numeric, setting errors='coerce' will turn invalid parsing into NaN
        aqara_features[col] = pd.to_numeric(aqara_features[col], errors='coerce')
        # Calculate the mean per uid
        mean_per_uid = aqara_features.groupby('uid')[col].transform('mean')
        # Calculate variance and comparison
        aqara_features[f'{col}_deviation'] = aqara_features[col] - mean_per_uid
        aqara_features[f'{col}_comparison'] = (aqara_features[col] > mean_per_uid).astype(int) - (aqara_features[col] < mean_per_uid).astype(int)
            
    if yesterday == False:
        fridge_columns = [col for col in aqara_columns_only if 'fridge_ImmediatePast' in col]
        microwave_columns = [col for col in aqara_columns_only if 'microwave_ImmediatePast' in col]
        cleaner_columns = [col for col in aqara_columns_only if 'cleaner_ImmediatePast' in col]
        washer_columns = [col for col in aqara_columns_only if 'washer_ImmediatePast' in col]
        if fridge_columns and microwave_columns:
            aqara_features[f'aqara_eating_routine_ImmediatePast_{hour*60}min'] = (
                (aqara_features[fridge_columns[0]] >= 1) |
                (aqara_features[microwave_columns[0]] >= 1)
            ).astype(int)
        if cleaner_columns and washer_columns:
            aqara_features[f'aqara_chores_routine_ImmediatePast_{hour*60}min'] = (
                (aqara_features[cleaner_columns[0]] >= 1) |
                (aqara_features[washer_columns[0]] >= 1)
            ).astype(int)
    else:
        fridge_columns = [col for col in aqara_columns_only if 'fridge_yesterday' in col and 'mean' in col]
        microwave_columns = [col for col in aqara_columns_only if 'microwave_yesterday' in col and 'mean' in col]
        cleaner_columns = [col for col in aqara_columns_only if 'cleaner_yesterday' in col and 'mean' in col]
        washer_columns = [col for col in aqara_columns_only if 'washer_yesterday' in col and 'mean' in col]
        
        for fridge_col in fridge_columns:
            for microwave_col in microwave_columns:
                
                if fridge_col.split('fridge_yesterday_')[1] == microwave_col.split('microwave_yesterday_')[1]:
                    common_suffix = fridge_col.split('fridge_yesterday_')[1]
                    new_col_name = f'aqara_eating_routine_yesterday_{common_suffix}'
                    
                    aqara_features[new_col_name] = (
                        (aqara_features[fridge_col] >= 1) |
                        (aqara_features[microwave_col] >= 1)
                    ).astype(int)
                    
        for cleaner_col in cleaner_columns:
            for washer_col in washer_columns:

                if cleaner_col.split('cleaner_yesterday_')[1] == washer_col.split('washer_yesterday_')[1]:
                    common_suffix = cleaner_col.split('cleaner_yesterday_')[1] 
                    new_col_name = f'aqara_chores_routine_yesterday_{common_suffix}'
                    
                    aqara_features[new_col_name] = (
                        (aqara_features[cleaner_col] >= 1) |
                        (aqara_features[washer_col] >= 1)
                    ).astype(int)

    return aqara_features


def load_smartphone_data(file_name):
    smartphone = load(file_name)
    smartphone_feature, _, uids, dates = smartphone

    smartphone_feature = smartphone_feature.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x.astype('float64'), axis=0)

    df_smartphone = pd.concat([smartphone_feature, pd.DataFrame(uids.tolist(), columns=['uid']), pd.DataFrame(dates.tolist(), columns=['timestamp'])], axis=1)
    df_smartphone['uid'] = df_smartphone['uid'].str.replace('P', '').astype('int64')
    df_smartphone['hour'] = df_smartphone['timestamp'].dt.hour
    df_smartphone['part_of_day'] = df_smartphone['hour'].apply(get_part_of_day)
    df_smartphone['timestamp'] = df_smartphone['timestamp'].apply(to_unix_timestamp)

    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='_Today')))]

    return df_smartphone

def load_label_data(file_name):

    label = pd.read_csv(file_name, index_col=None)
    df_label = binarize_by_range(label)

    return df_label


def data_sources_combinations(base_df, data_sources):
    merged_df = base_df.copy()
    for source_df, on_columns, filter_condition, fill_na in data_sources:
        merged_df = pd.merge(merged_df, source_df, on=on_columns, how='left')
       
        if filter_condition is not None:
            merged_df = merged_df.loc[filter_condition(merged_df)]

        if fill_na is not None:
            merged_df.fillna(fill_na, inplace=True)

    return merged_df


def get_dataset(dataset_path='DATASET/df.csv'):
    df_label = load_label_data('FEATURES/label_2023.csv')

    df_audio = pd.read_csv('FEATURES/librosa_features.csv')
    df_speech = pd.read_csv('FEATURES/speech_data_2023.csv', index_col=None) 
    df_bluSensor = pd.read_csv('FEATURES/bluSensor_features_15min.csv', index_col=None)
    df_fitbit = pd.read_csv('FEATURES/fitbit_features_24h.csv', index_col=None)

    # if spliter == 'personalstratifiedkfold', deviation and comparison columns are not calculated
    df_aqara_before = load_aqara_data_updated('FEATURES/aqara_before_1h_3h_6h_12h_and.csv', 1, yesterday=False)
    df_aqara_yesterday = load_aqara_data_updated('FEATURES/aqara_yesterday_12h.csv', 12, yesterday=True)
    
    df_withings = pd.read_csv('FEATURES/withings_features_24h.csv', index_col=None)
    df_pre_survey = pd.read_csv('FEATURES/user_demographics_pre_test_2023.csv', index_col=None)
    df_smartphone = load_smartphone_data('FEATURES/smartphone_features_60min_yesterday_today_impute_median.pkl')
    df_demo = user_demographic(df_pre_survey)

    df_label.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df_audio.rename(columns={'startTime': 'timestamp'}, inplace=True)
    
    df_aqara_before.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df_aqara_yesterday.rename(columns={'startTime': 'timestamp'}, inplace=True)

    df_withings.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df_fitbit.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df_demo.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df_speech.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df_bluSensor.rename(columns={'startTime': 'timestamp'}, inplace=True)


    data_sources = [
        ## DEMOGRAPHIC DATA ##
        # (df_demo, ['uid'], None, None),                                        # demographic

        ## PHONE DATA ##
        (df_smartphone, ['uid', 'timestamp'], None, None),                     # Smartphone 
        
        ## WEARABLE DATA ##
        (df_fitbit, ['uid', 'timestamp'], None, 0),                            # Fitbit 

        ## IOT DATA ##
        (df_aqara_before, ['uid', 'timestamp'], None, 0),                      # aqara
        (df_aqara_yesterday, ['uid', 'timestamp'], None, None),                # aqara
        
        (df_withings, ['uid', 'timestamp'], None, 0),                          # withings
        (df_bluSensor, ['uid', 'timestamp'], None, None),                      # BluSensor

        ## AUDIO DATA ##
        (df_audio, ['uid', 'timestamp'], lambda df: df['duration'] > 0, None), # Audio
    ]

    base_df = df_label.copy()

    merged_df = data_sources_combinations(base_df, data_sources)

    # Missing value handling using KNNImputer
    merged_df.columns = merged_df.columns.astype(str)

    unnamed_columns = [col for col in merged_df.columns if 'Unnamed' in col]
    merged_df = merged_df.drop(columns=unnamed_columns)

    merged_df = merged_df.set_index('timestamp').sort_index()

    # if 'chromagram_1' not in merged_df.columns:
    #     merged_df = pd.merge(merged_df, df_audio[df_audio['duration'] > 0], on=['uid', 'timestamp'], how='left').dropna().iloc[:,:-182]

    df_info_dict = {
        'shape': merged_df.shape,
        'binarize_option': 'custom_range'
    }
    for i, (source_df, on_columns, _, _) in enumerate(data_sources):

        info_dict = {
            'source': i+1,
            'columns': source_df.columns.tolist()[len(on_columns):],
            'N_columns': len(source_df.columns.tolist()[len(on_columns):])
        }
        df_info_dict[f'source_{i+1}'] = info_dict

    with open(f'{dataset_path[:-4]}.json', 'w') as json_file:
        json.dump(df_info_dict, json_file, indent=4)
    merged_df.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")
    return merged_df

def remove_user_with_skewed_label(data, label):
    label_ratios = data[data[label] == 0].groupby('uid').size() / data.groupby('uid').size()
    label_ratios = label_ratios.fillna(0)

    users_to_remove = label_ratios[(label_ratios < 0.1) | (label_ratios > 0.9)].index

    return users_to_remove

def get_ablation_dataset(df, ablation_data):
    main_cols = [
        'uid',
        'phq2_result', 'gad2_result', 'stress_result', 'posNeg_result', 'arousal_result',
        'phq2_result_binary', 'gad2_result_binary', 'stress_result_binary', 'posNeg_result_binary', 'arousal_result_binary'
    ]

    feature_patterns = {
        'IoT': 'aqara|withings|bluSensor',
        'Voice': r'\bduration\b|word_count|chromagram|melspectrogram|mfcc',
        'Wearable': 'fitbit',
        'IoTVoice': 'aqara|withings|bluSensor|\bduration\b|word_count|chromagram|melspectrogram|mfcc',
    }

    if ablation_data == 'All':
        return df

    elif ablation_data in feature_patterns:
        selected = list(df.columns[df.columns.str.contains(feature_patterns[ablation_data])])
        return df.loc[:, main_cols + selected]

    elif ablation_data == 'Phone':
        exclude_patterns = '|'.join([feature_patterns[k] for k in ['IoT', 'Voice', 'Wearable']])
        excluded_cols = df.columns[df.columns.str.contains(exclude_patterns)]
        selected = [col for col in df.columns if col not in excluded_cols]
        return df.loc[:, selected]

    elif ablation_data == 'PhoneWearable':
        exclude_patterns = '|'.join([feature_patterns[k] for k in ['IoT', 'Voice']])
        excluded_cols = df.columns[df.columns.str.contains(exclude_patterns)]
        selected = [col for col in df.columns if col not in excluded_cols]
        return df.loc[:, selected]

    else:
        raise ValueError(f"Unknown ablation data type: {ablation_data}")