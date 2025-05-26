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

    # cudnn random seed를 고정할 경우 학습 속도가 느려질 수 있음
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
    
# Calculating variance for each 'aqara_' column based on uid
def load_aqara_data(file_name):
    aqara_features = pd.read_csv(file_name, index_col=None)
    aqara_columns_only = [col for col in aqara_features.columns if col.startswith('aqara_')]

    for col in aqara_columns_only:
        mean_per_uid = aqara_features.groupby('uid')[col].transform('mean')
        aqara_features[f'{col}_variance'] = aqara_features[col] - mean_per_uid
        aqara_features[f'{col}_comparison'] = (aqara_features[col] > mean_per_uid).astype(int) - (aqara_features[col] < mean_per_uid).astype(int)

    aqara_features['eating_routine'] = ((aqara_features['aqara_fridge'] >= 1) | (aqara_features['aqara_microwave'] >= 1)).astype(int)
    aqara_features['chores_routine'] = ((aqara_features['aqara_cleaner'] >= 1) | (aqara_features['aqara_washer'] >= 1)).astype(int)
    return aqara_features

def load_aqara_data_updated(file_name, hour, yesterday):
    aqara_data = pd.read_csv(file_name, index_col=None)
    suffix = f'_{hour*60}min'
    filtered_columns = [col for col in aqara_data.columns if col.endswith(suffix)]
    # Select the filtered columns along with any essential columns like 'uid', 'startTime'
    essential_columns = ['uid', 'startTime']
    aqara_features = aqara_data[essential_columns + filtered_columns]
    if yesterday == True:
        # 각 행에서 '[None]' 인 열이 하나라도 있으면, 그 행 제거
        aqara_features = aqara_features.replace(np.nan, 0)
        aqara_features = aqara_features.replace('[None]', np.nan)
    # Identify aqara columns only (after filtering by hour)
    aqara_columns_only = [col for col in aqara_features.columns if col.startswith('aqara_')]

    # 2024-11-05 personalstratifiedkfold에서 aqara feature 수정
    # if splitter == 'personalstratifiedkfold':
    # for col in aqara_columns_only:
    #     # Convert the column to numeric, setting errors='coerce' will turn invalid parsing into NaN
    #     aqara_features[col] = pd.to_numeric(aqara_features[col], errors='coerce')
    #     # Calculate the mean per uid
    #     mean_per_uid = aqara_features.groupby('uid')[col].transform('mean')
    #     # Calculate variance and comparison
    #     aqara_features[f'{col}_deviation'] = aqara_features[col] - mean_per_uid
    #     aqara_features[f'{col}_comparison'] = (aqara_features[col] > mean_per_uid).astype(int) - (aqara_features[col] < mean_per_uid).astype(int)


    # Calculate routines based on filtered columns
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
        # 공통 시간대 접미사를 찾고 새로운 열을 생성
        for fridge_col in fridge_columns:
            for microwave_col in microwave_columns:
                # 공통 시간대 접미사를 확인
                if fridge_col.split('fridge_yesterday_')[1] == microwave_col.split('microwave_yesterday_')[1]:
                    common_suffix = fridge_col.split('fridge_yesterday_')[1]  # 공통 접미사 추출
                    new_col_name = f'aqara_eating_routine_yesterday_{common_suffix}'
                    # 새로운 열을 생성하고 값을 설정
                    aqara_features[new_col_name] = (
                        (aqara_features[fridge_col] >= 1) |
                        (aqara_features[microwave_col] >= 1)
                    ).astype(int)
        # 공통 시간대 접미사를 찾고 새로운 열을 생성 (chores routine)
        for cleaner_col in cleaner_columns:
            for washer_col in washer_columns:
                # 공통 시간대 접미사를 확인
                if cleaner_col.split('cleaner_yesterday_')[1] == washer_col.split('washer_yesterday_')[1]:
                    common_suffix = cleaner_col.split('cleaner_yesterday_')[1]  # 공통 접미사 추출
                    new_col_name = f'aqara_chores_routine_yesterday_{common_suffix}'
                    # 새로운 열을 생성하고 값을 설정
                    aqara_features[new_col_name] = (
                        (aqara_features[cleaner_col] >= 1) |
                        (aqara_features[washer_col] >= 1)
                    ).astype(int)
    return aqara_features


# def load_smartphone_data(file_name):
#     smartphone = load(file_name)
#     smartphone_feature, _, uids, dates = smartphone

#     smartphone_feature = smartphone_feature.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x.astype('float64'), axis=0)

#     df_smartphone = pd.concat([smartphone_feature, pd.DataFrame(uids.tolist(), columns=['uid']), pd.DataFrame(dates.tolist(), columns=['timestamp'])], axis=1)
#     df_smartphone['uid'] = df_smartphone['uid'].str.replace('P', '').astype('int64')
#     df_smartphone['hour'] = df_smartphone['timestamp'].dt.hour
#     df_smartphone['part_of_day'] = df_smartphone['hour'].apply(get_part_of_day)
#     df_smartphone['timestamp'] = df_smartphone['timestamp'].apply(to_unix_timestamp)

#     df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='_Today')))]

#     return df_smartphone

def load_smartphone_data(file_name):
    smartphone = load(file_name)
    smartphone_feature, _, uids, dates = smartphone
    smartphone_feature = smartphone_feature.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x.astype('float64'), axis=0)
    df_smartphone = pd.concat([smartphone_feature, pd.DataFrame(uids.tolist(), columns=['uid']), pd.DataFrame(dates.tolist(), columns=['timestamp'])], axis=1)
    df_smartphone['uid'] = df_smartphone['uid'].str.replace('P', '').astype('int64')
    df_smartphone['hour'] = df_smartphone['timestamp'].dt.hour
    df_smartphone['part_of_day'] = df_smartphone['hour'].apply(get_part_of_day)
    df_smartphone['timestamp'] = df_smartphone['timestamp'].apply(to_unix_timestamp)
    # df_smartphone 의 LOC_LABEL#RLV_SUP=eating#ImmediatePast_60,LOC_LABEL#RLV_SUP=home#ImmediatePast_60,LOC_LABEL#RLV_SUP=work#ImmediatePast_60,LOC_LABEL#RLV_SUP=social#ImmediatePast_60,LOC_LABEL#RLV_SUP=others#ImmediatePast_60 열 중 LOC_LABEL#RLV_SUP=eating#ImmediatePast_60 이 가장 큰 값을 가지는 경우 1, 아니면 0
    df_smartphone['eating_routine_immediatePast_60'] = (
        df_smartphone['LOC_LABEL#RLV_SUP=eating#ImmediatePast_60'] == df_smartphone[
            [
                'LOC_LABEL#RLV_SUP=eating#ImmediatePast_60',
                'LOC_LABEL#RLV_SUP=home#ImmediatePast_60',
                'LOC_LABEL#RLV_SUP=work#ImmediatePast_60',
                'LOC_LABEL#RLV_SUP=social#ImmediatePast_60',
                'LOC_LABEL#RLV_SUP=others#ImmediatePast_60'
            ]
        ].max(axis=1)
        ).astype(int)
    df_smartphone['eating_routine_YesterdayDawn'] = (
        df_smartphone['LOC_LABEL#RLV_SUP=eating#YesterdayDawn'] == df_smartphone[
            [
                'LOC_LABEL#RLV_SUP=eating#YesterdayDawn',
                'LOC_LABEL#RLV_SUP=home#YesterdayDawn',
                'LOC_LABEL#RLV_SUP=work#YesterdayDawn',
                'LOC_LABEL#RLV_SUP=social#YesterdayDawn',
                'LOC_LABEL#RLV_SUP=others#YesterdayDawn'
            ]
        ].max(axis=1)
        ).astype(int)
    df_smartphone['eating_routine_YesterdayMorning'] = (
        df_smartphone['LOC_LABEL#RLV_SUP=eating#YesterdayMorning'] == df_smartphone[
            [
                'LOC_LABEL#RLV_SUP=eating#YesterdayMorning',
                'LOC_LABEL#RLV_SUP=home#YesterdayMorning',
                'LOC_LABEL#RLV_SUP=work#YesterdayMorning',
                'LOC_LABEL#RLV_SUP=social#YesterdayMorning',
                'LOC_LABEL#RLV_SUP=others#YesterdayMorning'
            ]
        ].max(axis=1)
        ).astype(int)
    df_smartphone['eating_routine_YesterdayAfternoon'] = (
        df_smartphone['LOC_LABEL#RLV_SUP=eating#YesterdayAfternoon'] == df_smartphone[
            [
                'LOC_LABEL#RLV_SUP=eating#YesterdayAfternoon',
                'LOC_LABEL#RLV_SUP=home#YesterdayAfternoon',
                'LOC_LABEL#RLV_SUP=work#YesterdayAfternoon',
                'LOC_LABEL#RLV_SUP=social#YesterdayAfternoon',
                'LOC_LABEL#RLV_SUP=others#YesterdayAfternoon'
            ]
        ].max(axis=1)
        ).astype(int)
    df_smartphone['eating_routine_YesterdayEvening'] = (
        df_smartphone['LOC_LABEL#RLV_SUP=eating#YesterdayEvening'] == df_smartphone[
            [
                'LOC_LABEL#RLV_SUP=eating#YesterdayEvening',
                'LOC_LABEL#RLV_SUP=home#YesterdayEvening',
                'LOC_LABEL#RLV_SUP=work#YesterdayEvening',
                'LOC_LABEL#RLV_SUP=social#YesterdayEvening',
                'LOC_LABEL#RLV_SUP=others#YesterdayEvening'
            ]
        ].max(axis=1)
        ).astype(int)
    df_smartphone['eating_routine_YesterdayNight'] = (
        df_smartphone['LOC_LABEL#RLV_SUP=eating#YesterdayNight'] == df_smartphone[
            [
                'LOC_LABEL#RLV_SUP=eating#YesterdayNight',
                'LOC_LABEL#RLV_SUP=home#YesterdayNight',
                'LOC_LABEL#RLV_SUP=work#YesterdayNight',
                'LOC_LABEL#RLV_SUP=social#YesterdayNight',
                'LOC_LABEL#RLV_SUP=others#YesterdayNight'
            ]
        ].max(axis=1)
        ).astype(int)
    # Today Epoch feature removed
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='_Today')))]
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='#SKW')))]
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='#KUR')))]
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='#BEP')))]
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='#TSC')))]
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='#ETP')))]
    # df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='#ASC')))]
    # df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='#VAL')))]
    # df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='ACE')))]
    # df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='ACT')))]
    # df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='APP_DUR')))]
    # df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='APP_CAT')))]
    # df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='MSG')))]
    # df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='LOC')))]
    # 'FCL' FitnessCalorie, 'FDI' FitnessDistance, 'FST' FitnessStepCount, 'FAC' FitnessActivity
    # df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='FST')))]
    # df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='FAC')))]
    # df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='FCL')))]
    # df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='FDI')))]
    # df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='CAE#FREQ')))]
### 우울과 관련 없어보이는 features
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='LOC_CLS#DSC')))]
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='LOC_LABEL#DSC')))]
    # Battery.csv
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='BAT_STA')))]
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='BAT_LEV')))]
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='BAT_TMP')))]
    # DataTraffic.csv
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='CON')))]
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='DAT')))]
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='WIF')))]
    # InstalledApp.csv
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='INS_JAC')))]
    # MediaEvent.csv
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='MED_VID')))]
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='MED_IMG')))]
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='MED_ALL')))]
    # RingerModeEvent.csv
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='RNG')))]
    # Notification.csv
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='NOT')))]
    # PowerSaveEvent.csv
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='PWS')))]
    # PhoneStateEvent.csv
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='PHS')))]
    # 'BTS': BatteryState,
    df_smartphone = df_smartphone[df_smartphone.columns.drop(list(df_smartphone.filter(regex='BTS')))]
    return df_smartphone

def load_label_data(file_name):

    label = pd.read_csv(file_name, index_col=None)
    df_label = binarize_by_range(label)
    # df_label = binarize_by_user(label)

    return df_label

######여기부터 일부 변경

def data_sources_combinations(base_df, data_sources):
    merged_df = base_df.copy()
    for source_df, on_columns, filter_condition, fill_na in data_sources:
        merged_df = pd.merge(merged_df, source_df, on=on_columns, how='left')
       
        if filter_condition is not None:
            merged_df = merged_df.loc[filter_condition(merged_df)]

        if fill_na is not None:
            merged_df.fillna(fill_na, inplace=True)

    return merged_df


def get_dataset(dataset='All', dataset_path='DATASET_04/All'):
    df_label = load_label_data('FEATURES/label_2023.csv')
    # df_audio = pd.read_pickle('FEATURES/librosa_modified.pkl')
    df_audio = pd.read_csv('FEATURES/librosa_modified.csv')
    df_speech = pd.read_csv('FEATURES/speech_data_2023.csv', index_col=None) 
    df_bluSensor = pd.read_csv('FEATURES/bluSensor_features_15min.csv', index_col=None)
    df_fitbit = pd.read_csv('FEATURES/fitbit_features_24h.csv', index_col=None)
    # df_aqara = load_aqara_data('FEATURES/aqara_features_12h.csv')
    df_aqara_before = load_aqara_data_updated('FEATURES/aqara_before_1h_3h_6h_12h_and.csv', 1, yesterday=False)
    df_aqara_yesterday = load_aqara_data_updated('FEATURES/aqara_yesterday_12h.csv', 12, yesterday=True)
    
    df_withings = pd.read_csv('FEATURES/withings_features_24h.csv', index_col=None)
    df_pre_survey = pd.read_csv('FEATURES/user_demographics_pre_test_2023.csv', index_col=None)
    df_smartphone = load_smartphone_data('FEATURES/smartphone_features_60min_yesterday_today_impute_median.pkl')
    df_demo = user_demographic(df_pre_survey)

    df_label.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df_audio.rename(columns={'startTime': 'timestamp'}, inplace=True)
    # df_aqara.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df_aqara_before.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df_aqara_yesterday.rename(columns={'startTime': 'timestamp'}, inplace=True)

    df_withings.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df_fitbit.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df_demo.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df_speech.rename(columns={'startTime': 'timestamp'}, inplace=True)
    df_bluSensor.rename(columns={'startTime': 'timestamp'}, inplace=True)


    data_sources = [
        ## DEMOGRAPHIC DATA ##
        # (df_demo, ['uid'], None, None),                                        # 데모그래픽 데이터

        ## PHONE DATA ##
        # (df_smartphone, ['uid', 'timestamp'], None, None),                     # Smartphone 데이터
        
        ## WEARABLE DATA ##
        # (df_fitbit, ['uid', 'timestamp'], None, None),                         # Fitbit 데이터
        # (df_fitbit, ['uid', 'timestamp'], None, 0),                         # Fitbit 데이터 2024-11-05 수정: missing 0으로 채움

        ## IOT DATA ##
        # (df_aqara, ['uid', 'timestamp'], None, 0),                             # aqara 센서 데이터
        (df_aqara_before, ['uid', 'timestamp'], None, 0),       # aqara 센서 데이터
        (df_aqara_yesterday, ['uid', 'timestamp'], None, None),       # aqara 센서 데이터
        
        (df_withings, ['uid', 'timestamp'], None, 0),                          # withings 센서 데이터
        (df_bluSensor, ['uid', 'timestamp'], None, None),                      # 블루센서 데이터 

        # ## AUDIO DATA ##
        # # (df_speech, ['uid', 'timestamp'], None, None),                         # speech rate
        
        (df_audio, ['uid', 'timestamp'], lambda df: df['duration'] > 0, None), # Audio 데이터, 조건 적용
    ]

    base_df = df_label.copy()

    merged_df = data_sources_combinations(base_df, data_sources)

    # Missing value handling using KNNImputer
    merged_df.columns = merged_df.columns.astype(str)

    unnamed_columns = [col for col in merged_df.columns if 'Unnamed' in col]
    merged_df = merged_df.drop(columns=unnamed_columns)

    # 2024-11-05 추가: 데이터 소스 합친 후에 timestamp 기준으로 정렬 및 기존 Imputer 코드 주석 처리
    merged_df = merged_df.set_index('timestamp').sort_index()

    # numeric_cols = merged_df.select_dtypes(include=['float64', 'int']).columns
    # uid_col = merged_df['uid']  # Assuming 'uid' is the non-numeric column causing issues

    # # Apply KNNImputer only to numeric columns 
    # knn_imputer = KNNImputer(n_neighbors=20)
    # df_imputed = knn_imputer.fit_transform(merged_df[numeric_cols])

    # # Replace the numeric columns in the original df with imputed values
    # merged_df[numeric_cols] = df_imputed
    # merged_df = merged_df.set_index('timestamp').sort_index()

    # 2024-11-05 추가 완료

    if 'chromagram_1' not in merged_df.columns:
        merged_df = pd.merge(merged_df, df_audio[df_audio['duration'] > 0], on=['uid', 'timestamp'], how='left').dropna().iloc[:,:-182]

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

    with open(f'{dataset_path}.json', 'w') as json_file:
        json.dump(df_info_dict, json_file, indent=4)
    merged_df.to_csv(f'{dataset_path}.csv', index=False)
    return merged_df

def remove_user_with_skewed_label(data, label):
    # 'phq_result'가 0인 비율 계산
    label_ratios = data[data[label] == 0].groupby('uid').size() / data.groupby('uid').size()
    label_ratios = label_ratios.fillna(0)

    users_to_remove = label_ratios[(label_ratios < 0.1) | (label_ratios > 0.9)].index
    # print(label_ratios)  

    return users_to_remove
