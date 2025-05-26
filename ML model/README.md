## Getting Started
Install the following libraries
- Python==3.9
- NumPy
- pandas==2.1.1
- scikit-learn
- XGBoost==1.6.1
- imbalanced-learn
- TQDM
- Python Dateutil
- Pytz

## Running the Code
- In the **FEATURES** folder, features are stored in the form of "feature name_window size"
- For example, "smartphone_features_60min_yesterday_today_impute_midean.pkl" includes 60 min immediately before the label, yesterday, and today epoch features

### Data Sources
You can use feature combination by uncommenting/commenting the code. The data sources and their configurations are listed below:
```python
  data_sources = [
      (df_smartphone, ['uid', 'timestamp'], None, None),   # Smartphone data
      (df_audio, ['uid', 'timestamp'], lambda df: df['duration'] > 0, None), # Audio data with condition
      (df_aqara, ['uid', 'timestamp'], None, 0),       # Aqara sensor data
      (df_withings, ['uid', 'timestamp'], None, 0),    # Withings sensor data
      (df_fitbit, ['uid', 'timestamp'], None, None),   # Fitbit data
      (df_demo, ['uid'], None, None),                  # Demographic data
      (df_speech, ['uid', 'timestamp'], None, None),   # Speech data
      (df_bluSensor, ['uid', 'timestamp'], None, None) # BluSensor data 
  ]
```
### Dataset Splitting
You can split the dataset using either LOSO (Leave-One-Subject-Out) or K-Fold. Choose one of the following methods:

#### LOSO (Leave-One-Subject-Out)
```python           
  # LOSO 
  res = classify(df, label, model, zero=zero_variance, high_pariwise=high_pariwise, lasso=lasso,
      oversample=oversample,
      sampling_method='auto', correlation_threshold = 0.95)
  table = pd.DataFrame(res, columns=['uid', 'auc', 'accuracy', 'f1_macro', 'f1_weighted'])
```

#### K-Fold
```python
  # K-Fold
  res = classify_k_fold(df, label, model, zero=zero_variance, high_pariwise=high_pariwise, lasso=lasso,
      oversample=oversample,
      sampling_method='auto', correlation_threshold = 0.95)
  table = pd.DataFrame(res, columns=['auc', 'accuracy', 'f1_macro', 'f1_weighted'])
  ```