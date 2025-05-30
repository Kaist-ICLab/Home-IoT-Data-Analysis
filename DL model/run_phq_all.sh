#!/bin/bash

# General - LOSO
python main.py --splitter loso --modelname fusion --save_dir RESULTS/General/PHQ_All --label phq2_result_binary --mixup
python main.py --splitter loso --modelname conv1d --save_dir RESULTS/General/PHQ_All --label phq2_result_binary --mixup

# General - LOSO with Ablation Studies
python main.py --splitter loso --modelname conv1d --save_dir RESULTS/General/PHQ_IoT --label phq2_result_binary --mixup --ablation_data IoT
python main.py --splitter loso --modelname conv1dattn --save_dir RESULTS/General/PHQ_Voice --label phq2_result_binary --mixup --ablation_data Voice
python main.py --splitter loso --modelname conv1d --save_dir RESULTS/General/PHQ_Phone --label phq2_result_binary --mixup --ablation_data Phone
python main.py --splitter loso --modelname conv1d --save_dir RESULTS/General/PHQ_Wearable --label phq2_result_binary --mixup --ablation_data Wearable
python main.py --splitter loso --modelname fusion --save_dir RESULTS/General/PHQ_IoTVoice --label phq2_result_binary --mixup --ablation_data IoTVoice
python main.py --splitter loso --modelname fusion --save_dir RESULTS/General/PHQ_PhoneWearable --label phq2_result_binary --mixup --ablation_data PhoneWearable

# Personal - Stratified K-Fold
python main.py --splitter personalstratifiedkfold --modelname conv1d --save_dir RESULTS/Personal/PHQ_All --label phq2_result_binary --mixup
python main.py --splitter personalstratifiedkfold --modelname fusion --save_dir RESULTS/Personal/PHQ_All --label phq2_result_binary --mixup