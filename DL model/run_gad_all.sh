#!/bin/bash

# General - LOSO
python main.py --splitter loso --modelname fusion --save_dir RESULTS/General/GAD_All --label gad2_result_binary --mixup --gpu 1
python main.py --splitter loso --modelname conv1d --save_dir RESULTS/General/GAD_All --label gad2_result_binary --mixup --gpu 1

# General - LOSO with Ablation Studies
python main.py --splitter loso --modelname conv1d --save_dir RESULTS/General/GAD_IoT --label gad2_result_binary --mixup --ablation_data IoT --gpu 1
python main.py --splitter loso --modelname conv1dattn --save_dir RESULTS/General/GAD_Voice --label gad2_result_binary --mixup --ablation_data Voice --gpu 1
python main.py --splitter loso --modelname conv1d --save_dir RESULTS/General/GAD_Phone --label gad2_result_binary --mixup --ablation_data Phone --gpu 1
python main.py --splitter loso --modelname conv1d --save_dir RESULTS/General/GAD_Wearable --label gad2_result_binary --mixup --ablation_data Wearable --gpu 1
python main.py --splitter loso --modelname fusion --save_dir RESULTS/General/GAD_IoTVoice --label gad2_result_binary --mixup --ablation_data IoTVoice --gpu 1
python main.py --splitter loso --modelname fusion --save_dir RESULTS/General/GAD_PhoneWearable --label gad2_result_binary --mixup --ablation_data PhoneWearable --gpu 1

# Personal - Stratified K-Fold
python main.py --splitter personalstratifiedkfold --modelname conv1d --save_dir RESULTS/Personal/GAD_All --label gad2_result_binary --mixup --gpu 1
python main.py --splitter personalstratifiedkfold --modelname fusion --save_dir RESULTS/Personal/GAD_All --label gad2_result_binary --mixup --gpu 1