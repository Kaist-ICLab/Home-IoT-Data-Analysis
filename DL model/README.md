# Running the Code

## Depression Detection

### Generalized Model
- Multimodal Learning

```bash
python main.py --splitter loso --modelname fusion --save_dir RESULTS/General/PHQ_All --label phq2_result_binary --mixup
python main.py --splitter loso --modelname conv1d --save_dir RESULTS/General/PHQ_All --label phq2_result_binary --mixup

```

- Ablation Study

```bash
python main.py --splitter loso --modelname conv1d --save_dir RESULTS/General/PHQ_IoT --label phq2_result_binary --mixup --ablation_data IoT
python main.py --splitter loso --modelname conv1dattn --save_dir RESULTS/General/PHQ_Voice --label phq2_result_binary --mixup --ablation_data Voice
python main.py --splitter loso --modelname conv1d --save_dir RESULTS/General/PHQ_Phone --label phq2_result_binary --mixup --ablation_data Phone
python main.py --splitter loso --modelname conv1d --save_dir RESULTS/General/PHQ_Wearable --label phq2_result_binary --mixup --ablation_data Wearable
python main.py --splitter loso --modelname fusion --save_dir RESULTS/General/PHQ_IoTVoice --label phq2_result_binary --mixup --ablation_data IoTVoice
python main.py --splitter loso --modelname fusion --save_dir RESULTS/General/PHQ_PhoneWearable --label phq2_result_binary --mixup --ablation_data PhoneWearable

```

Personalized Model
```bash
python main.py --splitter personalstratifiedkfold --modelname conv1d --save_dir RESULTS/Personal/PHQ_All --label phq2_result_binary --mixup
python main.py --splitter personalstratifiedkfold --modelname fusion --save_dir RESULTS/Personal/PHQ_All --label phq2_result_binary --mixup
python main.py --splitter personalstratifiedkfold --modelname conv1d --save_dir RESULTS/Personal/PHQ_IoT --label phq2_result_binary --mixup --ablation_data IoT
python main.py --splitter personalstratifiedkfold --modelname conv1d --save_dir RESULTS/Personal/PHQ_Voice --label phq2_result_binary --mixup --ablation_data Voice
python main.py --splitter personalstratifiedkfold --modelname conv1d --save_dir RESULTS/Personal/PHQ_Phone --label phq2_result_binary --mixup --ablation_data Phone
python main.py --splitter personalstratifiedkfold --modelname conv1d --save_dir RESULTS/Personal/PHQ_Wearable --label phq2_result_binary --mixup --ablation_data Wearable
python main.py --splitter personalstratifiedkfold --modelname conv1d --save_dir RESULTS/Personal/PHQ_IoTVoice --label phq2_result_binary --mixup --ablation_data IoTVoice
python main.py --splitter personalstratifiedkfold --modelname conv1d --save_dir RESULTS/Personal/PHQ_PhoneWearable --label phq2_result_binary --mixup --ablation_data PhoneWearable
```


## Anxiety Detection
위에로부터 label을 gad2_result_binary로만 바꾸면 됩니다.