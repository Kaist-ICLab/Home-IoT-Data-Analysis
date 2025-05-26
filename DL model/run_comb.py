import subprocess
from itertools import product

cmd_fusion = (f"python main_mixup.py --dataset All "
              f"--modelname fusion --log_file output_04/best/search_fusion2.txt "
              f"--batch_size 64 --criterion bce --epoch 50 --alpha 1 --classifier_option original "
              f"--attention_type multihead --attention_mechanism none --mixup_ratio 0.5 ")

cmd_conv1d = (f"python main_mixup.py --dataset All "
              f"--modelname conv1d --log_file output_04/best/search_one.txt "
              f"--batch_size 64 --criterion bce --epoch 50 --alpha 1 --classifier_option original "
              f"--attention_type multihead --attention_mechanism none --mixup --mixup_ratio 0.5 ")

cmd_source_1d = (f"python main_mixup.py --log_file output_04/best/search_source_1d_4.txt --batch_size 64 --criterion bce --epoch 50 --alpha 1 "
                 f"--classifier_option original --attention_type multihead --attention_mechanism none ")

splitter = ['loso']
label = ['phq2_result_binary', 'gad2_result_binary']
# label = ['gad2_result_binary']

phone_modelname = ['conv1d', 'dnn_m']
wearable_modelname = ['conv1d', 'dnn_m']
iot_modelname = ['conv1d', 'dnn_m']
voice_modelname = ['conv1dattn']


def run_search(base_cmd, splitter, label, phone_modelname, wearable_modelname, iot_modelname, voice_modelname):
    commands = []
    
    # 모든 조합 생성
    for s, l, pm, wm, im, vm in product(splitter, label, phone_modelname, wearable_modelname, iot_modelname, voice_modelname):
        
        # 기본 command 구성
        cmd = f"{base_cmd} --splitter {s} --label {l} --phone_modelname {pm} --wearable_modelname {wm} " \
              f"--other_modelname {im} --voice_modelname {vm}"
        
        # cmd += " --viz_location output_04/exp_best/loso_phq --confusion_matrix --auroc_curve --auroc_bar"
        
        commands.append(cmd)
    
    return commands

def run_search_one(base_cmd, splitter, label):
    commands = []
    
    # 모든 조합 생성
    for s, l in product(splitter, label):
        
        # 기본 command 구성
        cmd = f"{base_cmd} --splitter {s} --label {l}"
        
        # cmd += " --viz_location output_04/exp_best/loso_phq --confusion_matrix --auroc_curve --auroc_bar"
        
        commands.append(cmd)
    
    return commands

def run_search_source_1d(base_cmd, splitter, label, sources, mixup):
    commands = []
    
    # 모든 조합 생성
    for s, l, m, src in product(splitter, label, mixup, sources):
        
        # 기본 command 구성
        cmd = f"{base_cmd} --splitter {s} --label {l} --dataset {src} --modelname fusion --phone_modelname conv1d --wearable_modelname conv1d"
        
        if m:
            cmd += " --mixup"
        
        # cmd += " --viz_location output_04/exp_best/loso_phq --confusion_matrix --auroc_curve --auroc_bar"
        
        commands.append(cmd)
    
    return commands

# commands = run_search(cmd_fusion, splitter, label, phone_modelname, wearable_modelname, iot_modelname, voice_modelname)
# commands = run_search_one(cmd_conv1d, splitter, label)
# commands = run_search_source_1d(cmd_source_1d, splitter, label, ['Phone', 'Wearable', 'Iot', 'Voice', 'PhoneWearable', 'IotVoice'], mixup=[False, True])
commands = run_search_source_1d(cmd_source_1d, splitter, label, ['PhoneWearable'], mixup=[False, True])

print(len(commands))
# for cmd in new_commands:
for cmd in commands:
    subprocess.run(cmd, shell=True)