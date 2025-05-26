import subprocess

mixup = True

params = {
    "dataset": "IotVoice",
    "label": "gad2_result_binary", # gad2_result_binary, phq2_result_binary
    "splitter": "loso",  # default: "loso", str: ["loso", "kfold", "personal", "personal2", "personalkfold"] # personal2: stratify=False
    "seed": 42,
    "gpu": '1',
}

params["log_file"] = f"output_04/{params['label']}/{params['splitter']}/{params['dataset']}.txt"
if params['splitter'] == "personalkfold" or params['splitter'] == "personalstratifiedkfold":
    print_meta = True
else:
    print_meta = False

modelnames = ['dnn_s', 'dnn_m', 'conv1d', 'conv1dcat', 'conv1dattn']

if 'PhoneWearable' in params['dataset'] or 'All' in params['dataset']:
    phone_modelnames = ['dnn_m', 'conv1d', 'conv1dattn']
    wearable_modelnames = ['dnn_s', 'dnn_m', 'conv1d']
else:
    phone_modelnames, wearable_modelnames = [None], [None]

if 'IotVoice' in params['dataset'] or 'All' in params['dataset']:
    iot_modelnames = ['dnn_m', 'conv1d', 'conv1dattn']
    voice_modelnames = ['dnn_m', 'conv1d', 'conv1dattn'] # conv1dcat
else:
    iot_modelnames, voice_modelnames = [None], [None]

commands = []

# 단일 모델
for modelname in modelnames:
    cmd = (f"python main.py --dataset {params['dataset']} --label {params['label']} --splitter {params['splitter']} "
           f"--modelname {modelname} --gpu {params['gpu']} --log_file {params['log_file']}")
    
    if print_meta:
        cmd += " --print_meta"
    
    commands.append(cmd)

# Fusion 모델
if 'All' in params['dataset']:
    for phone_modelname in phone_modelnames:
        for wearable_modelname in wearable_modelnames:
            for iot_modelname in iot_modelnames:
                for voice_modelname in voice_modelnames:
                    cmd = (f"python main.py --dataset {params['dataset']} --label {params['label']} --splitter {params['splitter']} "
                           f"--modelname fusion --phone_modelname {phone_modelname} --wearable_modelname {wearable_modelname} "
                           f"--other_modelname {iot_modelname} --voice_modelname {voice_modelname} --gpu {params['gpu']} "
                           f"--log_file {params['log_file']}")
                    
                    if print_meta:
                        cmd += " --print_meta"
                    
                    commands.append(cmd)

elif 'PhoneWearable' in params['dataset']:
    for phone_modelname in phone_modelnames:
        for wearable_modelname in wearable_modelnames:
            cmd = (f"python main.py --dataset {params['dataset']} --label {params['label']} --splitter {params['splitter']} "
                   f"--modelname fusion --phone_modelname {phone_modelname} --wearable_modelname {wearable_modelname} "
                   f"--gpu {params['gpu']} --log_file {params['log_file']}")
            
            if print_meta:
                cmd += " --print_meta"
            
            commands.append(cmd)

elif 'IotVoice' in params['dataset']:
    for iot_modelname in iot_modelnames:
        for voice_modelname in voice_modelnames:
            cmd = (f"python main.py --dataset {params['dataset']} --label {params['label']} --splitter {params['splitter']} "
                   f"--modelname fusion --other_modelname {iot_modelname} --voice_modelname {voice_modelname} "
                   f"--gpu {params['gpu']} --log_file {params['log_file']}")
            
            if print_meta:
                cmd += " --print_meta"
            
            commands.append(cmd)

# new_commands = []
if mixup:
    commands_copy = commands.copy()
    for cmd in commands_copy:
        cmd += " --mixup"
        commands.append(cmd)
        # new_commands.append(cmd)

print(len(commands))
# for cmd in new_commands:
for cmd in commands:
    subprocess.run(cmd, shell=True)
