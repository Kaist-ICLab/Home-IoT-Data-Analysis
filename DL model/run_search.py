import subprocess

cmd_loso_phq = (f"python main_mixup.py --dataset All --label phq2_result_binary --splitter loso "
                f"--modelname fusion --phone_modelname conv1d --wearable_modelname conv1d "
                f" --other_modelname conv1d --voice_modelname conv1dattn "
                f"--log_file output_04/best/loso_phq_exp3_mixup.txt --mixup")

cmd_personal_phq = (f"python main_mixup.py --dataset All --label phq2_result_binary --splitter personalstratifiedkfold "
                    f"--modelname fusion --phone_modelname dnn_m --wearable_modelname conv1d "
                    f" --other_modelname conv1d --voice_modelname conv1dattn "
                    f"--log_file output_04/best/personal_phq_exp3_mixup.txt --mixup")

cmd_loso_gad = (f"python main_mixup.py --dataset All --label gad2_result_binary --splitter loso "
                f"--modelname conv1d "
                # f"--modelname fusion --phone_modelname conv1d --wearable_modelname conv1d "
                # f" --other_modelname conv1d --voice_modelname conv1dattn "
                # f"--log_file output_04/best/loso_gad_exp3.txt")
                f"--log_file output_04/best/loso_gad_exp3.txt")

cmd_personal_gad = (f"python main_mixup.py --dataset All --label gad2_result_binary --splitter personalstratifiedkfold "
                    # f"--modelname conv1d "
                    f"--modelname fusion --phone_modelname conv1d --wearable_modelname conv1d "
                    f" --other_modelname conv1d --voice_modelname conv1dattn "
                    f"--log_file output_04/best/personal_gad_exp3.txt --mixup")


weights_search = ['uniform']
n_neighbors_search = [3]
corr_search = [2]

# def run_search(cmd, weights_search, n_neighbors_search, corr_search):
#     commands = []
#     for weights in weights_search:
#         for n_neighbors in n_neighbors_search:
#             for corr in corr_search:
#                 cmd += (f" --corr_thr {corr} --imputer_n_neighbors {n_neighbors} --imputer_weights {weights}")
#                 commands.append(cmd)
#     return commands


batch_size = [64]

criterion = ['bce', 'weighted_bce']

# epoch = [10, 20, 50, 100]

alpha = [1.0, 0.6, 0.4]
# classifier_option = ['original', 'complex', 'simple']
clasiifier_option = ['original', 'complex']
use_attention_weights = [False, True]

attention_type = ['multihead', 'temporal']
# attention_mechanism = ['none', 'feature_wise', 'cross_attention', 'transformer']
attention_mechanism = ['none', 'feature_wise']

# use_se_block = [False, True]

from itertools import product


def run_search(base_cmd, batch_size, criterion, epoch, alpha, classifier_option, use_attention_weights, 
               attention_type, attention_mechanism, use_se_block, mixup_ratio):
    commands = []
    
    # 모든 조합 생성
    for b, c, e, a, co, uaw, at, am, use_se, mr in product(
            batch_size, criterion, epoch, alpha, classifier_option, use_attention_weights,
            attention_type, attention_mechanism, use_se_block, mixup_ratio):
        
        # 기본 command 구성
        cmd = f"{base_cmd} --batch_size {b} --criterion {c} --epoch {e} --alpha {a} " \
              f"--classifier_option {co} --attention_type {at} --attention_mechanism {am} --mixup_ratio {mr}"
        
        # use_attention_weights와 use_se_block가 True일 때만 옵션 추가
        if uaw:
            cmd += " --use_attention_weights"
        if use_se:
            cmd += " --use_se_block"

        # cmd += " --viz_location output_04/exp_best/loso_phq --confusion_matrix --auroc_curve --auroc_bar"
        
        commands.append(cmd)
    
    return commands

commands = run_search(
    base_cmd=cmd_personal_gad,
    batch_size=[64],
    criterion=['bce'],
    epoch=[50],
    alpha=[1.0],
    classifier_option=['original'],
    use_attention_weights=[False],
    attention_type=['multihead'],
    attention_mechanism=['none'],
    use_se_block=[False],
    mixup_ratio=[0.5]
)

print(len(commands))
# for cmd in new_commands:
for cmd in commands:
    subprocess.run(cmd, shell=True)