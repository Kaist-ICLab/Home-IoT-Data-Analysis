import argparse
import logging
import os
import time
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, KFold, StratifiedKFold, train_test_split
from models import *
from setup import *
from customdataset import *
# from tabnet_setup import *
from visualize import *
import shap
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logging.getLogger('hyperopt').setLevel(logging.ERROR)
logging.getLogger('shap').setLevel(logging.ERROR)

# Define metrics functions
def calculate_accuracy(labels, outputs):
    predicted = (outputs > 0.5).float()
    return (predicted == labels).float().mean().item()

def calculate_auc(labels, outputs):
    return roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())

def calculate_f1(labels, outputs):
    return f1_score(labels.cpu().numpy(), (outputs > 0.5).cpu().numpy(), average='macro')

def balanced_accuracy(labels, outputs):
    predicted = (outputs > 0.5).float()
    return balanced_accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())

# General utility functions
def update_dict_with_uid(dictionary, uid, value, extend=False):
    if extend:
        dictionary.setdefault(uid, []).extend(value)
    else:
        dictionary.setdefault(uid, []).append(value)
    return dictionary

def append_to_file(filename, text):
    with open(filename, 'a') as file:
        file.write(text + '\n')

def ensure_save_path(base_path, sub_dir):
    save_path = f'{base_path}/{sub_dir}'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return save_path

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

models = {
    'dnn_s': DNN_small,
    'dnn_m': DNN_medium,
    'conv1dcat': CNN1dCat,
    'conv1d': CNN1d,
    'conv1dattn': CNN1dAttn,
    'fusion': FusionBase,
}

def create_submodel(modelname):
    """ 
    모델 이름과 `args`의 옵션에 따라 `conv1d` 또는 `conv1dattn`을 생성하고 필요한 옵션을 적용 
    """
    if modelname == 'conv1d':
        return models[modelname](use_se_block=args.use_se_block)
    elif modelname == 'conv1dattn':
        return models[modelname](attention_type=args.attention_type, attention_mechanism=args.attention_mechanism)
    else:
        # `conv1d`나 `conv1dattn`이 아닌 다른 모델 생성
        return models[modelname]()

from collections import Counter
def create_model(modelname='conv1dcat', y_train=None):
    if modelname=="fusion":
        if 'PhoneWearable' in args.dataset:
            model = models[modelname](
                models=[
                    create_submodel(args.phone_modelname),
                    create_submodel(args.wearable_modelname)
                    ],
                input_slices=[slice(None, -9), slice(-9, None)],
                classifier_option=args.classifier_option,
                use_attention_weights=args.use_attention_weights
            )
        elif 'All' in args.dataset:
            if args.wearable_modelname != 'None':
                model = models[modelname](
                    models=[
                        create_submodel(args.phone_modelname),
                        create_submodel(args.wearable_modelname),
                        create_submodel(args.other_modelname),
                        create_submodel(args.voice_modelname)
                    ],
                    input_slices=[slice(None, -351), slice(-351, -342), slice(-342, -182), slice(-182, None)],
                    classifier_option=args.classifier_option,
                    use_attention_weights=args.use_attention_weights
                )
            else:
                model = models[modelname](
                    models=[
                        create_submodel(args.phone_modelname),
                        create_submodel(args.other_modelname),
                        create_submodel(args.voice_modelname)
                        ],
                    input_slices=[slice(None, -342), slice(-342, -182), slice(-182, None)],
                    classifier_option=args.classifier_option,
                    use_attention_weights=args.use_attention_weights
                )
        else:
            model = models[modelname](
                models=[
                    create_submodel(args.other_modelname),
                    create_submodel(args.voice_modelname)
                ],
                input_slices=[slice(None, -182), slice(-182, None)],
                classifier_option=args.classifier_option,
                use_attention_weights=args.use_attention_weights
            )
    else:
        model = create_submodel(modelname)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    if args.criterion_type == 'weighted_bce' and y_train is not None:
        # 클래스 불균형에 따른 가중치 계산
        class_counts = Counter(y_train)
        pos_weight = torch.tensor([class_counts[0] / (class_counts[1] + 1e-6)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCELoss()

    metrics = {
        'accuracy': calculate_accuracy,
        'auc': calculate_auc,
        'f1': calculate_f1,
        'balanced_accuracy': balanced_accuracy
    }

    return model, optimizer, criterion, metrics

#### Augmentation ####
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
########################

#### Data Splitting ####
def holdout_by_user(features, labels, groups, stratify=True):
    
    features = features.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    for group in np.unique(groups):
        indices = np.where(groups == group)[0]
        if stratify:
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=args.seed, stratify=labels[indices])
        else:
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=args.seed)
        yield train_indices, test_indices

def personalized_kfold(features, labels, groups, n_splits=5):
    features = features.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    for group in np.unique(groups):
        indices = np.where(groups == group)[0]
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        for train_indices, test_indices in kf.split(indices):
            yield indices[train_indices], indices[test_indices]

def personal_stratified_kfold(features, labels, groups, n_splits=5):
    features = features.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)

    for group in np.unique(groups):
        indices = np.where(groups == group)[0]
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        for train_indices, test_indices in kf.split(indices, labels[indices]):
            yield indices[train_indices], indices[test_indices]

########################

#### Data Splitting ####
def split_data(df, label, groups, splitter):
    if splitter=='loso':
        return LeaveOneGroupOut().split(df, df[label], groups)
    elif splitter=='kfold':
        return KFold(n_splits=10, shuffle=True, random_state=args.seed).split(df)
    elif splitter=='personal': # stratify=True
        return holdout_by_user(df, df[label], groups, stratify=True)
    elif splitter=='personal2': # stratify=False
        return holdout_by_user(df, df[label], groups, stratify=False)
    elif splitter=='personalkfold':
        return personalized_kfold(df, df[label], groups, n_splits=5)
    elif splitter=='personalstratifiedkfold':
        return personal_stratified_kfold(df, df[label], groups, n_splits=5)

def train_iteration(model, train_dataset, optimizer, criterion, epoch, use_mixup=False, alpha=1.0):
    model.train()

    for _ in range(epoch):
        for b_idx, (X_batch, y_batch) in enumerate(train_dataset):
            optimizer.zero_grad()

            X_batch = X_batch.float().to(device)
            y_batch = y_batch.unsqueeze(1).to(device)

            # Perform mixup augmentation
            if use_mixup and (b_idx % 2 == 0): # 2 단위로 mixup augmentation
                mixed_X, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha)
                mixed_loss = mixup_criterion(criterion, model(mixed_X), y_a, y_b, lam)
                mixed_loss.backward()

            # Process the original data
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()


def evaluate_model(model, test_dataset, metrics):
    model.eval()
    test_outputs, test_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_dataset:
            outputs = model(X_batch.float().to(device))
            test_outputs.append(outputs)
            test_labels.append(y_batch.to(device))

    # calculate scores and save themf
    test_outputs = torch.cat(test_outputs).squeeze()
    test_labels = torch.cat(test_labels)#.unsqueeze(1)
    accuracy = metrics['accuracy'](test_labels, test_outputs)
    auc_score = metrics['auc'](test_labels, test_outputs) # auc_score 오류 발생시 처리 방법..?
    f1 = metrics['f1'](test_labels, test_outputs)
    balanced_accuracy = metrics['balanced_accuracy'](test_labels, test_outputs)
    return accuracy, auc_score, f1, balanced_accuracy, (test_outputs > 0.5).float().cpu().numpy(), test_outputs.cpu().numpy(), test_labels.cpu().numpy()


def log_initialization(splitter, label, modelname, device, args):
    logger.info(f'{splitter.upper()} | DATASET {args.dataset} | LABEL {label} | MODEL {modelname} | MIXUP {args.mixup} | SEED {args.seed} | DEVICE {device}')

def log_model_info(args):
    model_info_text = ''
    for model_type in ['phone', 'wearable', 'other', 'voice']:
        model_name = getattr(args, f"{model_type}_modelname", 'None')
        if model_name != 'None':
            model_info_text += f'{model_type}: {model_name} | '
    logger.info(f'MODEL INFO: {model_info_text}')

def preprocess_data(df, label):
    single_class_groups = remove_user_with_skewed_label(df, label)
    df = df[~df['uid'].isin(single_class_groups)]
    label_counts = df.groupby('uid')[label].value_counts().unstack(fill_value=0)
    valid_uids = label_counts[(label_counts[0] > 5) & (label_counts[1] > 5)].index
    return df[df['uid'].isin(valid_uids)], df['uid'].values, valid_uids

def apply_knn_imputer(X_train, X_test, n_neighbors=3, weights='uniform'):
    # X_train.columns = X_train.columns.astype(str)
    numeric_cols = X_train.select_dtypes(include=['float64', 'int']).columns
    knn_imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    X_train[numeric_cols] = knn_imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = knn_imputer.transform(X_test[numeric_cols])
    return X_train, X_test

def calculate_feature_deviation(X_train, X_test, columns):
    for col in columns:
        mean_per_uid = X_train[col].mean()
        X_train[f'{col}_deviation'] = X_train[col] - mean_per_uid
        X_train[f'{col}_comparison'] = (X_train[col] > mean_per_uid).astype(int) - (X_train[col] < mean_per_uid).astype(int)
        X_test[f'{col}_deviation'] = X_test[col] - mean_per_uid
        X_test[f'{col}_comparison'] = (X_test[col] > mean_per_uid).astype(int) - (X_test[col] < mean_per_uid).astype(int)

def evaluate_and_log(model, test_dataset, metrics, splitter, uid, accuracies, auc_scores, f1_scores, balanced_accuracies, dicts):
    accuracy, auc_score, f1, balanced_accuracy, test_outputs, test_proba, test_labels = evaluate_model(model, test_dataset, metrics)
    if splitter not in ['personalkfold', 'personalstratifiedkfold']:
        accuracies.append(accuracy)
        auc_scores.append(auc_score)
        f1_scores.append(f1)
        balanced_accuracies.append(balanced_accuracy)
    else:
        for key, val in zip(["accuracies", "auc_scores", "f1_scores", "balanced_accuracies"], [accuracy, auc_score, f1, balanced_accuracy]):
            dicts[key] = update_dict_with_uid(dicts[key], uid, val)
    return accuracy, auc_score, f1, balanced_accuracy

def calculate_shap_values(model, train_dataset, test_dataset, device):
    background_data = next(iter(train_dataset))[0].to(device)
    explainer = shap.GradientExplainer(model, background_data)
    
    shap_values = []
    for X_batch, y_batch in test_dataset:
        X_batch = X_batch.float().to(device)
        shap_values_batch = explainer.shap_values(X_batch)
        shap_values.append(shap_values_batch)
    return torch.cat([torch.tensor(s) for s in shap_values])

# def train_test(label='phq2_result_binary', splitter='loso', modelname='dnn_s', batch_size=64, epoch=100):
#     log_initialization(splitter, label, modelname, device, args)
#     if modelname == 'fusion':
#         log_model_info(args)

#     accuracies, auc_scores, f1_scores, balanced_accuracies = [], [], [], []

#     if splitter in ['personalkfold', 'personalstratifiedkfold']:
#         accuracies_dict, auc_scores_dict, f1_scores_dict, balanced_accuracies_dict = {}, {}, {}, {}

#     y_real_dict, y_pred_dict, y_proba_dict = {}, {}, {}

#     if shap:
#         shap_values_dict, shap_expected_values_dict, shap_X = {}, {}, pd.DataFrame()

#     df, groups, valid_uids = preprocess_data(merged_df.copy(), label)
#     cv = split_data(df, label, groups, splitter)

#     for split_idx, (train_index, test_index) in enumerate(cv):
#         # uid = int(df.iloc[test_index].uid.values[0]) # log 뽑을 때 uid로 나오도록
#         # UID 및 split_idx 할당
#         if splitter in ['personalkfold', 'personalstratifiedkfold']:
#             uid = int(df.iloc[test_index].uid.values[0])
#         else:
#             split_idx = int(df.iloc[test_index].uid.values[0])
#             uid = split_idx


def train_test(label='phq2_result_binary', splitter='loso', modelname='dnn_s', batch_size=64, epoch=100):
    log_initialization(splitter, label, modelname, device, args)
    if modelname == 'fusion':
        log_model_info(args)

    logger.info(f'Imputer | n_neighbors: {args.imputer_n_neighbors}, weights: {args.imputer_weights} | Corr_thr: {args.corr_thr}')

    accuracies, auc_scores, f1_scores, balanced_accuracies = [], [], [], []

    if splitter in ['personalkfold', 'personalstratifiedkfold']:
        accuracies_dict, auc_scores_dict, f1_scores_dict, balanced_accuracies_dict = {}, {}, {}, {}

    y_real_dict, y_pred_dict, y_proba_dict = {}, {}, {}

    if shap:
        shap_values_dict, shap_expected_values_dict, shap_X = {}, {}, pd.DataFrame()

    df = merged_df.copy()

    single_class_groups = remove_user_with_skewed_label(df, label)
    df = df[~df['uid'].isin(single_class_groups)]

    # Remove users with less than or equal to 5 instances of each label
    label_counts = df.groupby('uid')[label].value_counts().unstack(fill_value=0)
    valid_uids = label_counts[(label_counts[0] > 5) & (label_counts[1] > 5)].index
    df = df[df['uid'].isin(valid_uids)]

    groups = df['uid'].values

    #### Data Splitting ####
    cv = split_data(df, label, groups, splitter)
    corr_drop_list = []
    for split_idx, (train_index, test_index) in enumerate(cv):
        if splitter!='kfold' and splitter!='personalkfold' and splitter!='personalstratifiedkfold':
            split_idx = int(df.iloc[test_index].uid.values[0])
            uid = split_idx
        if splitter=='personalkfold' or splitter=='personalstratifiedkfold':
            uid = int(df.iloc[test_index].uid.values[0])

        train_df, test_df = df.iloc[train_index], df.iloc[test_index]
        
        # 2024-11-05 추가: Imputer 실행 위치 수정 및 X_train, X_test, y_train, y_test
        X_drop = ['phq2_result', 'gad2_result', 'stress_result', 'posNeg_result', 'arousal_result',
                    'phq2_result_binary', 'gad2_result_binary', 'stress_result_binary', 'posNeg_result_binary', 'arousal_result_binary',
                    'uid', 'timestamp']
        
        if 'timestamp' not in train_df.columns:
            X_drop.remove('timestamp')

        X_train, X_test = train_df.drop(columns=X_drop), test_df.drop(columns=X_drop)
        y_train, y_test = train_df[label], test_df[label]
        X_train, X_test = apply_knn_imputer(X_train, X_test, n_neighbors=args.imputer_n_neighbors, weights=args.imputer_weights)

        if splitter in ['personalkfold', 'personalstratifiedkfold']:
            calculate_feature_deviation(X_train, X_test, [col for col in X_train.columns if col.startswith('aqara_')])

        if args.corr_thr < 1:
            corr_drop = filter_high_correlation(X_train, threshold=args.corr_thr)
            X_train, X_test = X_train.drop(columns=corr_drop), X_test.drop(columns=corr_drop)
        else:
            corr_drop = []
        corr_drop_list.append(corr_drop)

        train_dataset = df_to_dataset_new(X_train, y_train, batch_size=batch_size, splitter=splitter)
        test_dataset = df_to_dataset_new(X_test, y_test, batch_size=batch_size, shuffle=False)

        model, optimizer, criterion, metrics = create_model(modelname)
        model = model.to(device)

        train_iteration(model, train_dataset, optimizer, criterion, epoch, use_mixup=args.mixup, alpha=args.alpha)

        # validate
        accuracy, auc_score, f1, balanced_accuracy, test_outputs, test_proba, test_labels = evaluate_model(model, test_dataset, metrics)


        if splitter!='personalkfold' and splitter!='personalstratifiedkfold':
            accuracies.append(accuracy)
            auc_scores.append(auc_score)
            f1_scores.append(f1)
            balanced_accuracies.append(balanced_accuracy)
        else:
            accuracies_dict = update_dict_with_uid(accuracies_dict, uid, accuracy)
            auc_scores_dict = update_dict_with_uid(auc_scores_dict, uid, auc_score)
            f1_scores_dict = update_dict_with_uid(f1_scores_dict, uid, f1)
            balanced_accuracies_dict = update_dict_with_uid(balanced_accuracies_dict, uid, balanced_accuracy)
        
        y_real_dict = update_dict_with_uid(y_real_dict, uid, test_labels, extend=True)
        y_pred_dict = update_dict_with_uid(y_pred_dict, uid, test_outputs, extend=True)
        y_proba_dict = update_dict_with_uid(y_proba_dict, uid, test_proba, extend=True)

        if splitter not in ['personalkfold', 'personalstratifiedkfold']:
            logger.info(f'{splitter.upper()} {uid:2d} | AUC: {auc_score:.4f} | ACC: {accuracy:.4f} | F1: {f1:.4f} | Balanced_ACC: {balanced_accuracy:.4f}')
        elif splitter in ['personalkfold', 'personalstratifiedkfold'] and len(auc_scores_dict[uid]) == 5:
            logger.info(f'{splitter.upper()} {uid:2d} | AUC: {np.mean(auc_scores_dict[uid]):.4f} | ACC: {np.mean(accuracies_dict[uid]):.4f} | F1: {np.mean(f1_scores_dict[uid]):.4f} | Balanced_ACC: {np.mean(balanced_accuracies_dict[uid]):.4f} ')
            if args.print_meta:
                logger.info(f'   * meta AUC {[f"{score:.4f}" for score in auc_scores_dict[uid]]} σ: {np.std(auc_scores_dict[uid]):.4f}\n   * meta ACC {[f"{score:.4f}" for score in accuracies_dict[uid]]} σ: {np.std(accuracies_dict[uid]):.4f}\n   * meta F1  {[f"{score:.4f}" for score in f1_scores_dict[uid]]} σ: {np.std(f1_scores_dict[uid]):.4f}\n   * meta BCC {[f"{score:.4f}" for score in balanced_accuracies_dict[uid]]} σ: {np.std(balanced_accuracies_dict[uid]):.4f}')

        ## SHAP
        if args.shap:
            shap_values = calculate_shap_values(model, train_dataset, test_dataset, device)
            shap_X = pd.concat([shap_X, X_test], axis=0)
            if splitter in ['personalkfold', 'personalstratifiedkfold']:
                shap_values_dict[split_idx] = shap_values[:,:,0].cpu().numpy()  # Save SHAP values
            else:
                shap_values_dict[uid] = shap_values[:,:,0].cpu().numpy()  # Save SHAP values

    if splitter in ['personalkfold', 'personalstratifiedkfold']:
        for uid in auc_scores_dict:
            if len(auc_scores_dict[uid]) == 5: # 5개 fold로 정상 작동한 경우만
                auc_scores.extend(auc_scores_dict[uid]) # append로 하면 list 자체를 넣고 extend를 하면 원소를 넣음
                accuracies.extend(accuracies_dict[uid])
                f1_scores.extend(f1_scores_dict[uid])
                balanced_accuracies.extend(balanced_accuracies_dict[uid])


    logger.info('\n------------MEAN SUMMARY------------')
    logger.info(f'     AUC      ACC      F1     B-ACC\nμ: {np.mean(auc_scores):.4f}   {np.mean(accuracies):.4f}   {np.mean(f1_scores):.4f}   {np.mean(balanced_accuracies):.4f}\nσ: {np.std(auc_scores):.4f}   {np.std(accuracies):.4f}   {np.std(f1_scores):.4f}   {np.std(balanced_accuracies):.4f}\n')

    logger.info(f'X_train.shape[1]: {X_train.shape[1]}\n# of dropped columns: {[len(corr_drop) for corr_drop in corr_drop_list]}')

    # Result Sumamry
    if modelname == "fusion":
        modelname += ' ('
        for mn in [args.phone_modelname, args.wearable_modelname, args.other_modelname, args.voice_modelname]:
            if mn != 'None':
                modelname += f"{mn} + "
        modelname = modelname[:-3] + ')'
    augment = f'Mixup(alpha={args.alpha})' if args.mixup else '-'
    result_text = f"{args.dataset},{modelname},{augment},{label},{np.mean(auc_scores):.4f}(σ={np.std(auc_scores):.4f}),{np.mean(accuracies):.4f}(σ={np.std(accuracies):.4f}),{np.mean(f1_scores):.4f}(σ={np.std(f1_scores):.4f}),{np.mean(balanced_accuracies):.4f}(σ={np.std(balanced_accuracies):.4f})"
    result_text += f",{args.corr_thr},{args.imputer_n_neighbors},{args.imputer_weights}"
    
    se_block = f'se_block' if args.use_se_block else '-'
    attention_weights = f'attention_weights' if args.use_attention_weights else '-'
    result_text += f",{args.classifier_option},{attention_weights},{se_block},{args.attention_type},{args.attention_mechanism},{args.criterion_type}"

    result_text += f",{args.batch_size},{args.epoch},{args.alpha}"

    append_to_file(f'{args.log_file[:-4]}_results.txt', result_text)

    if args.viz_location:
        if not os.path.exists(args.viz_location):
            os.makedirs(args.viz_location)

    if args.shap:
        # SHAP 관련 파일 저장 경로 설정 및 생성
        shap_location = os.path.join(args.viz_location, 'shap')
        if not os.path.exists(shap_location):
            os.makedirs(shap_location)

        # SHAP 값 저장
        shap_values_dict['column_names'] = X_test.columns
        # X_test_shap = df.drop(columns=X_drop).values.astype(float)
        if splitter in ['personalkfold', 'personalstratifiedkfold']:
            shap_uids = list(map(int, valid_uids))
            shap_values_dict_temp = {}
            for key in list(shap_values_dict.keys())[:-1]:
                if (key + 1) % 5 == 0:
                    shap_values_dict_temp[shap_uids.pop(0)] = np.concatenate(
                        [shap_values_dict[key - i] for i in range(4, -1, -1)], axis=0
                    )
            shap_values_dict_temp['column_names'] = shap_values_dict['column_names']
            shap_values_dict = shap_values_dict_temp

        # save_pickle(shap_values_dict, f"{shap_save_path}/shap_values_dict.pkl")
        # save_pickle(shap_expected_values_dict, f"{shap_save_path}/shap_expected_values_dict.pkl")
        # save_pickle(df.drop(columns=to_drop).values.astype(float), f"{shap_save_path}/X_test_shap.pkl")

        # SHAP 시각화 저장
        plot_shap(shap_values_dict, shap_X, show_uid=False, figsize=(8, 6), plot_type='dot',
                dataset=args.dataset, splitter=args.splitter, save_path=shap_location)
        plot_shap(shap_values_dict, shap_X, show_uid=False, figsize=(8, 6), plot_type='bar',
                dataset=args.dataset, splitter=args.splitter, save_path=shap_location)
        for uid in list(shap_values_dict.keys())[:-1]:
            plot_shap(shap_values_dict, shap_X, show_uid=uid, figsize=(8, 6), plot_type='dot',
                    dataset=args.dataset, splitter=args.splitter, save_path=shap_location)

    
    if args.confusion_matrix:
        plot_confusion_matrix(y_real_dict, y_pred_dict,
                            title=f'{splitter}-{label[:4]}-{args.dataset}-{modelname}',
                            save_path=f'{args.viz_location}/confusion_matrix.png')

    if args.auroc_curve:
        plot_auroc_curves(y_real_dict, y_proba_dict,
                        title=f'{splitter}-{label[:4]}-{args.dataset}-{modelname}',
                        save_path=f'{args.viz_location}/auroc_curves.png')
        
    if args.auroc_bar:
        plot_auroc_bar(y_real_dict, y_proba_dict,
                    title=f'{splitter}-{label[:4]}-{args.dataset}-{modelname}',
                    save_path=f'{args.viz_location}/auroc_bar.png')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default='Iot', help="final_df name")
    parser.add_argument("--label", type=str, default='phq2_result_binary', help="Label for the training")
    parser.add_argument("--splitter", type=str, default='loso', help="Splitter type for data splitting")
    parser.add_argument("--modelname", type=str, default='dnn_s', help="Name of the model to use")
    parser.add_argument("--other_modelname", type=str, default='None', help="Name of the model to use")
    parser.add_argument("--voice_modelname", type=str, default='None', help="Name of the model to use")
    #
    parser.add_argument("--phone_modelname", type=str, default="None", help="Only used for fusion_all")
    parser.add_argument("--wearable_modelname", type=str, default="None", help="Only used for fusion_all")
    #
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="Device")
    parser.add_argument("--log_file", type=str, default='output.txt')

    parser.add_argument("--mixup", action='store_true', help="use mixup (default: False)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha value for Mixup")

    parser.add_argument("--print_meta", action='store_true', help="if the selected splitter is personalkfold or personalstratifiedkfold, print meta information")

    parser.add_argument("--viz_location", type=str, default='viz', help="Location for saving visualizations")
    parser.add_argument("--shap", action='store_true', help="if you want to use SHAP, set this flag")
    parser.add_argument("--confusion_matrix", action='store_true', help="if you want to use confusion matrix, set this flag")
    parser.add_argument("--auroc_curve", action='store_true', help="if you want to use AUROC, set this flag")
    parser.add_argument("--auroc_bar", action='store_true', help="if you want to use AUROC bar, set this flag")

    parser.add_argument("--tab_train_option", type=str, default="None", help='TabNet training option: [hyperopt, None]')

    parser.add_argument("--corr_thr", type=float, default=2.0, help="Correlation threshold for feature selection")
    parser.add_argument("--imputer_n_neighbors", type=int, default=3, help="Number of neighbors for KNNImputer")
    parser.add_argument("--imputer_weights", type=str, default='uniform', help="Weights for KNNImputer")

    # FusionBase
    parser.add_argument('--classifier_option', type=str, default='original', choices=['original', 'complex', 'simple'],
                        help="Classifier option for FusionBase (e.g., 'original', 'complex', 'simple')")
    parser.add_argument('--use_attention_weights', action='store_true', help="Use attention weights in FusionBase")

    # CNN1d
    parser.add_argument('--use_se_block', action='store_true', help="Use Squeeze-and-Excitation block in CNN1d")

    # CNN1dAttn
    parser.add_argument('--attention_type', type=str, default='multihead', choices=['multihead', 'temporal'],
                        help="Attention type in CNN1dAttn (e.g., 'multihead', 'temporal')")
    parser.add_argument('--attention_mechanism', type=str, default='none', choices=['feature-wise', 'cross-attention', 'transformer', 'none'],
                        help="Additional attention mechanism in CNN1dAttn (e.g., 'feature-wise', 'cross-attention', 'transformer', 'none')")

    # Criterion
    parser.add_argument('--criterion_type', type=str, default='bce', choices=['bce', 'weighted_bce'],
                        help="Loss function type (e.g., 'bce' for BCELoss, 'weighted_bce' for Weighted BCELoss)")

    args = parser.parse_args()

    if args.dataset != 'Voice' and args.modelname == 'conv1dcat':
        print("conv1dcat model is only available for voice data")
        exit(0)

    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    seed_everything(args.seed)

    dataset_path = f"DATASET_04/{args.splitter}/{args.dataset}.csv"
    if not os.path.exists(dataset_path):
        merged_df = get_dataset(dataset=args.dataset)
        merged_df.to_csv(dataset_path, index=False)
        exit()
    else:
        merged_df = pd.read_csv(dataset_path)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    train_test(
        label=args.label,
        splitter=args.splitter,
        modelname=args.modelname,
        batch_size=args.batch_size,
        epoch=args.epoch
    )
    end_time = time.time()

    elapsed_time = end_time - start_time
    logger.info(f"Execution time: {elapsed_time:.4f} seconds\n")