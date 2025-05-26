
## Tabnet

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.augmentations import ClassificationSMOTE
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
###

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.optim import Adam, AdamW
import numpy as np
from sklearn.model_selection import train_test_split

###
aug = ClassificationSMOTE(p=0.2)
def get_space(seed, device_name):
    return {
        'n_d': hp.choice('n_d', [8, 16, 24, 32]),
        'n_a': hp.choice('n_a', [8, 16, 24, 32]),
        'n_steps': hp.choice('n_steps', [3, 4, 5, 6, 7]),
        'gamma': hp.uniform('gamma', 1.0, 2.0),
        'n_independent': hp.choice('n_independent', [1, 2, 3]),
        'n_shared': hp.choice('n_shared', [1, 2, 3]),
        'lambda_sparse': hp.uniform('lambda_sparse', 0.0001, 0.001),
        'optimizer_fn': hp.choice('optimizer_fn', [Adam, AdamW]),
        'optimizer_params': {
            'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-1))
        },
        'mask_type': hp.choice('mask_type', ['sparsemax', 'entmax']),
        # 'batch_size': hp.quniform('batch_size', 256, 1024, 1),
        # 'virtual_batch_size': hp.quniform('virtual_batch_size', 64, 256, 1),
        'momentum': hp.uniform('momentum', 0.01, 0.1),
        'clip_value': hp.uniform('clip_value', 1.0, 5.0),
        'seed': seed,
        'device_name': device_name,
    }

def objective(params):
    model = TabNetClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=['auc'],
        patience=5,
        max_epochs=500,
        augmentations=aug,
        batch_size=128,
        virtual_batch_size=64,
    )

    valid_outputs = model.predict_proba(X_valid)[:, 0]
    auc = roc_auc_score(y_valid, valid_outputs)
    return {'loss': -auc, 'status': STATUS_OK}

def train_tabnet(dataset, logger, train_option="None", seed=42, gpu=0):
    global X_train, y_train, X_valid, y_valid
    X_train, y_train, X_valid, y_valid, X_test, test_labels = dataset
    if train_option == 'hyperopt':
        trials = Trials()
        best = fmin(fn=objective, space=get_space(seed, f'cuda:{gpu}'), algo=tpe.suggest, max_evals=100, trials=trials)
        logger.info(f"  Best params: {best}")
        best_params = {
            'n_d': [8, 16, 24, 32][best['n_d']],
            'n_a': [8, 16, 24, 32][best['n_a']],
            'n_steps': [3, 4, 5, 6, 7][best['n_steps']],
            'gamma': best['gamma'],
            'n_independent': [1, 2, 3][best['n_independent']],
            'n_shared': [1, 2, 3][best['n_shared']],
            'lambda_sparse': best['lambda_sparse'],
            'optimizer_fn': Adam if best['optimizer_fn'] == 0 else AdamW,
            'optimizer_params': {'lr': best['lr']},
            # 'batch_size': int(best['batch_size']),  # Ensure batch_size is an integer
            # 'virtual_batch_size': int(best['virtual_batch_size']),  # Ensure virtual_batch_size is an integer
            'momentum': best['momentum'],
            'clip_value': best['clip_value'],
            'mask_type': ['sparsemax', 'entmax'][best['mask_type']],
            'verbose': 0,
            'seed': seed,
            'device_name': f'cuda:{gpu}',
            }
        
        model = TabNetClassifier(**best_params)
    else:
        model = TabNetClassifier(device_name=f'cuda:{gpu}')
    
    model.fit(
		    X_train, y_train,
		    eval_set = [(X_valid, y_valid)],
		    eval_metric=['auc'],
		    max_epochs=100,
		    # patience=5,
		    # augmentations=aug,
		    # batch_size=128,
		    # virtual_batch_size=64,
    )
    test_outputs = model.predict_proba(X_test)[:, 0]
    accuracy = accuracy_score(test_labels, test_outputs>0.5)
    auc_score = roc_auc_score(test_labels, test_outputs)
    f1 = f1_score(test_labels, test_outputs>0.5, average='macro')
    loss_list = []
    # return accuracy, auc_score, f1, test_outputs, test_labels, loss_list
    return accuracy, auc_score, f1

def ml_dataset(df, train_index, test_index, label, seed):
    train_index, valid_index = train_test_split(train_index, test_size=0.2, random_state=seed)

    to_drop = ['phq2_result', 'gad2_result',
            'stress_result', 'posNeg_result', 'arousal_result',
            'phq2_result_binary', 'gad2_result_binary', 
            'stress_result_binary', 'posNeg_result_binary', 'arousal_result_binary',
            'uid', 'timestamp'] #, 'word_count', 'duration']
    X = df.drop(columns=to_drop)

    y = df[label]

    X_train, y_train = X.iloc[train_index].values, y.iloc[train_index].values
    X_valid, y_valid = X.iloc[valid_index].values, y.iloc[valid_index].values
    X_test, test_labels = X.iloc[test_index].values, y.iloc[test_index].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    return (X_train, y_train, X_valid, y_valid, X_test, test_labels)
##
