import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_real_dict, y_pred_dict, title='Confusion Matrix', save_path='confusion_matrix.png'):
    
    num_users = len(y_real_dict)
    num_cols = 3
    num_rows = (num_users + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, num_rows * 4))
    axes = axes.flatten()

    for idx, uid in enumerate(y_real_dict.keys()):
        y_true = y_real_dict[uid]
        y_pred = y_pred_dict[uid]
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot(ax=axes[idx], values_format='d', cmap='Blues', colorbar=False)
        axes[idx].set_title(f'User {uid}')

    for idx in range(len(y_real_dict), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    fig.suptitle(title, y=1.05)
    plt.savefig(save_path)
    plt.close()

    
from sklearn.metrics import roc_curve, auc
def plot_auroc_curves(y_real_dict, y_proba_dict, title='AUROC Curves', save_path='auroc_curves.png'):
    num_users = len(y_real_dict)
    num_cols = 3
    num_rows = (num_users + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    for idx, uid in enumerate(y_real_dict.keys()):
        y_true = y_real_dict[uid]
        y_proba = y_proba_dict[uid]
        
        # Compute the ROC curve and AUROC score
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot the ROC curve
        axes[idx].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUROC = {roc_auc:.2f}')
        axes[idx].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[idx].set_xlim([0.0, 1.0])
        axes[idx].set_ylim([0.0, 1.05])
        axes[idx].set_xlabel('False Positive Rate')
        axes[idx].set_ylabel('True Positive Rate')
        axes[idx].set_title(f'User {uid}')
        axes[idx].legend(loc="lower right")

    # Remove unused subplots
    for idx in range(len(y_real_dict), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    fig.suptitle(title, y=1.05)
    plt.savefig(save_path)
    plt.close()

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

def plot_auroc_bar(y_real_dict, y_proba_dict, title='AUC by UID', save_path='auroc_bar.png', sort=False):
    plt.set_loglevel('WARNING')
    
    # Calculate AUROC for each user
    auc_scores = {str(uid): roc_auc_score(y_real_dict[uid], y_proba_dict[uid]) for uid in y_real_dict.keys()}
    
    # Sort by AUROC if specified
    if sort:
        auc_scores = dict(sorted(auc_scores.items(), key=lambda item: item[1], reverse=True))
    
    # Prepare data for plotting
    uids = list(auc_scores.keys())
    auc_values = list(auc_scores.values())
    mean_auc = np.mean(auc_values)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(uids, auc_values, color='gray')
    plt.axhline(mean_auc, color='red', linestyle='--', label=f'Mean AUC: {mean_auc:.4f}')
    
    # Labels and title
    plt.xlabel('UID')
    plt.ylabel('AUC')
    plt.title(f'{title} | μ: {mean_auc:.4f}, σ: {np.std(auc_values):.4f}')
    plt.xticks(ticks=range(len(uids)), labels=uids, ha='right')  # Set categorical labels
    plt.legend()
    
    # Save and show the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



import shap
import numpy as np
def plot_shap(values, X_test_shap, show_uid=False, figsize=(8, 6), plot_type='dot', dataset='All', splitter='loso', save_path=''):
    import matplotlib.pyplot as plt

    shap_values_concat = values[list(values.keys())[0]]
    for label, shap_values in list(values.items())[1:-1]:
        shap_values_concat = np.concatenate([shap_values_concat, shap_values], axis=0)
    
    fig = plt.figure()
    fig.set_facecolor('white')
    ax = fig.add_subplot()

    if show_uid == False:
        shap.summary_plot(shap_values_concat, X_test_shap, feature_names=values['column_names'], show=False, plot_size=figsize, plot_type=plot_type, color='k')
    else:
        shap.summary_plot(values[show_uid], X_test_shap[:values[show_uid].shape[0]], feature_names=values['column_names'], show=False, plot_size=figsize, plot_type=plot_type, color='k')

    # Modifying main plot parameters
    ax.tick_params(labelsize=10)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    ax.set_title(f'SHAP Summary Plot | Dataset: {dataset} | splitter={splitter}', fontsize=12)

    plt.tight_layout()
    if show_uid==False:
        uid='total'
    else:
        uid=show_uid
    plt.savefig(f'{save_path}/{uid}-{plot_type}.png', dpi=300, bbox_inches='tight')
    plt.close()