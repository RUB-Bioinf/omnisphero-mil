'''
This submodule contains miscellaneous functions to compute and analyze metrics. 
Originally used in the JoshNet
'''

# IMPORTS
#########

import itertools
from typing import Dict
from typing import List

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, auc, roc_curve
import torch
import util.utils

# matplotlib.use('Agg')
# plt.style.use('ggplot')


# FUNCTIONS
###########

# METRICS
from util import log
from util import utils


def multi_class_accuracy(outputs, targets):
    # TODO ?
    assert targets.size() == outputs.size()
    _, predictions = torch.max(outputs, dim=1)
    _, targets = torch.max(targets, dim=1)
    return (predictions == targets).sum().item() / targets.size(0)


def binary_accuracy(outputs, targets):
    assert targets.size() == outputs.size()
    y_prob = torch.ge(outputs, 0.5).float()
    return (targets == y_prob).sum().item() / targets.size(0)


# PLOTS

def plot_accuracy(history, save_path: str, include_raw: bool = False, include_tikz: bool = False):
    ''' takes a history object and plots the accuracies
    '''
    if include_raw:
        plot_metric(history, 'val_acc', save_path, include_tikz=include_tikz)
        plot_metric(history, 'train_acc', save_path, include_tikz=include_tikz)
    plot_metric(history, 'val_acc', save_path, 'train_acc', include_tikz=include_tikz)


def plot_accuracy_tiles(history, save_path: str, include_raw: bool = False, include_tikz: bool = False):
    ''' takes a history object and plots the accuracies
    '''
    if include_raw:
        plot_metric(history, 'val_acc_tiles', save_path, include_tikz=include_tikz)
        plot_metric(history, 'train_acc_tiles', save_path, include_tikz=include_tikz)
    plot_metric(history, 'val_acc_tiles', save_path, 'train_acc_tiles', include_tikz=include_tikz)


def plot_losses(history, save_path: str, include_raw: bool = False, include_tikz: bool = False):
    ''' takes a history object and plots the losses
    '''
    if include_raw:
        plot_metric(history, 'val_loss', save_path, include_tikz=include_tikz)
        plot_metric(history, 'train_loss', save_path, include_tikz=include_tikz)
    plot_metric(history, 'val_loss', save_path, 'train_loss', include_tikz=include_tikz)


def plot_metric(history, metric_name: str, out_dir: str, second_metric_name: str = None, dpi: int = 600,
                include_tikz: bool = False):
    metric_values = [i[metric_name] for i in history]
    metric_title = _get_metric_title(metric_name)
    metric_color, metric_type = _get_metric_color(metric_name)
    tikz_data_list = [metric_values]
    tikz_colors = [metric_color]
    tikz_legend = None

    plt.clf()
    plt.plot(metric_values, color=metric_color)
    if second_metric_name is None:
        out_file_name = 'raw-' + metric_name
        plt_title = metric_type + ': ' + metric_title
    else:
        out_file_name = metric_name.lower().replace('val_', '').replace('train_', '')
        second_metric_values = [i[second_metric_name] for i in history]
        second_metric_color, second_metric_type = _get_metric_color(second_metric_name)

        plt.plot(second_metric_values, color=second_metric_color)
        plt.legend([metric_type + " " + metric_title, second_metric_type + " " + metric_title])
        plt_title = 'Training & Validation'

        tikz_data_list.append(second_metric_values)
        tikz_colors.append(second_metric_color)
        tikz_legend = [metric_type, second_metric_type]

    plt.title(plt_title)
    plt.xlabel('Epoch')
    plt.ylabel(metric_title)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir + os.sep + out_file_name + '.pdf', dpi=dpi)
    plt.savefig(out_dir + os.sep + out_file_name + '.png', dpi=dpi)
    plt.clf()

    if include_tikz:
        tikz = utils.get_plt_as_tex(data_list_y=tikz_data_list, plot_colors=tikz_colors, title=plt_title,
                                    label_y=metric_title, plot_titles=tikz_legend)
        f = open(out_dir + os.sep + out_file_name + '.tex', 'w')
        f.write(tikz)
        f.close()


def plot_accuracies(history,out_dir:str, dpi: int = 600, include_tikz: bool = False):
    values_train_acc_tiles = [i['train_acc_tiles'] for i in history]
    values_val_acc_tiles = [i['val_acc_tiles'] for i in history]
    values_train_acc_bags = [i['train_acc'] for i in history]
    values_val_acc_bags = [i['val_acc'] for i in history]

    title = 'Accuracy'
    legend_entries = ['Tiles: Training','Tiles: Validation','Bags: Training','Bags: Validation']

    tikz_data_list = [values_train_acc_tiles, values_val_acc_tiles, values_train_acc_bags, values_val_acc_bags]
    tikz_colors = ['red', 'blue', 'teal', 'orange']

    plt.clf()
    plt.plot(values_train_acc_tiles, color='red')
    plt.plot(values_val_acc_tiles, color='blue')
    plt.plot(values_train_acc_bags, color='teal')
    plt.plot(values_val_acc_bags, color='orange')

    plt.legend(legend_entries)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir + os.sep + 'accuracy_combined.pdf', dpi=dpi)
    plt.savefig(out_dir + os.sep + 'accuracy_combined.png', dpi=dpi)
    plt.clf()

    if include_tikz:
        tikz = utils.get_plt_as_tex(data_list_y=tikz_data_list, plot_colors=tikz_colors, title=title,
                                    label_y='Accuracy', plot_titles=legend_entries)
        f = open(out_dir + os.sep + 'accuracy_combined.tex', 'w')
        f.write(tikz)
        f.close()


def _get_metric_color(metric_name: str):
    if metric_name.startswith('train_'):
        return 'blue', 'Training'
    if metric_name.startswith('val_'):
        return 'orange', 'Validation'
    return 'black', '??'


def _get_metric_title(metric_name: str):
    metric_name = metric_name.replace('val_', '')
    metric_name = metric_name.replace('train_', '')

    if metric_name == 'acc':
        metric_name = 'accuracy (Bags)'
    if metric_name == 'acc_tiles':
        metric_name = 'accuracy (Tiles)'

    return metric_name.capitalize()


def plot_conf_matrix(y_true, y_pred, save_path,
                     target_names,
                     title='Confusion Matrix',
                     normalize=True):
    '''computes and plots the confusion matrix using sklearn
    Title can be set arbitrarily but target_names should be a list of class names eg. ['positive', 'negative']
    '''
    conf_mat = confusion_matrix(y_true, y_pred)

    if len(target_names) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        binary_classification_counts = list((tn, fp, fn, tp))
        # print('TN, FP, FN, TP')
        # print(binary_classification_counts)

    acc = np.trace(conf_mat) / float(np.sum(conf_mat))
    miss_class = 1 - acc
    cmap = plt.get_cmap('Blues')

    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        title = title + ' (Normalized)'

    # plt.figure(figsize=(8,7))
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.grid(False)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    plt.title(title)

    thresh = conf_mat.max() / 1.5 if normalize else conf_mat.max() / 2
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(conf_mat[i, j]),
                     horizontalalignment="center",
                     color="white" if conf_mat[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(conf_mat[i, j]),
                     horizontalalignment="center",
                     color="white" if conf_mat[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(acc, miss_class))
    plt.tight_layout()

    plt.savefig(save_path + '_confusion_matrix.pdf', dpi=600)
    plt.clf()


def binary_roc_curve(y_true, y_hat_scores):
    ''' Only works for the binary classfication task.
    y_hat_scores are the raw sigmoidal network output probabilities
    (no torch.ge thresholding)

    Returns false positive rate, true positive rate and thresholds
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_hat_scores)
    return fpr, tpr, thresholds


def plot_binary_roc_curve(fpr, tpr, save_path: str, dpi: int = 600):
    ''' plots a ROC curve with AUC score
    in a binary classification setting
    '''
    area = auc(fpr, tpr)

    lw = 2
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC={:0.3f})'.format(area))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate (1-specificity)')
    plt.ylabel('True Positive Rate (sensitivity)')

    plt.savefig(save_path + 'roc_curve.pdf', dpi=dpi)
    plt.savefig(save_path + 'roc_curve.png', dpi=dpi)
    plt.savefig(save_path + 'roc_curve.svg', dpi=dpi)
    plt.clf()


def write_history(history: List[Dict[str, float]], history_keys: [str], metrics_dir: str, verbose: bool = False):
    out_file = metrics_dir + os.sep + 'history.csv'
    keys = history_keys.copy()
    keys.sort()

    out_text = 'Epoch'
    for key in keys:
        out_text = out_text + ';' + key

    for i in range(len(history)):
        out_text = out_text + '\n' + str(i + 1)
        for key in keys:
            out_text = out_text + ';' + str(history[i][key])

    f = open(out_file, 'w')
    f.write(out_text)
    f.close()

    if verbose:
        log.write('Saved training history: ' + out_file)
