'''
This submodule contains miscellaneous functions to compute and analyze metrics. 
Originally used in the JoshNet
'''

# IMPORTS
#########

import itertools
import math
import os
import traceback
from typing import Dict
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

import models
from models import BaselineMIL
from util import dose_response
from util import log
from util import utils
from util.utils import get_plt_as_tex
from util.utils import line_print
from util.well_metadata import TileMetadata


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

def plot_accuracy_bags(history, save_path: str, include_raw: bool = False, include_tikz: bool = False,
                       include_line_fit: bool = False):
    ''' takes a history object and plots the accuracies
    '''
    if include_raw:
        plot_metric(history, 'val_acc', save_path, include_tikz=include_tikz, include_line_fit=False)
        plot_metric(history, 'train_acc', save_path, include_tikz=include_tikz, include_line_fit=False)
    plot_metric(history, 'val_acc', save_path, 'train_acc', include_tikz=include_tikz,
                include_line_fit=include_line_fit)


def plot_accuracy_tiles(history, save_path: str, include_raw: bool = False, include_tikz: bool = False,
                        include_line_fit: bool = False):
    ''' takes a history object and plots the accuracies
    '''
    if include_raw:
        plot_metric(history, 'val_acc_tiles', save_path, include_tikz=include_tikz, include_line_fit=False)
        plot_metric(history, 'train_acc_tiles', save_path, include_tikz=include_tikz, include_line_fit=False)
    plot_metric(history, 'val_acc_tiles', save_path, 'train_acc_tiles', include_tikz=include_tikz,
                include_line_fit=include_line_fit)


def plot_binary_roc_curves(history, save_path: str, include_raw: bool = False, include_tikz: bool = False,
                           clamp: float = None):
    if include_raw:
        plot_metric(history, 'train_roc_auc', save_path, include_tikz=include_tikz, clamp=clamp, include_line_fit=False)
        plot_metric(history, 'val_roc_auc', save_path, include_tikz=include_tikz, clamp=clamp, include_line_fit=False)
    plot_metric(history, 'val_roc_auc', save_path, 'train_roc_auc', include_tikz=include_tikz, clamp=clamp,
                include_line_fit=False)


def plot_dice_scores(history, save_path: str, include_raw: bool = False, include_tikz: bool = False,
                     include_line_fit: bool = False,
                     clamp: float = None):
    if include_raw:
        plot_metric(history, 'train_dice_score', save_path, include_tikz=include_tikz, clamp=clamp,
                    include_line_fit=False)
        plot_metric(history, 'val_dice_score', save_path, include_tikz=include_tikz, clamp=clamp,
                    include_line_fit=False)
    plot_metric(history, 'val_dice_score', save_path, 'train_dice_score', include_tikz=include_tikz, clamp=clamp,
                include_line_fit=include_line_fit)


def plot_attention_otsu_threshold(history, save_path: str, label: int, include_raw: bool = False,
                                  include_tikz: bool = False, clamp: float = None):
    if include_raw:
        plot_metric(history, 'train_otsu_threshold_label' + str(label), save_path, include_tikz=include_tikz,
                    clamp=clamp, include_line_fit=False)
        plot_metric(history, 'val_otsu_threshold_label' + str(label), save_path, include_tikz=include_tikz, clamp=clamp,
                    include_line_fit=False)
    plot_metric(history, 'val_otsu_threshold_label' + str(label), save_path, 'train_otsu_threshold_label' + str(label),
                include_tikz=include_tikz, clamp=clamp, include_line_fit=False)


def plot_attention_entropy(history, save_path: str, label: int, include_raw: bool = False, include_tikz: bool = False,
                           clamp: float = None):
    if include_raw:
        plot_metric(history, 'train_entropy_attention_label' + str(label), save_path, include_tikz=include_tikz,
                    clamp=clamp, include_line_fit=False)
        plot_metric(history, 'val_entropy_attention_label' + str(label), save_path, include_tikz=include_tikz,
                    clamp=clamp, include_line_fit=False)
    plot_metric(history, 'val_entropy_attention_label' + str(label), save_path,
                'train_entropy_attention_label' + str(label), include_tikz=include_tikz, clamp=clamp,
                include_line_fit=False)


def plot_sigmoid_scores(history, save_path: str, include_tikz: bool = False, include_line_fit: bool = False):
    plot_metric(history, 'val_mean_sigmoid_scores', save_path, include_tikz=include_tikz, clamp=None,
                include_line_fit=include_line_fit)


def plot_losses(history, save_path: str, include_raw: bool = False, include_tikz: bool = False, clamp: float = None,
                include_line_fit: bool = False):
    ''' takes a history object and plots the losses
    '''
    if include_raw:
        plot_metric(history, 'val_loss', save_path, include_tikz=include_tikz, clamp=clamp, include_line_fit=False)
        plot_metric(history, 'train_loss', save_path, include_tikz=include_tikz, clamp=clamp, include_line_fit=False)
    plot_metric(history, 'val_loss', save_path, 'train_loss', include_tikz=include_tikz, clamp=clamp,
                include_line_fit=include_line_fit)


def plot_metric(history, metric_name: str, out_dir: str, second_metric_name: str = None, dpi: int = 350,
                include_tikz: bool = False, clamp: float = None, include_line_fit: bool = False):
    error_file = out_dir + os.sep + 'all_metric_errors.txt'
    if not os.path.exists(error_file):
        f = open(error_file, 'w')
        f.write('Errors will be written here.')
        f.close()

    try:
        _plot_and_save(history=history, metric_name=metric_name, out_dir=out_dir, second_metric_name=second_metric_name,
                       dpi=dpi, include_tikz=include_tikz, clamp=clamp, include_line_fit=include_line_fit)
    except Exception as e:
        error_text = "Error while rendering metric '" + metric_name + "'! Reason: " + str(e) + "."
        if include_line_fit:
            error_text = error_text + '\n["line_fit" was on. Trying again!]'
        error_text = error_text + "\nParams: history=" + str(history) + ", metric_name=" + str(
            metric_name) + ", out_dir=" + str(out_dir) + ", second_metric_name=" + str(
            second_metric_name) + ",dpi=" + str(dpi) + ", include_tikz=" + str(
            include_tikz) + ", clamp=" + str(clamp) + ", include_line_fit=" + str(include_line_fit)

        log.write(error_text)
        f = open(error_file, 'a')
        f.write(error_text + '\n')

        tb = traceback.TracebackException.from_exception(e)
        for line in tb.stack:
            log.write(str(line))
            f.write('\n' + str(line))
        f.close()

        if include_line_fit:
            try:
                plot_metric(history=history, metric_name=metric_name, out_dir=out_dir,
                            second_metric_name=second_metric_name, dpi=dpi, include_tikz=include_tikz, clamp=clamp,
                            include_line_fit=False)
            except Exception as e:
                error_text = "Error while re-rendering metric '" + metric_name + "'! Reason: " + str(e)
                log.write(error_text)
                f = open(error_file, 'a')
                f.write(error_text + '\n')
                f.close()


def _plot_and_save(history, metric_name: str, out_dir: str, second_metric_name: str, dpi: int, include_tikz: bool,
                   clamp: float, include_line_fit: bool):
    metric_values = [i[metric_name] for i in history]
    metric_title = _get_metric_title(metric_name)
    metric_color, metric_type = _get_metric_color(metric_name)

    if clamp is not None:
        metric_values = [max(min(i, clamp), clamp * -1) for i in metric_values]

    tikz_data_list = [metric_values]
    tikz_colors = [metric_color]
    tikz_legend = None
    plt_legend = [metric_type + " " + metric_title]

    metric_alpha = 1.0
    if include_line_fit:
        metric_alpha = 0.4
    plt.clf()
    plt.plot(metric_values, color=metric_color, alpha=metric_alpha)

    # fitting and drawing secondary line
    if include_line_fit:
        poly_color = _get_metric_color_poly_fit(metric_name)
        poly = np.polyfit(list(range(len(metric_values))), metric_values, 15)
        poly_y = np.poly1d(poly)(list(range(len(metric_values))))
        plt.plot(list(range(len(metric_values))), poly_y, color=poly_color, linewidth=1.1337)

        tikz_data_list.append(poly_y)
        tikz_colors.append(poly_color)
        plt_legend.append('Fit')
        tikz_legend = [metric_type + " " + metric_title, 'Fit']

        del poly, poly_y, poly_color

    if second_metric_name is None:
        out_file_name = 'raw-' + metric_name
        plt_title = metric_type + ': ' + metric_title
    else:
        out_file_name = metric_name.lower().replace('val_', '').replace('train_', '')
        second_metric_values = [i[second_metric_name] for i in history]
        second_metric_color, second_metric_type = _get_metric_color(second_metric_name)
        plt_title = 'Training & Validation'

        if clamp is not None:
            second_metric_values = [max(min(i, clamp), clamp * -1) for i in second_metric_values]

        plt.plot(second_metric_values, color=second_metric_color, alpha=metric_alpha)
        tikz_colors.append(second_metric_color)
        plt_legend.append(second_metric_type + " " + metric_title)

        if tikz_legend is None:
            tikz_legend = [metric_type + " " + metric_title, second_metric_type + " " + metric_title]

        if include_line_fit:
            poly_color = _get_metric_color_poly_fit(second_metric_name)
            poly = np.polyfit(list(range(len(second_metric_values))), second_metric_values, 15)
            poly_y = np.poly1d(poly)(list(range(len(second_metric_values))))
            plt.plot(list(range(len(second_metric_values))), poly_y, color=poly_color, linewidth=1.1337)

            tikz_data_list.append(poly_y)
            tikz_legend.append(second_metric_type + " " + metric_title)
            tikz_legend.append('Fit')
            plt_legend.append('Fit')
            tikz_colors.append(poly_color)

            del poly, poly_y, poly_color

        plt.legend(plt_legend)
        tikz_data_list.append(second_metric_values)

    plt.title(plt_title)
    plt.xlabel('Epoch')
    plt.ylabel(metric_title)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.autoscale()
    plt.savefig(out_dir + os.sep + out_file_name + '.pdf', dpi=dpi, bbox_inches='tight')
    plt.savefig(out_dir + os.sep + out_file_name + '.png', dpi=dpi, bbox_inches='tight')
    plt.clf()

    if include_tikz:
        tikz = utils.get_plt_as_tex(data_list_y=tikz_data_list, plot_colors=tikz_colors, title=plt_title,
                                    label_y=metric_title, plot_titles=tikz_legend)
        f = open(out_dir + os.sep + out_file_name + '.tex', 'w')
        f.write(tikz)
        f.close()


def plot_accuracies(history, out_dir: str, dpi: int = 600, include_tikz: bool = False, clamp: float = None,
                    include_line_fit: bool = False,  # TODO implement
                    ):
    values_train_acc_tiles = [i['train_acc_tiles'] for i in history]
    values_val_acc_tiles = [i['val_acc_tiles'] for i in history]
    values_train_acc_bags = [i['train_acc'] for i in history]
    values_val_acc_bags = [i['val_acc'] for i in history]

    if clamp is not None:
        values_train_acc_tiles = [max(min(i, clamp), clamp * -1) for i in values_train_acc_tiles]
        values_val_acc_tiles = [max(min(i, clamp), clamp * -1) for i in values_val_acc_tiles]
        values_train_acc_bags = [max(min(i, clamp), clamp * -1) for i in values_train_acc_bags]
        values_val_acc_bags = [max(min(i, clamp), clamp * -1) for i in values_val_acc_bags]

    title = 'Accuracy'
    legend_entries = ['Tiles: Training', 'Tiles: Validation', 'Bags: Training', 'Bags: Validation']

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
    plt.autoscale()
    plt.savefig(out_dir + os.sep + 'acc_combined.pdf', dpi=dpi, bbox_inches='tight')
    plt.savefig(out_dir + os.sep + 'acc_combined.png', dpi=dpi, bbox_inches='tight')
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


def _get_metric_color_poly_fit(metric_name: str):
    color, _ = _get_metric_color(metric_name=metric_name)
    if color == 'blue':
        return 'darkblue'
    if color == 'orange':
        return 'red'
    return 'black'


def _get_metric_title(metric_name: str):
    metric_name = metric_name.replace('val_', '')
    metric_name = metric_name.replace('train_', '')

    if metric_name == 'otsu_threshold_label0':
        metric_name = 'Attention (Normalized) Otsu Threshold (Label 0)'
    if metric_name == 'entropy_attention_label0':
        metric_name = 'Attention Entropy (Label 0)'
    if metric_name == 'otsu_threshold_label1':
        metric_name = 'Attention (Normalized) Otsu Threshold (Label 1)'
    if metric_name == 'entropy_attention_label1':
        metric_name = 'Attention Entropy (Label 1)'
    if metric_name == 'mean_sigmoid_scores':
        metric_name = 'Mean Sigmoid Scores'
    if metric_name == 'acc':
        metric_name = 'Accuracy (Bags)'
    if metric_name == 'acc_tiles':
        metric_name = 'Accuracy (Tiles)'
    if metric_name == 'roc_auc':
        metric_name = 'Binary ROC: AUC'
    if metric_name == 'dice_score':
        metric_name = 'Dice Score'
    else:
        metric_name = metric_name.capitalize()

    return metric_name


def plot_conf_matrix(y_true, y_pred, out_dir, target_names, title='Confusion Matrix', dpi=800, normalize=True):
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
    plt.clf()
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

    out_name = 'confusion_matrix_raw'
    if normalize:
        out_name = 'confusion_matrix_normalized'

    plt.autoscale()
    plt.savefig(out_dir + out_name + '.pdf', dpi=dpi, bbox_inches='tight')
    plt.savefig(out_dir + out_name + '.png', dpi=dpi, bbox_inches='tight')
    plt.clf()


def binary_roc_curve(y_true, y_hat_scores):
    ''' Only works for the binary classfication task.
    y_hat_scores are the raw sigmoidal network output probabilities
    (no torch.ge thresholding)

    Returns false positive rate, true positive rate and thresholds
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_hat_scores)
    return fpr, tpr, thresholds


def binary_pr_curve(y_true, y_hat_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_hat_scores)
    return precision, recall, thresholds


def plot_binary_pr_curve(precision, recall, thresholds, y_true, save_path: str, title: str, dpi: int = 600):
    pr_auc = float('NaN')
    try:
        pr_auc = auc(recall, precision)
    except Exception as e:
        log.write(str(e))

    y_true = np.asarray(y_true)
    pr_no_skill = len(y_true[y_true == 1]) / len(y_true)

    filename_base = 'pr_curve-' + title
    log.write('PR-Curve AUC: ' + str(pr_auc))
    log.write('Saving "' + title + '" PR to: ' + save_path)

    plt.plot([0, 1], [pr_no_skill, pr_no_skill], linestyle='--')
    plt.plot(recall, precision, label='PR (Area = {:.3f})'.format(pr_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (TPR)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall Curve: ' + title)
    plt.legend(loc='best')

    plt.autoscale()
    plt.savefig(save_path + os.sep + filename_base + '.png', bbox_inches='tight', dpi=dpi)
    plt.savefig(save_path + os.sep + filename_base + '.pdf', bbox_inches='tight', dpi=dpi, transparent=True)
    plt.savefig(save_path + os.sep + filename_base + '.svg', bbox_inches='tight', dpi=dpi, transparent=True)
    plt.clf()

    # Writing PR as .tex
    f = open(save_path + os.sep + filename_base + '.tex', 'w')
    f.write(get_plt_as_tex(data_list_x=[recall], data_list_y=[precision],
                           title='Precision Recall Curve', label_y='True positive rate',
                           label_x='False Positive Rate',
                           plot_titles=['PR (Area = {:.3f})'.format(pr_auc)],
                           plot_colors=['blue'], legend_pos='south west'))
    f.close()

    # Writing raw PR data as CSV
    f = open(save_path + os.sep + filename_base + '.csv', 'w')
    f.write('Baseline: ' + str(pr_no_skill) + '\n')
    f.write('i;Recall;Precision;Thresholds\n')
    for i in range(len(precision)):
        f.write(
            str(i + 1) + ';' + str(recall[i]) + ';' + str(precision[i]) + ';' + str(thresholds[0]) + ';\n')
    f.close()


def plot_binary_roc_curve(fpr, tpr, thresholds, save_path: str, title: str, dpi: int = 600):
    ''' plots a ROC curve with AUC score
    in a binary classification setting
    '''
    title = title.lower()

    area = float('NaN')
    try:
        area = auc(fpr, tpr)
    except Exception as e:
        log.write(str(e))

    lw = 2
    plt.clf()
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC Curve (AUC={:0.3f})'.format(area))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate (1-specificity)')
    plt.ylabel('True Positive Rate (sensitivity)')

    # Writing ROC curve as image data
    filename_base = 'roc_curve-' + title
    log.write('Saving "' + title + '" ROC to: ' + save_path)
    log.write('ROC-Curve AUC: ' + str(area))

    plt.autoscale()
    plt.savefig(save_path + os.sep + filename_base + '.pdf', dpi=dpi, bbox_inches='tight')
    plt.savefig(save_path + os.sep + filename_base + '.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(save_path + os.sep + filename_base + '.svg', dpi=dpi, bbox_inches='tight')
    plt.clf()

    # Writing ROC as .tex
    f = open(save_path + os.sep + filename_base + '.tex', 'w')
    f.write(get_plt_as_tex(data_list_x=[fpr], data_list_y=[tpr], title='ROC Curve: ' + title.capitalize(),
                           label_y='True positive rate', label_x='False Positive Rate', plot_colors=['blue']))
    f.close()

    # Writing raw ROC data as CSV
    f = open(save_path + os.sep + filename_base + '.csv', 'w')
    f.write('i;FPR;TPR;Thresholds\n')
    for i in range(len(thresholds)):
        f.write(
            str(i + 1) + ';' + str(fpr[i]) + ';' + str(tpr[i]) + ';' + str(thresholds[i]) + ';\n')
    f.close()


def write_history(history: List[Dict[str, float]], history_keys: [str], metrics_dir: str, verbose: bool = False):
    out_file = metrics_dir + os.sep + 'history.csv'
    keys = history_keys.copy()
    keys.sort()

    out_text = 'Epoch'
    for key in keys:
        out_text = out_text + ';' + str(key)

    for i in range(len(history)):
        out_text = out_text + '\n' + str(i + 1)
        for key in keys:
            out_text = out_text + ';' + str(history[i][key])

    f = open(out_file, 'w')
    f.write(out_text)
    f.close()

    if verbose:
        log.write('Saved training history: ' + out_file)


def save_tile_attention(out_dir: str, model: BaselineMIL, dataset: DataLoader, X_raw: np.ndarray, y_tiles: np.ndarray,
                        bag_names: [str], colormap_name: str = 'jet', dpi: int = 350, overlay_alpha: float = 0.65,
                        normalized: bool = False):
    if not model.enable_attention:
        log.write('Not using attention. Scores skipped.')
        return

    image_width = None
    image_height = None
    color_map = plt.get_cmap(colormap_name)
    os.makedirs(out_dir, exist_ok=True)
    y_hats, all_predictions, all_true, _, all_y_tiles_binarized, all_tiles_true, all_attentions, original_bag_indices = models.get_predictions(
        model, dataset)

    max_attention_tilesFP = []
    max_attention_tilesTP = []
    max_attention_tilesFN = []
    max_attention_tilesTN = []
    min_attention_tilesFP = []
    min_attention_tilesTP = []
    min_attention_tilesFN = []
    min_attention_tilesTN = []

    print('')
    for i in range(len(y_hats)):
        line_print('Writing Attentions for Bag: ' + str(i + 1) + '/' + str(len(y_hats)), include_in_log=False)
        original_bag_index = original_bag_indices[i]
        tile_attentions = all_attentions[i]

        # Extracting predictions for the current bag
        y_tile_predictions = all_y_tiles_binarized[i]
        y_tile_predictions_true = all_tiles_true[i]
        y_bag = all_predictions[i]
        y_bag_true = all_true[i]
        raw_bag = X_raw[original_bag_index]
        bag_name = bag_names[original_bag_index]

        # Preparing parameters
        colored_tiles = []
        tile_count = raw_bag.shape[0]
        correct_tiles: float = 0.0

        # Extracting the tiles with the most attentions
        max_attention_indexes = np.where(tile_attentions == max(tile_attentions))
        min_attention_indexes = np.where(tile_attentions == min(tile_attentions))
        max_attention_tile = None
        min_attention_tile = None
        added_attention_tiles_count = 0
        for current_index in max_attention_indexes[0]:
            max_attention_tile = raw_bag[current_index]
            max_attention_tile = max_attention_tile.astype('uint8').copy()
            max_attention_tile = outline_rgb_array(max_attention_tile, None, None, outline=2,
                                                   override_colormap=[255, 255, 255])
            if y_bag_true == 1:
                if y_bag == 1:
                    max_attention_tilesTP.append(max_attention_tile)
                    added_attention_tiles_count = added_attention_tiles_count + 1
                elif y_bag == 0:
                    max_attention_tilesFP.append(max_attention_tile)
                    added_attention_tiles_count = added_attention_tiles_count + 1
            elif y_bag_true == 0:
                if y_bag == 1:
                    max_attention_tilesFN.append(max_attention_tile)
                    added_attention_tiles_count = added_attention_tiles_count + 1
                elif y_bag == 0:
                    max_attention_tilesTN.append(max_attention_tile)
                    added_attention_tiles_count = added_attention_tiles_count + 1
        del max_attention_tile

        # Also extracting the minimum attention tiles
        for current_index in min_attention_indexes[0]:
            min_attention_tile = raw_bag[current_index]
            min_attention_tile = min_attention_tile.astype('uint8').copy()
            min_attention_tile = outline_rgb_array(min_attention_tile, None, None, outline=2,
                                                   override_colormap=[255, 255, 255])
            if y_bag_true == 1:
                if y_bag == 1:
                    # TODO why is this error?
                    min_attention_tilesTP.append(min_attention_tile)
                    added_attention_tiles_count = added_attention_tiles_count + 1
                elif y_bag == 0:
                    min_attention_tilesFP.append(min_attention_tile)
                    added_attention_tiles_count = added_attention_tiles_count + 1
            elif y_bag_true == 0:
                if y_bag == 1:
                    min_attention_tilesFN.append(min_attention_tile)
                    added_attention_tiles_count = added_attention_tiles_count + 1
                elif y_bag == 0:
                    min_attention_tilesTN.append(min_attention_tile)
                    added_attention_tiles_count = added_attention_tiles_count + 1

        del min_attention_tile
        assert len(max_attention_indexes[0]) + len(min_attention_indexes[0]) == added_attention_tiles_count
        del added_attention_tiles_count

        # Overlapping the bags with the attention and saving the files
        for j in range(tile_count):
            current_tile = raw_bag[j]
            attention = tile_attentions[j]

            if tile_attentions.max() == 0 or (tile_attentions.max() - tile_attentions.min()) == 0:
                normalized_attention = 0
            else:
                normalized_attention = (attention - tile_attentions.min()) / (
                        tile_attentions.max() - tile_attentions.min())

            correct_tiles = float(correct_tiles + float(y_tile_predictions[j] == y_tile_predictions_true[j]))
            r = current_tile[0] / 255 * overlay_alpha
            g = current_tile[1] / 255 * overlay_alpha
            b = current_tile[2] / 255 * overlay_alpha

            attention_color = color_map(attention)
            if normalized:
                attention_color = color_map(normalized_attention)

            r = r + (attention_color[0] * (1 - overlay_alpha))
            g = g + (attention_color[1] * (1 - overlay_alpha))
            b = b + (attention_color[2] * (1 - overlay_alpha))

            r = r * 255
            g = g * 255
            b = b * 255
            r = r.astype('uint8')
            g = g.astype('uint8')
            b = b.astype('uint8')

            rgb = np.dstack((r, g, b)).copy()
            rgb = outline_rgb_array(rgb, None, None, outline=6,
                                    override_colormap=[attention_color[0] * 255, attention_color[1] * 255,
                                                       attention_color[2] * 255])
            rgb = outline_rgb_array(rgb, true_value=y_tile_predictions_true[j], prediction=y_tile_predictions[j],
                                    outline=3)
            colored_tiles.append(rgb)
            image_width, image_height = r.shape

        tile_accuracy: float = correct_tiles / float(tile_count)
        # Saving base image
        filename_base = out_dir + 'bag-' + str(original_bag_index)
        if normalized:
            filename_base = filename_base + '-normalized'
        out_image = fuse_image_tiles(images=colored_tiles, image_width=image_width, image_height=image_height)
        plt.imsave(filename_base + '.png', out_image)

        # Saving as annotated py plot
        plt.clf()
        color_bar_min = 0.0
        color_bar_max = 1.0
        if normalized:
            color_bar_min = tile_attentions.min()
            color_bar_max = tile_attentions.max()

        # Creating a dummy image for the color bar to fit
        img = plt.imshow(np.array([[color_bar_min, color_bar_max]]), cmap=colormap_name)
        img.set_visible(False)
        c_bar = plt.colorbar(orientation='vertical')

        color_bar_title = 'Attention'
        if normalized:
            color_bar_title = 'Attention (Normalized)'
        c_bar.ax.set_ylabel(color_bar_title, rotation=270)

        plt.imshow(out_image)
        plt.xticks([], [])
        plt.yticks([], [])

        tile_accuracy_formatted = str("{:.4f}".format(tile_accuracy))
        plt.xlabel('Bag label: ' + str(int(y_bag_true)) + '. Prediction: ' + str(int(y_bag)))
        plt.ylabel('Tiles: ' + str(tile_count) + '. Tile Accuracy: ' + tile_accuracy_formatted)
        plt.title('Attention Scores: Bag #' + str(original_bag_index) + ' - ' + bag_name)

        plt.tight_layout()
        plt.autoscale()
        plt.savefig(filename_base + '-detail.png', dpi=dpi, bbox_inches='tight')
        plt.savefig(filename_base + '-detail.pdf', dpi=dpi, bbox_inches='tight')
        plt.clf()

        # Writing debug texts
        f = open(out_dir + os.sep + 'debug-bag-accuracy.txt', 'a')
        f.write('bag-' + str(original_bag_index) + ': Tile Accuracy: ' + str(tile_accuracy) + ', ' + str(
            correct_tiles) + ' out of ' + str(tile_count) + '.\n')
        f.close()

    # Writing attention tiles
    for (max_attention_tiles, min_attention_tiles, metric_name) in zip(
            [max_attention_tilesTP, max_attention_tilesFP, max_attention_tilesFN, max_attention_tilesTN],
            [min_attention_tilesTP, min_attention_tilesFP, min_attention_tilesFN, min_attention_tilesTN],
            ['TP', 'FP', 'FN', 'TN']):
        log.write('Saving ' + str(len(max_attention_tiles)) + ' max and ' + str(
            len(min_attention_tiles)) + ' min tiles for metric ' + metric_name)

        # Writing as png images, if they exist
        if len(max_attention_tiles) > 0:
            out_image = fuse_image_tiles(images=max_attention_tiles, image_width=image_width, image_height=image_height)
            max_attention_file = out_dir + 'max_attention_' + metric_name + '.png'
            plt.imsave(max_attention_file, out_image)

        if len(min_attention_tiles) > 0:
            out_image = fuse_image_tiles(images=min_attention_tiles, image_width=image_width, image_height=image_height)
            min_attention_file = out_dir + 'min_attention_' + metric_name + '.png'
            plt.imsave(min_attention_file, out_image)

        # Writing as text files
        f = open(out_dir + 'max_attention_' + metric_name + '.txt', 'w')
        f.write('Tile count: ' + str(len(max_attention_tiles)))
        f.close()

        f = open(out_dir + 'min_attention_' + metric_name + '.txt', 'w')
        f.write('Tile count: ' + str(len(min_attention_tiles)))
        f.close()


def calculate_dice_score(TP: int, FP: int, FN: int):
    TP = float(math.floor(TP))
    FP = float(math.floor(FP))
    FN = float(math.floor(FN))

    if (2 * TP + FP + FN) == 0:
        return math.nan

    return (2 * TP) / (2 * TP + FP + FN)


def fuse_image_tiles(images: [np.ndarray], image_width: int = None, image_height: int = None, light_mode: bool = False):
    # assert image_width is None
    # assert image_height is None

    if image_width is None or image_height is None:
        image_width_max = 0
        image_height_max = 0
        for image in images:
            s = image.shape
            image_width_max = max(image_width_max, s[0])
            image_height_max = max(image_height_max, s[1])

        image_width = max(image_width_max, image_height_max)
        image_height = max(image_width_max, image_height_max)
        del s, image, image_height_max, image_width_max

    image_count = len(images)
    assert image_count > 0

    if image_count == 1:
        # Special case, if there is only 1 image in the list
        return images[0].astype(np.uint8)

    if image_count == 2:
        # Special case, if there are only 2 images in the list
        combined_img = np.zeros((image_height, image_width * 2, 3), dtype=np.uint8)

        padded_img_1 = np.zeros((image_width, image_height, 3), dtype=np.uint8)
        padded_img_0 = np.zeros((image_width, image_height, 3), dtype=np.uint8)
        if light_mode:
            padded_img_0 = np.ones((image_width, image_height, 3), dtype=np.uint8) * 255
            padded_img_1 = np.ones((image_width, image_height, 3), dtype=np.uint8) * 255

        padded_img_0[:images[0].shape[0], :images[0].shape[1], :] = images[0]
        padded_img_1[:images[1].shape[0], :images[1].shape[1], :] = images[1]

        combined_img[0:image_width, 0:image_height] = padded_img_0
        combined_img[0:image_width, image_height:image_height * 2] = padded_img_1
        return combined_img

    out_image_bounds = math.ceil(math.sqrt(image_count))
    combined_img = np.zeros((out_image_bounds * image_width, out_image_bounds * image_height, 3), dtype=np.uint8)
    y = -1
    x = -1
    for i in range(image_count):
        x = (x + 1) % out_image_bounds
        if x == 0:
            y = y + 1
        current_img = (images[i]).astype(np.uint8)
        s = current_img.shape

        current_img_padded = np.zeros((image_width, image_height, 3), dtype=np.uint8)
        if light_mode:
            current_img_padded = np.ones((image_width, image_height, 3), dtype=np.uint8) * 255
        current_img_padded[:s[0], :s[1], :] = current_img

        combined_img[x * image_width:x * image_width + image_width,
        y * image_height:y * image_height + image_height] = current_img_padded

    return combined_img


def outline_rgb_array(image: [np.ndarray], true_value: float, prediction: float, outline: int = 3,
                      bright_mode: bool = True, override_colormap: [int, int, int] = None):
    # This function assumes an input of shape: x, y, 3
    width, height, _ = image.shape

    if width == 3 and width < height:
        image = np.einsum('abc->bca', image)

    colormap: [int, int, int] = None
    if override_colormap is None:
        is_class0: bool = true_value == 0
        is_hit = true_value == prediction
        colormap: [int, int, int] = [
            int(float(is_class0) * 255),
            int(float(is_hit) * 255),
            int(float(bright_mode) * 255)]
    else:
        colormap = override_colormap

    # Creating a deep copy so it's not overwritten
    image = image.copy()
    image = np.copy(image)

    image[0:outline, :] = colormap
    image[width - outline:width, :] = colormap
    image[:, 0:outline] = colormap
    image[:, height - outline:height] = colormap
    image = image.astype('uint8')
    image = image.copy()

    return image


def save_sigmoid_prediction_csv(experiment_name: str, file_path: str, all_well_letters: [str], prediction_dict: dict,
                                prediction_dict_well_names: dict, verbose: bool = False):
    if verbose:
        log.write('Writing prediction CSV: ' + file_path)

    log.write('DeprecationWarning: save_sigmoid_prediction_csv() is unused!')

    # Saving the results to a CSV file
    f = open(file_path, 'w')
    f.write(experiment_name + ';')
    [f.write(letter + ';') for letter in all_well_letters]
    f.write('Mean')
    for well_index in prediction_dict.keys():
        f.write('\n' + str(well_index) + ';')
        current_prediction = prediction_dict[well_index]
        current_well_names = prediction_dict_well_names[well_index]

        for letter in all_well_letters:
            for (prediction_value, predicted_well) in zip(current_prediction, current_well_names):
                if predicted_well == letter + str(well_index):
                    f.write(str(prediction_value))
            f.write(';')
        f.write(str(np.mean(current_prediction)))
    f.close()


def save_sigmoid_prediction_img(file_path: str, title: str, prediction_dict: dict, prediction_dict_well_names: dict,
                                include_curve_fit: bool = True, include_ideal_fit: bool = True,
                                dpi: int = 900, x_ticks_angle: int = 30, x_ticks_font_size: int = 4,
                                verbose: bool = False):
    log.write('DeprecationWarning: save_sigmoid_prediction_img() is unused!')

    # Writing the results as dose-response png images
    if verbose:
        log.write('Rendering dose response curve: "' + title + '" at ' + file_path)

    plt.clf()
    x_labels = [str(l) for l in prediction_dict_well_names.values()]
    ticks = list(range(len(x_labels)))
    plt.xticks(ticks=ticks, labels=x_labels, rotation=x_ticks_angle, fontsize=x_ticks_font_size)

    x = list(range(len(prediction_dict.keys())))
    y = [np.mean(prediction_dict[p]) for p in prediction_dict]

    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.xlabel('Wells')
    plt.ylabel('Dose Response Predictions')
    plt.title(title)

    if include_curve_fit or include_ideal_fit:
        plt.plot(x, y, 'o', label='Predictions', color='red')
        plt.legend(loc='best')
    else:
        plt.plot(x, y)

    if include_ideal_fit:
        y, x = dose_response.curve_fit_ideal(len(prediction_dict) - 1)
        plt.plot(x, y, label='Ideal', color='lightgreen', linestyle='dotted')
        plt.legend(loc='best')

    if include_curve_fit:
        y, x = dose_response.curve_fit_prediction(prediction_dict=prediction_dict)

        if y is not None and x is not None:
            legend_label = 'Sigmoid Fit'
            if include_ideal_fit:
                d, f = dose_response.curve_fit_prediction_accuracy(prediction_dict=prediction_dict)
                legend_label = 'Sigmoid Fit (Frechet: {:.4f})'.format(f)

            plt.plot(x, y, label=legend_label, color='darkblue')
            plt.legend(loc='best')
        else:
            plt.plot([0, len(prediction_dict) - 1], [0, 0], label='Sigmoid fit: Failed', color='darkblue')
            plt.legend(loc='best')

    del x, y

    plt.autoscale()
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')


def attention_metrics(attention: np.ndarray, normalized: bool, hist_bins_override: int = None):
    a = np.asarray(attention, dtype=np.float64)
    if normalized:
        if a.max() == 0:
            a = np.zeros((1, len(a))).squeeze()
        else:
            a = a / a.max()
        a = a.astype(np.float32)

    # Applying override
    log.write('Histogram bin overrides: ' + str(hist_bins_override), print_to_console=False)
    if hist_bins_override is not None:
        log.write('Applying override.', print_to_console=False)
        n, bins = np.histogram(a, bins=hist_bins_override - 1)
        n = n.tolist()
        n.append(0)
    else:
        log.write('Not applying override.', print_to_console=False)
        n, bins = utils.sparse_hist(a)
    log.write('Histogram done. Length of "n": ' + str(len(n)) + ', "bins": ' + str(len(bins)), print_to_console=False)
    n = np.asarray(n)
    bins = np.asarray(bins)

    otsu_index = utils.lecture_otsu(n=n)
    entropy_attention = utils.lecture_shannon_entropy(n=a)
    entropy_hist = utils.lecture_shannon_entropy(n=n)

    # Checking if the otsu is valid
    if otsu_index is None or math.isnan(otsu_index) or otsu_index == len(bins):
        log.write('Illegal otsu threshold: "' + str(otsu_index) + '"!')
        otsu_threshold = float('nan')
    else:
        # otsu_threshold = bins[n[otsu_index]]
        otsu_threshold = bins[otsu_index]

    return n, bins, otsu_index, otsu_threshold, entropy_attention, entropy_hist


def attention_metrics_batch(all_attentions: [np.ndarray], X_metadata: [[TileMetadata]], normalized: bool,
                            hist_bins_override: int = None):
    assert len(all_attentions) == len(X_metadata)

    metadata_list = []
    otsu_index_list = []
    otsu_threshold_list = []
    entropy_attention_list = []
    entropy_hist_list = []
    n_list = []
    bins_list = []
    error_list = []

    log.write('Calculating metrics for ' + str(len(all_attentions)) + ' attentions.')
    print('')
    for (attention, metadata) in zip(all_attentions, X_metadata):
        print('Len attention: ' + str(len(attention)))
        print('Len metadata: ' + str(len(metadata)))

        if not len(attention) == len(metadata):
            error_list.append(metadata[0])
            continue
        metadata: TileMetadata = metadata[0]

        line_print(
            'Calculating metrics for: ' + metadata.experiment_name + ' - ' + metadata.get_formatted_well(long=True))

        n, bins, otsu_index, otsu_threshold, entropy_attention, entropy_hist = attention_metrics(attention=attention,
                                                                                                 hist_bins_override=hist_bins_override,
                                                                                                 normalized=normalized)

        n_list.append(n)
        bins_list.append(bins)
        metadata_list.append(metadata)
        otsu_index_list.append(otsu_index)
        otsu_threshold_list.append(otsu_threshold)
        entropy_attention_list.append(entropy_attention)
        entropy_hist_list.append(entropy_hist)

        del otsu_index, otsu_threshold, entropy_attention, entropy_hist, attention, metadata, n, bins

    log.write('Finished evaluating attention metrics.')
    return metadata_list, n_list, bins_list, otsu_index_list, otsu_threshold_list, entropy_attention_list, entropy_hist_list, error_list


def bayesian_bootstrap_attention(attention: np.ndarray, n_replications: int = 1000, resample_size: int = 100,
                                 seed: int = None):
    ###
    # DEPRECATED
    ###

    from imports import bayesian_bootstrap
    bootstrap = bayesian_bootstrap.bayesian_bootstrap(X=np.asarray(attention, dtype=np.float64), statistic=np.mean,
                                                      n_replications=n_replications, resample_size=resample_size,
                                                      seed=seed)
    bootstrap = np.asarray(bootstrap, dtype=np.float64)
    threshold_index = utils.lecture_otsu(n=bootstrap)
    threshold = bootstrap[threshold_index]

    return bootstrap, threshold, threshold_index


def bayesian_bootstrap_attention_batch(all_attentions: [np.ndarray], X_metadata: [[TileMetadata]],
                                       n_replications: int = 1000, resample_size: int = 100,
                                       seed: int = None) -> [np.ndarray]:
    ###
    # DEPRECATED
    ###

    bootstrap_list = []
    threshold_indices_list = []
    metadata_list = []
    assert len(all_attentions) == len(X_metadata)

    log.write('Bootstrapping ' + str(len(all_attentions)) + ' attentions. Replications: ' + str(
        n_replications) + '. Re-samples: ' + str(resample_size))
    print('')
    for (attention, metadata) in zip(all_attentions, X_metadata):
        assert len(attention) == len(metadata)
        metadata: TileMetadata = metadata[0]
        line_print(
            'Bootstrapping attentions: ' + metadata.experiment_name + ' - ' + metadata.get_formatted_well(long=True))
        bootstrap, threshold, threshold_index = bayesian_bootstrap_attention(attention=attention,
                                                                             n_replications=n_replications,
                                                                             resample_size=resample_size, seed=seed)

        bootstrap_list.append(bootstrap)
        threshold_indices_list.append(threshold_index)
        metadata_list.append(metadata)
        del attention, bootstrap, threshold, threshold_index, metadata

    log.write('Finished bootstrapping attention histograms.')
    return bootstrap_list, threshold_indices_list, metadata_list


def get_cells_per_attention(all_attentions: [[float]], X_metadata: [[TileMetadata]], normalized: bool = True):
    assert len(all_attentions) == len(X_metadata)

    distributions = []
    for (attention, metadata) in zip(all_attentions, X_metadata):
        assert len(attention) == len(metadata)

        current_distributions = {}
        current_distributions['attention_min'] = attention.min()
        current_distributions['attention_max'] = attention.max()

        if normalized and attention.max() > 0:
            attention = (attention - np.min(attention)) / np.ptp(attention)

        experiment_name = metadata[0].experiment_name
        well_name = metadata[0].get_formatted_well()
        log.write('Calculation cell / attention density for: ' + experiment_name + '-' + well_name)

        attention = np.array(attention)
        metadata = np.array(metadata)

        sort_indices = np.argsort(attention)
        attention = attention[sort_indices]
        metadata = metadata[sort_indices]
        del sort_indices

        nucleus_distributions = []
        neuron_distributions = []
        oligo_distributions = []
        for (current_attention, current_metadata) in zip(attention, metadata):
            nucleus_distributions.append((current_attention, current_metadata.count_nuclei))
            neuron_distributions.append((current_attention, current_metadata.count_neurons))
            oligo_distributions.append((current_attention, current_metadata.count_oligos))

        current_distributions['nucleus'] = nucleus_distributions
        current_distributions['neuron'] = neuron_distributions
        current_distributions['oligo'] = oligo_distributions

        distributions.append(current_distributions)

    return distributions
