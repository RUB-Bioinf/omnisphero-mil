'''
This submodule contains miscellaneous functions to compute and analyze metrics. 
Originally used in the JoshNet
'''

# IMPORTS
#########

import itertools
import math
import os
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

import models
from models import BaselineMIL
from util import log
from util import utils
# matplotlib.use('Agg')
# plt.style.use('ggplot')
# FUNCTIONS
###########
from util.utils import get_plt_as_tex
from util.utils import line_print


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


def plot_binary_roc_curves(history, save_path: str, include_raw: bool = False, include_tikz: bool = False,
                           clamp: float = None):
    if include_raw:
        plot_metric(history, 'train_roc_auc', save_path, include_tikz=include_tikz, clamp=clamp)
        plot_metric(history, 'val_roc_auc', save_path, include_tikz=include_tikz, clamp=clamp)
    plot_metric(history, 'val_roc_auc', save_path, 'train_roc_auc', include_tikz=include_tikz, clamp=clamp)


def plot_dice_scores(history, save_path: str, include_raw: bool = False, include_tikz: bool = False,
                     clamp: float = None):
    if include_raw:
        plot_metric(history, 'train_dice_score', save_path, include_tikz=include_tikz, clamp=clamp)
        plot_metric(history, 'val_dice_score', save_path, include_tikz=include_tikz, clamp=clamp)
    plot_metric(history, 'val_dice_score', save_path, 'train_dice_score', include_tikz=include_tikz, clamp=clamp)


def plot_losses(history, save_path: str, include_raw: bool = False, include_tikz: bool = False, clamp: float = None):
    ''' takes a history object and plots the losses
    '''
    if include_raw:
        plot_metric(history, 'val_loss', save_path, include_tikz=include_tikz, clamp=clamp)
        plot_metric(history, 'train_loss', save_path, include_tikz=include_tikz, clamp=clamp)
    plot_metric(history, 'val_loss', save_path, 'train_loss', include_tikz=include_tikz, clamp=clamp)


def plot_metric(history, metric_name: str, out_dir: str, second_metric_name: str = None, dpi: int = 350,
                include_tikz: bool = False, clamp: float = None):
    metric_values = [i[metric_name] for i in history]
    metric_title = _get_metric_title(metric_name)
    metric_color, metric_type = _get_metric_color(metric_name)

    if clamp is not None:
        metric_values = [max(min(i, clamp), clamp * -1) for i in metric_values]

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

        if clamp is not None:
            second_metric_values = [max(min(i, clamp), clamp * -1) for i in second_metric_values]

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


def plot_accuracies(history, out_dir: str, dpi: int = 600, include_tikz: bool = False, clamp: float = None):
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


def _get_metric_title(metric_name: str):
    metric_name = metric_name.replace('val_', '')
    metric_name = metric_name.replace('train_', '')

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
        added_attention_tiles = 0
        for current_index in max_attention_indexes[0]:
            max_attention_tile = raw_bag[current_index]
            max_attention_tile = max_attention_tile.astype('uint8')
            max_attention_tile = outline_rgb_array(max_attention_tile, None, None, outline=2,
                                                   override_colormap=[255, 255, 255])
            if y_bag_true == 1:
                if y_bag == 1:
                    max_attention_tilesTP.append(max_attention_tile)
                    added_attention_tiles = added_attention_tiles + 1
                elif y_bag == 0:
                    max_attention_tilesFP.append(max_attention_tile)
                    added_attention_tiles = added_attention_tiles + 1
            elif y_bag_true == 0:
                if y_bag == 1:
                    max_attention_tilesFN.append(max_attention_tile)
                    added_attention_tiles = added_attention_tiles + 1
                elif y_bag == 0:
                    max_attention_tilesTN.append(max_attention_tile)
                    added_attention_tiles = added_attention_tiles + 1
        assert len(max_attention_indexes[0]) == added_attention_tiles

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

            rgb = np.dstack((r, g, b))
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
    for (max_attention_tiles, metric_name) in zip(
            [max_attention_tilesTP, max_attention_tilesFP, max_attention_tilesFN, max_attention_tilesTN],
            ['TP', 'FP', 'FN', 'TN']):
        log.write('Saving ' + str(len(max_attention_tiles)) + ' tiles for metric ' + metric_name)

        # Writing as png images, if they exist
        if len(max_attention_tiles) > 0:
            out_image = fuse_image_tiles(images=max_attention_tiles, image_width=image_width, image_height=image_height)
            max_attention_file = out_dir + 'max_attention_' + metric_name + '.png'
            plt.imsave(max_attention_file, out_image)

        # Writing as text files
        f = open(out_dir + 'max_attention_' + metric_name + '.txt', 'w')
        f.write('Tile count: ' + str(len(max_attention_tiles)))
        f.close()


def calculate_dice_score(TP: int, FP: int, FN: int):
    TP = float(math.floor(TP))
    FP = float(math.floor(FP))
    FN = float(math.floor(FN))

    return (2 * TP) / (2 * TP + FP + FN)


def fuse_image_tiles(images: [np.ndarray], image_width: int, image_height: int):
    # assert image_width is None
    # assert image_height is None

    image_count = len(images)
    assert image_count > 0

    if image_count == 1:
        # Special case, if there is only 1 image in the list
        return images[0].astype(np.uint8)

    if image_count == 2:
        # Special case, if there are only 2 images in the list
        combined_img = np.zeros((image_height, image_width * 2, 3), dtype=np.uint8)
        combined_img[0:image_width, 0:image_height] = images[0].astype(np.uint8)
        combined_img[0:image_width, image_height:image_height * 2] = images[1].astype(np.uint8)
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
        combined_img[x * image_width:x * image_width + image_width,
        y * image_height:y * image_height + image_height] = current_img

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

    image[0:outline, :] = colormap
    image[width - outline:width, :] = colormap
    image[:, 0:outline] = colormap
    image[:, height - outline:height] = colormap
    image = image.astype('uint8')

    return image
