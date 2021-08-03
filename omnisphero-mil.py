import math
import os
import random
import sys
from datetime import datetime
from sys import getsizeof

import numpy as np
import torch

import hardware
import loader
import mil_metrics
import models
import torch_callbacks
from models import BaselineMIL
from util import log
from util import sample_preview
from util import utils
from util.omnisphero_data_loader import OmniSpheroDataLoader
from util.utils import line_print
from util.utils import shuffle_and_split_data

# On windows, if there's not enough RAM:
# https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial/issues/10


default_source_dir_win = "U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_win"
default_out_dir_win_base = "U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\win"

default_source_dirs_unix = [
    # New CNN
    "/mil/oligo-diff/training_data/curated_linux/EFB18",
    "/mil/oligo-diff/training_data/curated_linux/ESM36",
    "/mil/oligo-diff/training_data/curated_linux/ELS411",
    "/mil/oligo-diff/training_data/curated_linux/ELS517",
    "/mil/oligo-diff/training_data/curated_linux/ELS637",
    "/mil/oligo-diff/training_data/curated_linux/ELS681",
    "/mil/oligo-diff/training_data/curated_linux/ELS682",
    "/mil/oligo-diff/training_data/curated_linux/ELS719",
    "/mil/oligo-diff/training_data/curated_linux/ELS744"
]

ideal_source_dirs_unix = [
    # New CNN
    "/mil/oligo-diff/training_data/curated_linux/ESM36",
    "/mil/oligo-diff/training_data/curated_linux/ELS517",
    "/mil/oligo-diff/training_data/curated_linux/ELS637",
    "/mil/oligo-diff/training_data/curated_linux/ELS681",
    "/mil/oligo-diff/training_data/curated_linux/ELS682"
]

default_out_dir_unix_base = "/mil/oligo-diff/models/linux"

# normalize_enum is an enum to determine normalisation as follows:
# 0 = no normalisation
# 1 = normalize every cell between 0 and 255 (8 bit)
# 2 = normalize every cell individually with every color channel independent
# 3 = normalize every cell individually with every color channel using the min / max of all three
# 4 = normalize every cell but with bounds determined by the brightest cell in the bag
# 5 = z-score every cell individually with every color channel independent
# 6 = z-score every cell individually with every color channel using the mean / std of all three
# 7 = z-score every cell individually with every color channel independent using all samples in the bag
# 8 = z-score every cell individually with every color channel using the mean / std of all three from all samples in the bag
normalize_enum_default = 3

max_workers_default = 5


def train_model(training_label: str, source_dirs: [str], loss_function: str, device_ordinals: [int],
                epochs: int = 3, max_workers: int = max_workers_default, normalize_enum: int = normalize_enum_default,
                out_dir: str = None, gpu_enabled: bool = False, invert_bag_labels: bool = False,
                shuffle_data_loaders: bool = True, model_enable_attention: bool = False, model_use_max: bool = True,
                repack_percentage: float = 0.0, global_log_dir: str = None, optimizer: str = 'adadelta',
                clamp_min: float = None, clamp_max: float = None, positive_bag_min_samples: int = None,
                tile_constraints_0: [int] = loader.tile_constraints_none,
                tile_constraints_1: [int] = loader.tile_constraints_none,
                data_split_percentage_validation: float = 0.3, data_split_percentage_test: float = 0.15,
                ):
    if out_dir is None:
        out_dir = source_dirs[0] + os.sep + 'training_results'

    out_dir = out_dir + os.sep + training_label + os.sep
    loading_preview_dir = out_dir + os.sep + 'loading_previews' + os.sep
    metrics_dir = out_dir + os.sep + 'metrics' + os.sep
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(loading_preview_dir, exist_ok=True)

    print('Model classification - Use Max: ' + str(model_use_max))
    print('Model classification - Use Attention: ' + str(model_enable_attention))

    print('Saving logs and protocols to: ' + out_dir)
    # Logging params and args
    protocol_f = open(out_dir + os.sep + 'protocol.txt', 'w')
    protocol_f.write('Start time: ' + utils.gct())
    protocol_f.write('\n\n == General Params ==')
    protocol_f.write('\nSource dirs: ' + str(len(source_dirs)))
    protocol_f.write('\nLoss function: ' + loss_function)
    protocol_f.write('\nDevice ordinals: ' + str(device_ordinals))
    protocol_f.write('\nEpochs: ' + str(epochs))
    protocol_f.write('\nShuffle data loader: ' + str(shuffle_data_loaders))
    protocol_f.write('\nMax Loader Workers: ' + str(max_workers))
    protocol_f.write('\nNormalize Enum: ' + str(normalize_enum))
    protocol_f.write('\nGPU Enabled: ' + str(gpu_enabled))
    protocol_f.write('\nInvert Bag Labels: ' + str(invert_bag_labels))
    protocol_f.write('\nRepack: Percentage: ' + str(repack_percentage))
    protocol_f.write('\nRepack: Minimum Positive Samples: ' + str(positive_bag_min_samples))
    protocol_f.write('\nClamp Min: ' + str(clamp_min))
    protocol_f.write('\nClamp Max: ' + str(clamp_max))

    global_log_filename = None
    local_log_filename = out_dir + os.sep + 'log.txt'
    log.add_file(local_log_filename)
    if global_log_dir is not None:
        global_log_filename = global_log_dir + os.sep + 'log-' + training_label + '.txt'
        os.makedirs(global_log_dir, exist_ok=True)
        log.add_file(global_log_filename)

    # PREPARING DATA
    protocol_f.write('\n\n == Directories ==')
    protocol_f.write('\nGlobal Log dir: ' + str(global_log_dir))
    protocol_f.write('\nLocal Log dir: ' + str(local_log_filename))
    protocol_f.write('\nOut dir: ' + str(out_dir))
    protocol_f.write('\nMetrics dir: ' + str(metrics_dir))
    protocol_f.write('\nPreview tiles: ' + str(loading_preview_dir))

    protocol_f.write('\n\n == GPU Status ==')
    for line in hardware.print_gpu_status(silent=True):
        protocol_f.write(line)
        log.write(line)

    ################
    # LOADING START
    ################
    loading_start_time = datetime.now()
    X, y, y_tiles, X_raw, errors, loaded_files_list = loader.load_bags_json_batch(batch_dirs=source_dirs,
                                                                                  max_workers=max_workers,
                                                                                  include_raw=True,
                                                                                  constraints_0=tile_constraints_0,
                                                                                  constraints_1=tile_constraints_1,
                                                                                  normalize_enum=normalize_enum)
    X = [np.einsum('bhwc->bchw', bag) for bag in X]
    X_raw = [np.einsum('bhwc->bchw', bag) for bag in X_raw]
    # Hint: Dim should be (xxx, 3, 150, 150)
    loading_time = utils.get_time_diff(loading_start_time)
    log.write('Loading finished in: ' + str(loading_time))

    # Finished loading. Printing errors and data
    f = open(out_dir + 'loading_errors.txt', 'w')
    for e in errors:
        f.write(str(e))
    f.close()

    # Saving one random image from every bag to the disk
    log.write('Writing loading preview samples to: ' + loading_preview_dir)
    print('\n')
    for i in range(len(X)):
        line_print('Writing loading preview: ' + str(i + 1) + '/' + str(len(X)), include_in_log=False)
        current_x: np.ndarray = X[i]
        j = random.randint(0, current_x.shape[0] - 1)

        preview_image_file_base = loading_preview_dir + 'preview_' + str(i) + '-' + str(j) + '_' + str(y[i])
        current_x = current_x[j]
        current_x: np.ndarray = np.einsum('abc->cba', current_x)
        if current_x.min() >= 0 and current_x.max() <= 1:
            # Normalized Image
            preview_image_file = preview_image_file_base + '.png'
            sample_preview.save_normalized_rgb(current_x, preview_image_file)
        if normalize_enum >= 5:
            sample_preview.save_z_scored_image(current_x, dim_x=150, dim_y=150,
                                               fig_titles=['r (Nuclei)', 'g (Oligos)', 'b (Neurites)'],
                                               filename=preview_image_file_base + '-z.png',
                                               min=-3.0, max=3.0, normalize_enum=normalize_enum)

        del current_x
    print('\n')

    # Calculating Bag Size and possibly inverting labels
    log.write('Finished loading data. Number of bags: ' + str(len(X)) + '. Number of labels: ' + str(len(y)))
    X_size = 0
    X_size_raw = 0
    for i in range(len(X)):
        X_size = X_size + X[i].nbytes
        X_size_raw = X_size_raw + X_raw[i].nbytes
        if invert_bag_labels:
            y[i] = int(not y[i])
            y_tiles[i] = not y_tiles[i]
    f.close()

    X_s = utils.convert_size(X_size)
    y_s = utils.convert_size(getsizeof(y))

    log.write("X-size in memory: " + str(X_s))
    log.write("y-size in memory: " + str(y_s))

    protocol_f.write('\n\n == Loaded Data ==')
    protocol_f.write('\nNumber of Bags: ' + str(len(X)))
    protocol_f.write('\nBags with label 0: ' + str(len(np.where(np.asarray(y) == 0)[0])))
    protocol_f.write('\nBags with label 1: ' + str(len(np.where(np.asarray(y) == 1)[0])))
    protocol_f.write("\nX-size in memory: " + str(X_s))
    protocol_f.write("\ny-size in memory: " + str(y_s))

    # Printing more data
    f = open(out_dir + 'loading_data_statistics.csv', 'w')
    for i in range(len(loaded_files_list)):
        f.write(str(i) + ';' + loaded_files_list[i] + '\n')
    f.write('\n\nX-size in memory: ' + str(X_s))
    f.write('\n\ny-size in memory: ' + str(y_s))
    f.write('\n\nLoading time: ' + str(loading_time))
    f.close()
    del X_s, y_s, X_size, f

    if len(X) == 0:
        log.write('WARNING: NO DATA LOADED')
        protocol_f.write('\n\nWARNING: NO DATA LOADED')
        return

    # Printing Bag Shapes
    # Setting up bags for MIL
    if repack_percentage > 0:
        print_bag_metadata(X, y, y_tiles, file_name=out_dir + 'bags_pre-packed.csv')
        X, X_raw, y, y_tiles = loader.repack_bags_merge(X, X_raw, y, repack_percentage=repack_percentage,
                                                        positive_bag_min_samples=positive_bag_min_samples)
        print_bag_metadata(X, y, y_tiles, file_name=out_dir + 'bags_repacked.csv')
    else:
        print_bag_metadata(X, y, y_tiles, file_name=out_dir + 'bags.csv')

    # Setting up datasets
    dataset, input_dim = loader.convert_bag_to_batch(X, y, y_tiles)
    log.write('Detected input dim: ' + str(input_dim))
    del X, y

    # Train-Test Split
    log.write('Shuffling and splitting data into train and val set')
    test_data = []
    test_data_tiles = None
    training_data, validation_data = shuffle_and_split_data(dataset,
                                                            split_percentage=data_split_percentage_validation)
    if data_split_percentage_test is not None and data_split_percentage_test > 0:
        training_data, test_data = shuffle_and_split_data(dataset=training_data,
                                                          split_percentage=data_split_percentage_test)
    del dataset

    f = open(out_dir + 'data-distribution.txt', 'w')
    training_data_tiles: int = sum([training_data[i][0].shape[0] for i in range(len(training_data))])
    validation_data_tiles: int = sum([validation_data[i][0].shape[0] for i in range(len(validation_data))])
    log.write('Training data: ' + str(training_data_tiles) + ' tiles over ' + str(len(training_data)) + ' bags.')
    log.write('Validation data: ' + str(validation_data_tiles) + ' tiles over ' + str(len(validation_data)) + ' bags.')
    protocol_f.write(
        '\nTraining data: ' + str(training_data_tiles) + ' tiles over ' + str(len(training_data)) + ' bags.')
    protocol_f.write(
        '\nValidation data: ' + str(validation_data_tiles) + ' tiles over ' + str(len(validation_data)) + ' bags.')
    f.write('Training data: ' + str(training_data_tiles) + ' tiles over ' + str(len(training_data)) + ' bags.\n')
    f.write('Validation data: ' + str(validation_data_tiles) + ' tiles over ' + str(len(validation_data)) + ' bags.\n')

    if data_split_percentage_test is not None:
        test_data_tiles: int = sum([test_data[i][0].shape[0] for i in range(len(test_data))])
        log.write('Test data: ' + str(test_data_tiles) + ' tiles over ' + str(len(test_data)) + ' bags.')
        f.write('Test data: ' + str(test_data_tiles) + ' tiles over ' + str(len(test_data)) + ' bags.\n')
        protocol_f.write('Test data: ' + str(test_data_tiles) + ' tiles over ' + str(len(test_data)) + ' bags.')
    f.close()

    # Loading Hardware Device
    device = hardware.get_hardware_device(gpu_preferred=gpu_enabled)
    log.write('Selected device: ' + str(device))

    # Setting up Model
    log.write('Setting up model...')
    accuracy_function = 'binary'
    model = BaselineMIL(input_dim=input_dim, device=device,
                        use_max=model_use_max,
                        enable_attention=model_enable_attention,
                        device_ordinals=device_ordinals,
                        loss_function=loss_function,
                        accuracy_function=accuracy_function)

    # Loader args
    loader_kwargs = {}
    data_loader_cores = math.ceil(os.cpu_count() * 0.5 + 1)
    data_loader_cores = 0
    data_loader_pin_memory = False
    if torch.cuda.is_available():
        #    #model.cuda()
        loader_kwargs = {'num_workers': data_loader_cores, 'pin_memory': data_loader_pin_memory}

    model_optimizer = models.choose_optimizer(model, selection=optimizer)
    log.write('Finished loading data and model')
    log.write('Optimizer: ' + str(model_optimizer))

    ################
    # DATA START
    ################
    # Data Generators
    test_dl = None
    train_dl = OmniSpheroDataLoader(training_data, batch_size=1, shuffle=shuffle_data_loaders, **loader_kwargs)
    validation_dl = OmniSpheroDataLoader(validation_data, batch_size=1, shuffle=shuffle_data_loaders, **loader_kwargs)
    if data_split_percentage_test is not None:
        test_dl = OmniSpheroDataLoader(test_data, batch_size=1, shuffle=shuffle_data_loaders, **loader_kwargs)

    del training_data, validation_data, test_data
    # test_dl = DataLoader(test_data, batch_size=1, shuffle=True, **loader_kwargs)

    # Callbacks
    callbacks = []
    callbacks.append(torch_callbacks.EarlyStopping(epoch_threshold=int(epochs / 5 + 1)))
    callbacks.append(torch_callbacks.UnreasonableLossCallback(loss_max=40.0))

    protocol_f.write('\n\n == Model Information==')
    protocol_f.write('\nDevice Ordinals: ' + str(device_ordinals))
    protocol_f.write('\nClassification: Use Max: ' + str(model_use_max))
    protocol_f.write('\nClassification: Use Attention: ' + str(model_enable_attention))
    protocol_f.write('\nInput dim: ' + str(input_dim))
    protocol_f.write('\ntorch Device: ' + str(device))
    protocol_f.write('\nLoss Function: ' + str(loss_function))
    protocol_f.write('\nAccuracy Function: ' + str(accuracy_function))
    protocol_f.write('\nModel classification - Use Max: ' + str(model_use_max))
    protocol_f.write('\nModel classification - Use Attention: ' + str(model_enable_attention))
    protocol_f.write('\n\nData Loader - Cores: ' + str(data_loader_cores))
    protocol_f.write('\nData Loader - Pin Memory: ' + str(data_loader_pin_memory))
    protocol_f.write('\nCallback Count: ' + str(len(callbacks)))
    protocol_f.write('\nCallbacks: ' + str(callbacks))
    protocol_f.write('\nBuilt Optimizer: ' + str(model_optimizer))
    protocol_f.close()
    del protocol_f

    # Printing the model
    f = open(out_dir + os.sep + 'model.txt', 'w')
    f.write(str(model))
    f.close()

    ################
    # TRAINING START
    ################
    log.write('Start of training for ' + str(epochs) + ' epochs.')
    log.write('Training: "' + training_label + '"!')
    history, history_keys, model_save_path_best = models.fit(model=model, optimizer=model_optimizer, epochs=epochs,
                                                             training_data=train_dl,
                                                             validation_data=validation_dl,
                                                             out_dir_base=out_dir,
                                                             checkpoint_interval=None,
                                                             clamp_min=clamp_min, clamp_max=clamp_max,
                                                             callbacks=callbacks)
    log.write('Finished training!')
    del train_dl

    log.write('Plotting and saving loss and acc plots...')
    mil_metrics.write_history(history, history_keys, metrics_dir)
    mil_metrics.plot_losses(history, metrics_dir, include_raw=True, include_tikz=True, clamp=2.0)
    mil_metrics.plot_accuracy(history, metrics_dir, include_raw=True, include_tikz=True)
    mil_metrics.plot_accuracy_tiles(history, metrics_dir, include_raw=True, include_tikz=True)
    mil_metrics.plot_accuracies(history, metrics_dir, include_tikz=True)

    ##################
    # TESTING START
    ##################
    log.write('Testing best model on validation and test data to determine performance')
    test_dir = out_dir + 'metrics' + os.sep + 'performance-validation-data' + os.sep
    test_model(model, model_save_path_best, model_optimizer, data_loader=validation_dl, out_dir=test_dir, X_raw=X_raw,
               y_tiles=y_tiles)
    if data_split_percentage_test > 0:
        test_dir = out_dir + 'metrics' + os.sep + 'performance-test-data' + os.sep
        test_model(model, model_save_path_best, model_optimizer, data_loader=test_dl, out_dir=test_dir, X_raw=X_raw,
                   y_tiles=y_tiles)

    ##################################
    # FINISHED TRAINING - Cleaning up
    ##################################
    log.write("Finished training and testing for this run. Job's done!")
    del validation_dl, test_dl, X_raw, y_tiles

    log.remove_file(local_log_filename)
    if global_log_dir is not None:
        log.remove_file(global_log_filename)
    del f

    # Run finished
    # Noting more to do beyond this point


def test_model(model, model_save_path_best, model_optimizer, data_loader, X_raw, y_tiles, out_dir):
    attention_out_dir = out_dir + os.sep + 'attention' + os.sep
    os.makedirs(attention_out_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Get best saved model from this run
    model, model_optimizer, _, _ = models.load_checkpoint(model_save_path_best, model, model_optimizer)
    y_hats, y_pred, y_true, y_samples_pred, y_samples_true, _, _ = models.get_predictions(model, data_loader)

    # Flattening sample predictions
    y_samples_pred = [item for sublist in y_samples_pred for item in sublist]
    y_samples_true = [item for sublist in y_samples_true for item in sublist]

    # Saving attention scores for every tile in every bag!
    log.write('Writing attention scores to: ' + attention_out_dir)
    try:
        mil_metrics.save_tile_attention(out_dir=attention_out_dir, model=model, normalized=True,
                                        dataset=data_loader, X_raw=X_raw, y_tiles=y_tiles)
        mil_metrics.save_tile_attention(out_dir=attention_out_dir, model=model, normalized=False,
                                        dataset=data_loader, X_raw=X_raw, y_tiles=y_tiles)
    except Exception as e:
        log.write('Failed to save the attention score for every tile!')
        f = open(attention_out_dir + 'attention-error.txt', 'a')
        f.write('\nError: ' + str(e.__class__) + '\n' + str(e))
        f.close()
    log.write('Finished writing attention scores.')

    # Confidence Matrix
    try:
        log.write('Saving Confidence Matrix')
        log.write(str(e))
        mil_metrics.plot_conf_matrix(y_true, y_pred, out_dir, target_names=['Control', 'Oligo Diff'],
                                     normalize=False)
        mil_metrics.plot_conf_matrix(y_true, y_pred, out_dir, target_names=['Control', 'Oligo Diff'], normalize=True)
    except Exception as e:
        log.write('Failed to save the Confidence Matrix!')
        log.write(str(e))
        f = open(attention_out_dir + 'confidence-errors.txt', 'a')
        f.write('\nError: ' + str(e.__class__) + '\n' + str(e))
        f.close()
    log.write('Finished writing confidence Matrix.')

    # ROC Curve
    log.write('Computing and plotting binary ROC-Curve.')
    try:
        fpr, tpr, thresholds = mil_metrics.binary_roc_curve(y_true, y_hats)
        mil_metrics.plot_binary_roc_curve(fpr, tpr, thresholds, out_dir, 'Bags')

        fpr, tpr, thresholds = mil_metrics.binary_roc_curve(y_samples_true, y_samples_pred)
        mil_metrics.plot_binary_roc_curve(fpr, tpr, thresholds, out_dir, 'Samples')
    except Exception as e:
        log.write('Failed to save the ROC curve!')
        log.write(str(e))
        f = open(out_dir + 'roc-error.txt', 'a')
        f.write('\nError: ' + str(e.__class__) + '\n' + str(e))
        f.close()
    log.write('Finished writing confidence binary ROC-Curve.')

    # PR Curve
    log.write('Computing and plotting binary PR-Curve.')
    try:
        precision, recall, thresholds = mil_metrics.binary_pr_curve(y_true, y_hats)
        mil_metrics.plot_binary_pr_curve(precision, recall, thresholds, y_true, out_dir, 'Bags')

        precision, recall, thresholds = mil_metrics.binary_pr_curve(y_samples_true, y_samples_pred)
        mil_metrics.plot_binary_pr_curve(precision, recall, thresholds, y_samples_true, out_dir, 'Samples')
    except Exception as e:
        log.write('Failed to save the PR curve!')
        log.write(str(e))
        f = open(out_dir + 'pr-error.txt', 'a')
        f.write('\nError: ' + str(e.__class__) + '\n' + str(e))
        f.close()
    log.write('Finished writing confidence binary PR-Curve.')


def print_bag_metadata(X, y, y_tiles, file_name: str):
    f = open(file_name, 'w')
    f.write(';Bag;Label;Tile Labels (Sum);Memory Raw;Memory Formatted;Shape;Shape;Shape;Shape')

    y0 = len(np.where(np.asarray(y) == 0)[0])
    y1 = len(np.where(np.asarray(y) == 1)[0])

    x_size = 0
    bag_count = 0
    for i in range(len(X)):
        current_X = X[i]
        x_size = x_size + current_X.nbytes
        x_size_converted = utils.convert_size(current_X.nbytes)

        shapes = ';'.join([str(s) for s in current_X.shape])
        bag_count = bag_count + current_X.shape[0]

        # print('Bag #' + str(i + 1) + ': ', shapes, ' -> label: ', str(y[i]))
        f.write('\n;' + str(i) + ';' + str(y[i]) + ';' + str(sum(y_tiles[i])) + ';' + str(
            current_X.nbytes) + ';' + x_size_converted + ';' + shapes)

    f.write('\n' + 'Sum;' + str(len(X)) + ';0: ' + str(y0) + ';' + str(x_size) + ';' + utils.convert_size(
        x_size) + ';' + str(bag_count))
    f.write('\n;;1: ' + str(y1))
    f.close()


def main(debug: bool = False):
    if sys.platform == 'win32':
        debug = True
    print('Debug mode: ' + str(debug))

    current_epochs = 800
    current_max_workers = 35
    default_out_dir_base = default_out_dir_unix_base
    current_sources_dir = default_source_dirs_unix
    current_gpu_enabled = True
    if debug:
        current_sources_dir = [current_sources_dir[0]]
        current_epochs = 5

    # Preparing for model loading
    current_device_ordinals = models.device_ordinals_ehrlich

    if sys.platform == 'win32':
        current_global_log_dir = 'U:\\bioinfdata\\work\\OmniSphero\\Sciebo\\HCA\\00_Logs\\mil_log\\win\\'
        log.add_file('U:\\bioinfdata\\work\\OmniSphero\\Sciebo\\HCA\\00_Logs\\mil_log\\win\\all_logs.txt')

        current_max_workers = 10
        current_sources_dir = [default_source_dir_win]
        default_out_dir_base = default_out_dir_win_base
        current_gpu_enabled = False
        current_device_ordinals = models.device_ordinals_local
    else:
        current_global_log_dir = '/Sciebo/HCA/00_Logs/mil_log/linux/'

    current_out_dir = default_out_dir_base + os.sep
    os.makedirs(current_out_dir, exist_ok=True)

    log.write('Starting Training...')
    if debug:
        train_model(source_dirs=current_sources_dir, out_dir=current_out_dir, epochs=current_epochs,
                    max_workers=current_max_workers, gpu_enabled=current_gpu_enabled,
                    device_ordinals=current_device_ordinals,
                    normalize_enum=0,
                    training_label='debug',
                    global_log_dir=current_global_log_dir,
                    repack_percentage=0.1,
                    model_use_max=False,
                    model_enable_attention=True,
                    invert_bag_labels=False,
                    positive_bag_min_samples=0,
                    loss_function='binary_cross_entropy'
                    )
    else:
        for l in ['binary_cross_entropy', 'negative_log_bernoulli']:
            for o in ['adadelta', 'adam']:
                for r in [0.15]:
                    for i in [5, 7, 6, 3, 4]:
                        train_model(source_dirs=current_sources_dir, out_dir=current_out_dir, epochs=current_epochs,
                                    max_workers=current_max_workers, gpu_enabled=current_gpu_enabled,
                                    normalize_enum=i,
                                    training_label='strongly-constrained-attention-' + o + '-' + l + '-normalize-' + str(
                                        i) + 'repack-' + str(r),
                                    global_log_dir=current_global_log_dir,
                                    invert_bag_labels=False,
                                    loss_function=l,
                                    repack_percentage=r,
                                    optimizer=o,
                                    model_enable_attention=True,
                                    model_use_max=False,
                                    positive_bag_min_samples=5,
                                    tile_constraints_0=loader.tile_constraints_oligos,
                                    tile_constraints_1=loader.tile_constraints_oligos,
                                    device_ordinals=current_device_ordinals
                                    )
    log.write('Finished every training!')


if __name__ == '__main__':
    print("Training OmniSphero MIL")
    debug: bool = False

    hardware.print_gpu_status()

    main(debug=False)
