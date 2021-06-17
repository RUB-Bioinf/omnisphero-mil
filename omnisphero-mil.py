import math
import os
import random
import sys
from datetime import datetime
from sys import getsizeof
from sys import platform as _platform

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import hardware
import loader
import mil_metrics
import models
from models import BaselineMIL
from util import utils
from util.utils import shuffle_and_split_data

default_source_dir_win = "U:\\bioinfdata\\work\\OmniSphero\\mil\\migration\\training_data\\curated_win"
default_out_dir_win_base = "U:\\bioinfdata\\work\\OmniSphero\\mil\\migration\\models\\win"

default_source_dirs_unix = ["/mil/migration/training_data/curated_linux/EFB18",
                            "/mil/migration/training_data/curated_linux/esm49",
                            "/mil/migration/training_data/curated_linux/jk242",
                            "/mil/migration/training_data/curated_linux/mp149"
                            ]

default_out_dir_unix_base = "/mil/migration/models/linux"

# normalize_enum is an enum to determine normalisation as follows:
# 0 = no normalisation
# 1 = normalize every cell between 0 and 255 (8 bit)
# 2 = normalize every cell individually with every color channel independent
# 3 = normalize every cell individually with every color channel using the min / max of all three
# 4 = normalize every cell but with bounds determined by the brightest cell in the same well
normalize_enum_default = 3

max_workers_default = 5


def main(source_dirs: [str], loss_function: str, epochs: int = 3, max_workers: int = max_workers_default,
         normalize_enum: int = normalize_enum_default, out_dir: str = None, gpu_enabled: bool = False,
         invert_bag_labels: bool = False):
    if out_dir is None:
        out_dir = source_dirs[0] + os.sep + 'training_results'
    print('Saving logs and protocols to: ' + out_dir)

    # PREPARING DATA
    # out_dir = source_dir + os.sep + 'train' + os.sep + 'debug' + os.sep
    metrics_dir = out_dir + os.sep + 'metrics' + os.sep
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    ################
    # LOADING START
    ################
    loading_start_time = datetime.now()
    X, y, errors, loaded_files_list = loader.load_bags_json_batch(batch_dirs=source_dirs, max_workers=max_workers,
                                                                  normalize_enum=normalize_enum)
    X = [np.einsum('bhwc->bchw', bag) for bag in X]
    # Hint: Dim should be (xxx, 3, 150, 150)
    loading_time = utils.get_time_diff(loading_start_time)
    print('Loading finished in: ' + str(loading_time))

    # Finished loading. Printing errors and data
    f = open(out_dir + 'loading_errors.txt', 'w')
    for e in errors:
        f.write(str(e))
    f.close()

    loading_preview_dir = out_dir + os.sep + 'loading_previews' + os.sep
    os.makedirs(loading_preview_dir, exist_ok=True)
    for i in range(len(X)):
        current_x = X[i]
        j = random.randint(0, current_x.shape[0] - 1)

        try:
            current_x = current_x[j]
            preview_image_file = loading_preview_dir + 'preview_' + str(i) + '-' + str(j) + '_' + str(y[i]) + '.png'

            current_x = loader.normalize_np(current_x, current_x.min(), current_x.max())
            current_x = current_x * 255
            current_x = current_x.astype(np.uint8)
            current_x = np.einsum('abc->cba', current_x)
            plt.imsave(preview_image_file, current_x)
        except Exception as e:
            # TODO display stacktrace
            preview_error_text = str(e.__class__.__name__) + ': "' + str(e) + '"'
            print('ERROR WHILE SMOTE!! (Reverting to un-smote)')
            print(preview_error_text)
        del current_x

    print('Finished loading data. Number of bags: ', len(X), '. Number of labels: ', len(y))
    X_size = 0
    f = open(out_dir + 'bags.csv', 'w')
    f.write('Bag;Shape;Label')
    for i in range(len(X)):
        X_size = X_size + X[i].nbytes
        print('Bag #' + str(i + 1) + ': ', str(X[i].shape), ' -> label: ', str(y[i]))
        f.write('\n' + str(i) + ';' + str(X[i].shape) + ';' + str(y[i]))

        if invert_bag_labels:
            y[i] = int(not y[i])
    f.close()

    X_s = utils.convert_size(X_size)
    y_s = utils.convert_size(getsizeof(y))

    print("X-size in memory: " + str(X_s))
    print("y-size in memory: " + str(y_s))

    # Printing more data
    f = open(out_dir + 'loading_data_statistics.csv', 'w')
    for i in range(len(loaded_files_list)):
        f.write(str(i) + ';' + loaded_files_list[i] + '\n')
    f.write('\nX-size in memory: ' + str(X_s))
    f.write('\ny-size in memory: ' + str(y_s))
    f.write('\ny-Loading time: ' + str(loading_time))
    f.close()
    del X_s, y_s, X_size, f

    if len(X) == 0:
        print('WARNING: NO DATA LOADED')
        return

    # Setting up bags for MIL
    dataset, input_dim = loader.convert_bag_to_batch(X, y)
    print('Detected input dim: ' + str(input_dim))
    del X, y

    # Train-Test Split
    print('Shuffling and splitting data into train and val set')
    training_data, validation_data = shuffle_and_split_data(dataset, train_percentage=0.7)
    del dataset

    training_data_tiles = sum([training_data[i][0].shape[0] for i in range(len(training_data))])
    validation_data_tiles = sum([validation_data[i][0].shape[0] for i in range(len(validation_data))])
    print('Training data: ' + str(training_data_tiles) + ' tiles over ' + str(len(training_data)) + ' bags.')
    print('Validation data: ' + str(validation_data_tiles) + ' tiles over ' + str(len(validation_data)) + ' bags.')

    # Preparing for model loading
    device_ordinals = models.device_ordinals_local
    if _platform == "linux" or _platform == "linux2":
        # looks like I am running on linux
        device_ordinals = models.device_ordinals_ehrlich
        # device_ordinals = models.device_ordinals_local

    # Loading Hardware Device
    device = hardware.get_hardware_device(gpu_preferred=gpu_enabled)
    print('Selected device: ' + str(device))

    # Setting up Model
    print('Setting up model...')
    model = BaselineMIL(input_dim=input_dim, device=device,
                        device_ordinals=device_ordinals,
                        loss_function=loss_function,
                        activation_function='binary')

    # Loader args
    loader_kwargs = {}
    data_loader_cores = math.ceil(os.cpu_count() * 0.5 + 1)
    if torch.cuda.is_available():
        #    #model.cuda()
        loader_kwargs = {'num_workers': data_loader_cores, 'pin_memory': True}

    optimizer = models.apply_optimizer(model)
    print('Finished loading data and model')

    ################
    # DATA START
    ################
    # Data Generators
    train_dl = DataLoader(training_data, batch_size=1, shuffle=True, **loader_kwargs)
    validation_dl = DataLoader(validation_data, batch_size=1, shuffle=False, **loader_kwargs)
    test_dl = None
    # test_dl = DataLoader(test_data, batch_size=1, shuffle=True, **loader_kwargs)

    # Callbacks
    callbacks = []
    # callbacks.append(UnreasonableLossCallback(loss_max=100.0))

    ################
    # TRAINING START
    ################
    print('Start of training for ' + str(epochs) + ' epochs.')
    history, history_keys, model_save_path_best = models.fit(model=model, optimizer=optimizer, epochs=epochs,
                                                             training_data=train_dl,
                                                             validation_data=validation_dl,
                                                             out_dir_base=out_dir,
                                                             callbacks=callbacks)
    print('Finished training!')
    del train_dl

    print('Plotting and saving loss and acc plots...')
    mil_metrics.write_history(history, history_keys, metrics_dir)
    mil_metrics.plot_losses(history, metrics_dir, include_raw=True, include_tikz=True)
    mil_metrics.plot_accuracy(history, metrics_dir, include_raw=True, include_tikz=True)

    ###############
    # TESTING START
    ###############
    # Get best saved model from this run
    model, optimizer, _, _ = models.load_checkpoint(model_save_path_best, model, optimizer)
    y_hats, y_pred, y_true = models.get_predictions(model, validation_dl)
    del validation_dl, test_dl

    # print('Saving Confidence Matrix')
    # mil_metrics.plot_conf_matrix(y_true, y_pred, metrics_dir, target_names=['Non Tox', 'Tox'], normalize=False)

    print('Computing and plotting binary ROC-Curve')
    print('Saving metrics here: ' + metrics_dir)
    fpr, tpr, _ = mil_metrics.binary_roc_curve(y_true, y_hats)
    mil_metrics.plot_binary_roc_curve(fpr, tpr, metrics_dir)


if __name__ == '__main__':
    print("OmniSphero MIL")
    debug: bool = False

    current_epochs = 150
    current_max_workers = 45
    default_out_dir_base = default_out_dir_unix_base
    current_sources_dir = default_source_dirs_unix
    current_gpu_enabled = True
    if debug:
        current_sources_dir = [current_sources_dir[0]]

    if sys.platform == 'win32':
        current_epochs = 15
        current_max_workers = 5
        current_sources_dir = [default_source_dir_win]
        default_out_dir_base = default_out_dir_win_base
        current_gpu_enabled = False

    out_dir_name = 'debug-training-bce'
    current_out_dir = default_out_dir_base + os.sep + out_dir_name + os.sep
    os.makedirs(current_out_dir, exist_ok=True)

    main(source_dirs=current_sources_dir, out_dir=current_out_dir,
         loss_function='binary_cross_entropy',
         epochs=current_epochs,
         max_workers=current_max_workers,
         gpu_enabled=current_gpu_enabled,
         invert_bag_labels=False)
