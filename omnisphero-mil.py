import torch
from torch.utils.data import DataLoader

import loader
import numpy as np
import os

from sys import getsizeof
from sys import platform as _platform

import mil_metrics
import models
from models import BaselineMIL
from models import device_ordinals_local
from util.utils import convert_size
from util.utils import get_hardware_device
from util.utils import shuffle_and_split_data

default_source_dir = "Y:\\bioinfdata\\work\\Omnisphero\\Sciebo\\HCA\\04_HTE_Batches\\BatchParamTest\\false_data\\mil-tiles\\selectedExperiment"
default_source_dir = "Y:\\bioinfdata\\work\\Omnisphero\\Sciebo\\HCA\\04_HTE_Batches\\BatchParamTest\\mil_labels\\exerpt"

# normalize_enum is an enum to determine normalisation as follows:
# 0 = no normalisation
# 1 = normalize every cell between 0 and 255 (8 bit)
# 2 = normalize every cell individually with every color channel independent
# 3 = normalize every cell individually with every color channel using the min / max of all three
# 4 = normalize every cell but with bounds determined by the brightest cell in the same well
normalize_enum_default = 4

max_workers_default = 5


def main(source_dir: str = default_source_dir, epochs: int = 3, max_workers: int = max_workers_default,
         normalize_enum: int = normalize_enum_default, gpu_enabled: bool = False):
    # PREPARING DATA
    out_dir = default_source_dir + os.sep + 'train' + os.sep + 'debug' + os.sep
    metrics_dir = out_dir + os.sep + 'metrics' + os.sep
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    ################
    # LOADING START
    ################
    X, y, errors = loader.load_bags_json_batch(batch_dirs=[source_dir], max_workers=max_workers,
                                               normalize_enum=normalize_enum)
    X = [np.einsum('bhwc->bchw', bag) for bag in X]
    # TODO: Dim should be (xxx, 3, 150, 150)

    print('Finished loading data. Number of bags: ', len(X), '. Number of labels: ', len(y))
    X_size = 0
    for i in range(len(X)):
        X_size = X_size + X[i].nbytes
        print('Bag #' + str(i + 1) + ': ', str(X[i].shape), ' -> label: ', str(y[i]))

    X_s = convert_size(X_size)
    y_s = convert_size(getsizeof(y))

    print("X-size in memory: " + str(X_s))
    print("y-size in memory: " + str(y_s))
    del X_s, y_s, X_size

    # Setting up bags for MIL
    dataset, input_dim = loader.convert_bag_to_batch(X, y)
    print('Detected input dim: ' + str(input_dim))
    del X, y

    # Train-Test Split
    print('Shuffling and splitting data into train and val set')
    training_data, validation_data = shuffle_and_split_data(dataset, train_percentage=0.7)
    del dataset

    # Preparing for model loading
    device_ordinals = models.device_ordinals_local
    if _platform == "linux" or _platform == "linux2":
        # looks like I am running on linux
        device_ordinals = models.device_ordinals_cluster

    # Loading Hardware Device
    device = get_hardware_device(gpu_enabled=gpu_enabled)
    print('Selected device: ' + str(device))
    # TODO an welcher Stelle sollte man das device laden?

    # Setting up Model
    print('Setting up model...')
    model = BaselineMIL(input_dim=input_dim, device=device, device_ordinals=device_ordinals)

    # Loader args
    loader_kwargs = {}
    if torch.cuda.is_available():
        #    #model.cuda()
        loader_kwargs = {'num_workers': max_workers, 'pin_memory': True}

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

    ################
    # TRAINING START
    ################
    print('Start of training for ' + str(epochs) + ' epochs.')
    history, history_keys, model_save_path_best = models.fit(model, optimizer, epochs, train_dl, validation_dl, out_dir)
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

    print('Saving Confidence Matrix')
    mil_metrics.plot_conf_matrix(y_true, y_pred, metrics_dir, target_names=['Non Tox', 'Tox'], normalize=False)

    print('Computing and plotting binary ROC-Curve')
    fpr, tpr, _ = mil_metrics.binary_roc_curve(y_true, y_hats)
    mil_metrics.plot_binary_roc_curve(fpr, tpr, metrics_dir)


if __name__ == '__main__':
    print("OmniSphero MIL")
    main()
