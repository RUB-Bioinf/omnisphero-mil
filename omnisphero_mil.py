import math
import os
import random
import sys
import time
from datetime import datetime
from sys import getsizeof

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

import hardware
import loader
import mil_metrics
import models
import omnisphero_mining
import predict_batch
import r
import torch_callbacks
import video_render_ffmpeg
from util import log
from util import paths
from util import sample_preview
from util import utils
from util.omnisphero_data_loader import OmniSpheroAugmentedDataLoader
from util.omnisphero_data_loader import OmniSpheroDataLoader
from util.paths import default_out_dir_unix_base
from util.paths import training_metrics_live_dir_name
from util.utils import line_print
from util.utils import shuffle_and_split_data

# setting env before importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from torch.optim import Optimizer

# On windows, if there's not enough RAM:
# https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial/issues/10

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


def train_model(
        # Basic training data params
        training_label: str, source_dirs: [str], image_folder: str,
        # Model fitting params
        loss_function: str, device_ordinals: [int],
        epochs: int = 3, max_workers: int = max_workers_default, normalize_enum: int = normalize_enum_default,
        out_dir: str = None, gpu_enabled: bool = False,
        shuffle_data_loaders: bool = True, model_enable_attention: bool = False, model_use_max: bool = True,
        global_log_dir: str = None, optimizer: str = 'adam',
        # Clamp Loss function
        clamp_min: float = None, clamp_max: float = None,
        # Callback configurations
        stop_when_spiking_loss: bool = True,
        early_stopping_enabled: bool = True,
        # Tile shuffling
        loading_preview_rate: float = 0.5,
        repack_percentage: float = 0.0,
        positive_bag_min_samples: int = None,
        # Tile Constraints (How many Nuclei / Oligos / Neurons must be at least in a sample?)
        tile_constraints_0: [int] = loader.default_tile_constraints_none,
        tile_constraints_1: [int] = loader.default_tile_constraints_none,
        # Well indices for labels. When a bag is loaded from a specific well index, the corresponding label is applied
        label_0_well_indices=loader.default_well_indices_none,
        label_1_well_indices=loader.default_well_indices_none,
        force_balanced_batch: bool = False,
        # Enable data augmentation?
        augment_train: bool = False, augment_validation: bool = False,
        # Training histogram bins override (if None, every histogram uses dynamic bin sizes)
        hist_bins_override: int = None,
        # Sigmoid Evaluation parameters
        save_sigmoid_plot_interval: int = 5,
        # Render sigmoid output as a video file after training?
        sigmoid_video_render_enabled: bool = True, render_fps: int = video_render_ffmpeg.default_fps,
        # What channels are enabled during loading?
        channel_inclusions: [bool] = loader.default_channel_inclusions_all,
        # Training / Validation Split percentages
        data_split_percentage_validation: float = 0.35, data_split_percentage_test: float = 0.20,
        # HNM Params
        use_hard_negative_mining: bool = True, hnm_magnitude: float = 5.0, hnm_new_bag_percentage=0.25,
        writing_metrics_enabled: bool = True,
        # Test the model on the test data, after training?
        testing_model_enabled: bool = True,
        # Sigmoid validation dirs
        sigmoid_validation_dirs: [str] = None, reserve_sigmoid_experiments_as_test_data: bool = True,
        # After training, should the model be run on input experiments again?
        predict_training_data_afterwards: bool = False,
        predict_sigmoid_data_afterwards: bool = False
):
    if out_dir is None:
        out_dir = source_dirs[0] + os.sep + 'training_results'
    if not testing_model_enabled:
        data_split_percentage_test = 0
    if sigmoid_validation_dirs is None:
        sigmoid_validation_dirs = []
        reserve_sigmoid_experiments_as_test_data = False
    sigmoid_evaluation_enabled = len(sigmoid_validation_dirs) > 0
    repack_percentage = float(repack_percentage)

    # This param is unused and should not be "True"!
    if type(label_0_well_indices) == list and type(label_1_well_indices) == list:
        assert len(label_0_well_indices) > 0
        assert len(label_1_well_indices) > 0

    data_loader_cores = math.ceil(os.cpu_count() * 0.5 + 1)
    data_loader_cores = int(min(data_loader_cores, 4))

    # Setting up directories
    out_dir = out_dir + os.sep + training_label + os.sep
    loading_preview_dir = out_dir + os.sep + 'loading_previews' + os.sep
    loading_preview_dir_whole_bag = loading_preview_dir + 'whole_bags' + os.sep
    metrics_dir = out_dir + os.sep + 'metrics' + os.sep
    sigmoid_validation_dir = out_dir + os.sep + training_metrics_live_dir_name + os.sep + 'sigmoid_live' + os.sep
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(loading_preview_dir, exist_ok=True)
    os.makedirs(loading_preview_dir_whole_bag, exist_ok=True)
    os.makedirs(sigmoid_validation_dir, exist_ok=True)

    if gpu_enabled:
        log.write('Number of visible devices: ' + str(torch.cuda.device_count()))

    log.write('Model classification - Use Max: ' + str(model_use_max))
    log.write('Model classification - Use Attention: ' + str(model_enable_attention))
    log.write('Model classification - Use HNM: ' + str(use_hard_negative_mining))
    log.write('R - Is pyRserve connection available: ' + str(r.has_connection(also_test_script=True)))

    log.write('Saving logs and protocols to: ' + out_dir)
    # Logging params and args
    protocol_f = open(out_dir + os.sep + 'protocol.txt', 'w')
    protocol_f.write('Start time: ' + utils.gct())
    protocol_f.write('\n\n == General Params ==')
    protocol_f.write('\nSource dirs: ' + str(len(source_dirs)))
    protocol_f.write('\nLoss function: ' + loss_function)
    protocol_f.write('\nDevice ordinals: ' + str(device_ordinals))
    protocol_f.write('\nR - Is pyRserve connection available: ' + str(r.has_connection()))
    protocol_f.write('\nSaving sigmoid plot interval: ' + str(save_sigmoid_plot_interval))
    protocol_f.write('\nEpochs: ' + str(epochs))
    protocol_f.write('\nShuffle data loader: ' + str(shuffle_data_loaders))
    protocol_f.write('\nMax File-Loader Workers: ' + str(max_workers))
    protocol_f.write('\nMax "DataLoader" Workers: ' + str(data_loader_cores))
    protocol_f.write('\nGPU Enabled: ' + str(gpu_enabled))
    protocol_f.write('\nHNM Enabled: ' + str(use_hard_negative_mining))
    protocol_f.write('\nClamp Min: ' + str(clamp_min))
    protocol_f.write('\nClamp Max: ' + str(clamp_max))

    protocol_f.write('\n\n == Loader Params ==')
    protocol_f.write('\nNormalization Enum: ' + str(normalize_enum))
    protocol_f.write('\nNormalization Strategy: ' + loader.normalize_enum_descriptions[normalize_enum])
    protocol_f.write('\nInvert Bag Labels: <deprecated>')
    protocol_f.write('\nRepack: Percentage: ' + str(repack_percentage))
    protocol_f.write('\nLoading Preview Rate: ' + str(loading_preview_rate))
    protocol_f.write('\nRepack: Minimum Positive Samples: ' + str(positive_bag_min_samples))

    protocol_f.write('\n\nWell indices label 0: ' + str(label_0_well_indices))
    protocol_f.write('\nWell indices label 1: ' + str(label_1_well_indices))
    protocol_f.write('\nForce Balanced Batch: ' + str(force_balanced_batch))
    protocol_f.write('\nTile constraints explained: Minimum number of x [Nuclei, Oligos, Neurons]')
    protocol_f.write('\nTile Constraints label 0: ' + str(tile_constraints_0))
    protocol_f.write('\nTile Constraints label 1: ' + str(tile_constraints_1))
    protocol_f.write('\nChannel_inclusions: ' + str(channel_inclusions))

    protocol_f.write('\n\n == Data splitting Params: ==')
    protocol_f.write('\nData Split percentage: Validation: ' + str(data_split_percentage_validation))
    protocol_f.write('\nData Split percentage: Test: ' + str(data_split_percentage_test))
    protocol_f.write('\nNumber of Sigmoid Validation Dirs: ' + str(len(sigmoid_validation_dirs)))
    protocol_f.write('\nSigmoid validation enabled: ' + str(sigmoid_evaluation_enabled))
    protocol_f.write(
        '\nSigmoid validation experiments reserved for test data: ' + str(reserve_sigmoid_experiments_as_test_data))

    global_log_filename = None
    local_log_filename = out_dir + os.sep + 'log.txt'
    log.add_file(local_log_filename)
    if global_log_dir is not None:
        global_log_filename = global_log_dir + os.sep + 'log-' + training_label + '.txt'
        os.makedirs(global_log_dir, exist_ok=True)
        log.add_file(global_log_filename)
    log.diagnose()

    # PREPARING DATA AND DIRECTORIES
    protocol_f.write('\n\n == Directories ==')
    protocol_f.write('\nGlobal Log dir: ' + str(global_log_dir))
    protocol_f.write('\nLocal Log dir: ' + str(local_log_filename))
    protocol_f.write('\nOut dir: ' + str(out_dir))
    protocol_f.write('\nMetrics dir: ' + str(metrics_dir))
    protocol_f.write('\nPreview tiles: ' + str(loading_preview_dir))
    protocol_f.write('\nPredict training input data afterwards: ' + str(predict_training_data_afterwards))
    protocol_f.write('\nPredict sigmoid input data afterwards: ' + str(predict_sigmoid_data_afterwards))

    print('==== List of Source Dirs: =====')
    [print(str(p)) for p in source_dirs]

    protocol_f.write('\n\n == GPU Status ==\n')
    for line in hardware.print_gpu_status(silent=True):
        protocol_f.write(line)
        log.write(line)

    ########################
    # LOADING SIGMOID DATA
    ########################
    f = open(out_dir + 'sigmoid-validation.txt', 'w')
    data_loader_sigmoid: DataLoader = None
    X_metadata_sigmoid: [np.ndarray] = None
    sigmoid_experiment_names = []
    X_sigmoid = []
    if sigmoid_evaluation_enabled:
        f.write('Sigmoid validation dirs:' + str(sigmoid_validation_dirs))

        X_sigmoid, _, _, _, X_metadata_sigmoid, _, _, sigmoid_experiment_names, _, errors_sigmoid, loaded_files_list_sigmoid = loader.load_bags_json_batch(
            batch_dirs=sigmoid_validation_dirs,
            max_workers=max_workers,
            include_raw=True,
            force_balanced_batch=False,
            channel_inclusions=channel_inclusions,
            constraints_0=loader.default_tile_constraints_nuclei,
            constraints_1=loader.default_tile_constraints_nuclei,
            label_0_well_indices=loader.default_well_indices_all,
            label_1_well_indices=loader.default_well_indices_all,
            normalize_enum=normalize_enum)
        X_sigmoid = [np.einsum('bhwc->bchw', bag) for bag in X_sigmoid]

        sigmoid_temp_entry = None
        f.write('\n\nList of loaded sigmoid files:')
        for sigmoid_temp_entry in loaded_files_list_sigmoid:
            f.write('\n' + str(sigmoid_temp_entry))
            # log.write('Loaded sigmoid file: ' + str(sigmoid_temp_entry))
        f.write('\n\nList of sigmoid loading errors:')
        for sigmoid_temp_entry in errors_sigmoid:
            f.write('\n' + str(sigmoid_temp_entry))
            log.write('Loading error: ' + str(sigmoid_temp_entry))

        if not reserve_sigmoid_experiments_as_test_data:
            sigmoid_experiment_names = []

        del errors_sigmoid, loaded_files_list_sigmoid, sigmoid_temp_entry
    else:
        reserve_sigmoid_experiments_as_test_data = False
        sigmoid_experiment_names = []
        f.write('Not sigmoid validating.')
        log.write('Sigmoid validating: Disabled.')
    f.close()

    ##############################
    # LOADING TRAINING DATA START
    ##############################
    unrestricted_experiments_override = None
    if sigmoid_evaluation_enabled and reserve_sigmoid_experiments_as_test_data and testing_model_enabled:
        unrestricted_experiments_override = sigmoid_experiment_names

    # TODO Write well / label mapping to protocol file!
    loading_start_time = datetime.now()
    X, y, y_tiles, X_raw, X_metadata, bag_names, _, _, _, errors, loaded_files_list = loader.load_bags_json_batch(
        batch_dirs=source_dirs,
        max_workers=max_workers,
        include_raw=True,
        force_balanced_batch=force_balanced_batch,
        channel_inclusions=channel_inclusions,
        constraints_0=tile_constraints_0,
        constraints_1=tile_constraints_1,
        label_0_well_indices=label_0_well_indices,
        label_1_well_indices=label_1_well_indices,
        unrestricted_experiments_override=unrestricted_experiments_override,
        normalize_enum=normalize_enum)
    X = [np.einsum('bhwc->bchw', bag) for bag in X]
    X_raw = [np.einsum('bhwc->bchw', bag) for bag in X_raw]
    # Hint: Dim should be (xxx, 3, 150, 150)
    loading_time = utils.get_time_diff(loading_start_time)
    log.write('Loading finished in: ' + str(loading_time))

    # Finished loading. Printing errors and data
    f = open(out_dir + 'loading-errors.txt', 'w')
    for e in errors:
        f.write(str(e))
        f.write('\n')
    f.close()

    # Saving one random image from random bags to the disk
    log.write('Writing loading preview samples to: ' + loading_preview_dir)
    preview_indices_0 = np.where(np.asarray(y) == 0)[0]
    np.random.shuffle(preview_indices_0)
    preview_indices_0 = preview_indices_0[:-max(math.floor(len(preview_indices_0) * (loading_preview_rate * 0.55)), 1)]
    preview_indices_1 = np.where(np.asarray(y) == 1)[0]
    np.random.shuffle(preview_indices_1)
    preview_indices_1 = preview_indices_1[:-max(math.floor(len(preview_indices_1) * (loading_preview_rate * 0.55)), 1)]

    preview_indices = np.append(preview_indices_0, preview_indices_1)
    preview_indices = list(preview_indices)
    preview_indices.sort()

    print('\n')
    for i in range(len(preview_indices)):
        # Not doing that on windows devices
        if os.name == 'nt':
            continue

        line_print('Writing loading preview: ' + str(i + 1) + '/' + str(len(preview_indices)), include_in_log=False)
        current_x: np.ndarray = X[preview_indices[i]]
        j = random.randint(0, current_x.shape[0] - 1)
        current_bag_name = bag_names[preview_indices[i]]
        preview_image_file_base = loading_preview_dir + 'preview_' + str(
            preview_indices[i]) + '-' + current_bag_name + '-' + str(
            j) + '_' + str(y[preview_indices[i]])

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
                                               vmin=-3.0, vmax=3.0, normalize_enum=normalize_enum)

        del current_x
    del preview_indices, preview_indices_0, preview_indices_1, unrestricted_experiments_override
    print('\n')

    # Calculating Bag Size and possibly inverting labels
    X_s = str(utils.byteSizeString(utils.listToBytes(X)))
    X_s_raw = str(utils.byteSizeString(utils.listToBytes(X_raw)))
    y_s = str(utils.byteSizeString(getsizeof(y)))

    log.write('Finished loading data. Number of bags: ' + str(len(X)) + '. Number of labels: ' + str(len(y)))
    log.write("X-size in memory (after loading all data): " + str(X_s))
    log.write("y-size in memory (after loading all data): " + str(y_s))
    log.write("X-size (raw) in memory (after loading all data): " + str(X_s_raw))

    protocol_f.write('\n\n == Loaded Data ==')
    protocol_f.write('\nNumber of Bags: ' + str(len(X)))
    protocol_f.write('\nBags with label 0: ' + str(len(np.where(np.asarray(y) == 0)[0])))
    protocol_f.write('\nBags with label 1: ' + str(len(np.where(np.asarray(y) == 1)[0])))
    protocol_f.write("\nX-size in memory: " + str(X_s))
    protocol_f.write("\ny-size in memory: " + str(y_s))

    # Printing more data
    f = open(out_dir + 'loading-data-statistics.csv', 'w')
    for i in range(len(loaded_files_list)):
        f.write(str(i) + ';' + loaded_files_list[i] + '\n')
    f.write('\n\nX-size in memory: ' + str(X_s))
    f.write('\n\ny-size in memory: ' + str(y_s))
    f.write('\n\nLoading time: ' + str(loading_time))
    f.close()
    del X_s, y_s, X_s_raw, f

    if len(X) == 0:
        log.write('WARNING: NO DATA LOADED')
        protocol_f.write('\n\nWARNING: NO DATA LOADED')
        return

    # Data Augmentation
    # TODO move this somewhere else?
    f = open(out_dir + 'data-augmentation.txt', 'w')
    f.write('## Augmentation Train:\n' + str(augment_train) + '\n\n')
    f.write('## Augmentation Validation:\n' + str(augment_validation) + '\n\n')
    f.close()

    #######################
    # SETTING UP DATASETS
    #######################
    X_sigmoid_overlap = None
    y_sigmoid_overlap = None
    y_tiles_sigmoid_overlap = None
    bag_names_sigmoid_overlap = None
    X_raw_sigmoid_overlap = None

    f = open(out_dir + 'reserve-sigmoid-as-test-data.txt', 'w')
    f.write('## Param sigmoid_evaluation_enabled: ' + str(sigmoid_evaluation_enabled) + '\n')
    f.write(
        '## Param reserve_sigmoid_experiments_as_test_data: ' + str(reserve_sigmoid_experiments_as_test_data) + '\n')
    f.write('## Param testing_model_enabled: ' + str(testing_model_enabled) + '\n\n')
    if sigmoid_evaluation_enabled and reserve_sigmoid_experiments_as_test_data and testing_model_enabled:
        f.write('Number of bags loaded: ' + str(len(X)) + '\n')
        f.write('Names of sigmoid experiments loaded: ' + str(sigmoid_experiment_names) + '\n')
        f.write('Number of sigmoid experiments loaded: ' + str(len(sigmoid_experiment_names)) + '\n')

        X, X_metadata, X_raw, y, y_tiles, bag_names, X_sigmoid_overlap, X_metadata_sigmoid_overlap, X_raw_sigmoid_overlap, y_sigmoid_overlap, y_tiles_sigmoid_overlap, bag_names_sigmoid_overlap = utils.extract_experiments_from_bags(
            X=X, X_raw=X_raw, y=y, y_tiles=y_tiles, bag_names=bag_names, X_metadata=X_metadata,
            experiment_names=sigmoid_experiment_names)

        f.write('\n\n### AFTER SIGMOID REMOVAL ###\n\n')
        f.write('Bags (without sigmoid experiments): ' + str(len(X)) + '\n')
        f.write('Bags (only sigmoid experiments): ' + str(len(X_sigmoid_overlap)) + '\n')
    else:
        f.write('Not running.\n')
    f.close()

    # Printing Bag Shapes
    # Setting up bags for MIL
    if repack_percentage > 0:
        log.write('Repack percent: ' + str(
            repack_percentage) +
                  '.That means, to build a positive bag, x% of positive samples will be added to a negative bag.')
        print_bag_metadata(X, y, y_tiles, bag_names, file_name=out_dir + 'bags-pre-packed.csv')
        X, X_raw, y, y_tiles, bag_names = loader.repack_bags_merge(X=X, X_raw=X_raw, y=y, bag_names=bag_names,
                                                                   repack_percentage=repack_percentage,
                                                                   positive_bag_min_samples=positive_bag_min_samples)
        print_bag_metadata(X, y, y_tiles, bag_names, file_name=out_dir + 'bags-repacked.csv')

        if sigmoid_evaluation_enabled and reserve_sigmoid_experiments_as_test_data:
            print_bag_metadata(X_sigmoid_overlap, y_sigmoid_overlap, y_tiles_sigmoid_overlap, bag_names_sigmoid_overlap,
                               file_name=out_dir + 'bags-pre-packed_sigmoid_overlap.csv')
            X_sigmoid_overlap, X_raw_sigmoid_overlap, y_sigmoid_overlap, y_tiles_sigmoid_overlap, bag_names_sigmoid_overlap = loader.repack_bags_merge(
                X=X_sigmoid_overlap, X_raw=X_raw_sigmoid_overlap, y=y_sigmoid_overlap,
                bag_names=bag_names_sigmoid_overlap,
                repack_percentage=repack_percentage,
                positive_bag_min_samples=positive_bag_min_samples)
            print_bag_metadata(X_sigmoid_overlap, y_sigmoid_overlap, y_tiles_sigmoid_overlap, bag_names_sigmoid_overlap,
                               file_name=out_dir + 'bags-repacked_sigmoid_overlap.csv')
    else:
        print_bag_metadata(X, y, y_tiles, bag_names, file_name=out_dir + 'bags.csv')
        if sigmoid_evaluation_enabled and reserve_sigmoid_experiments_as_test_data and testing_model_enabled:
            print_bag_metadata(X_sigmoid_overlap, y_sigmoid_overlap, y_tiles_sigmoid_overlap, bag_names_sigmoid_overlap,
                               file_name=out_dir + 'bags-sigmoid-overlap-csv')

    # After repacking, X_metadata is invalidated! Must be deleted now.
    del X_metadata
    # Removing unnecessary sigmoid overlap data
    del bag_names_sigmoid_overlap, X_raw_sigmoid_overlap

    #########################
    # WRITING BAG PREVIEWS
    #########################

    # Writing whole bags to the disc
    preview_indexes_positive = list(np.where(np.asarray(y) == 1)[0])
    preview_indexes_negative = list(np.where(np.asarray(y) == 0)[0])
    random.shuffle(preview_indexes_positive)
    random.shuffle(preview_indexes_negative)
    preview_indexes_negative = preview_indexes_negative[0:math.ceil(len(preview_indexes_negative) * 0.15)]
    preview_indexes_positive = preview_indexes_positive[0:math.ceil(len(preview_indexes_negative) * 0.25)]
    preview_indexes = preview_indexes_negative
    preview_indexes.extend(preview_indexes_positive)
    preview_indexes.sort()
    log.write('Number of whole preview bags to save: ' + str(len(preview_indexes)) + '. -> ' + str(preview_indexes))
    print('\n')

    for i in range(len(X)):
        preview_image_filename = loading_preview_dir_whole_bag + 'preview_' + str(i) + '-' + bag_names[i] + '_' + str(
            y[i]) + '_bag.png'
        line_print(
            'Writing whole bag loading preview: ' + str(i + 1) + '/' + str(len(X)) + ' -> ' + preview_image_filename,
            include_in_log=False)

        colored_tiles = []
        image_width = None
        image_height = None
        if i in preview_indexes:
            for rgb in X_raw[i]:
                # Creating a deep copy so it's not overwritten
                rgb = np.copy(rgb)

                image_width, image_height = rgb[0].shape
                rgb = np.einsum('abc->bca', rgb)
                rgb = mil_metrics.outline_rgb_array(rgb, None, None, outline=2, override_colormap=[255, 255, 255])
                colored_tiles.append(rgb)

        if len(colored_tiles) > 0 and image_height is not None:
            out_image = mil_metrics.fuse_image_tiles(images=colored_tiles, image_width=image_width,
                                                     image_height=image_height)
            plt.imsave(preview_image_filename, out_image)
            line_print('Saved: ' + preview_image_filename)

    dataset, input_dim = loader.convert_bag_to_batch(bags=X, labels=y, y_tiles=y_tiles)
    dataset_sigmoid_overlap = None
    log.write('Detected input dim: ' + str(input_dim))
    if sigmoid_evaluation_enabled and reserve_sigmoid_experiments_as_test_data and testing_model_enabled:
        dataset_sigmoid_overlap, _ = loader.convert_bag_to_batch(bags=X_sigmoid_overlap, labels=y_sigmoid_overlap,
                                                                 y_tiles=y_tiles_sigmoid_overlap)

    del X, y, preview_indexes, preview_indexes_positive, preview_indexes_negative
    del X_sigmoid_overlap, y_sigmoid_overlap, y_tiles_sigmoid_overlap

    ##########################
    # RANDOM TRAIN TEST SPLIT
    ##########################
    log.write('Shuffling and splitting data into train and val set')
    test_data = []
    training_data, validation_data = shuffle_and_split_data(dataset,
                                                            additional_dataset=None,
                                                            split_percentage=data_split_percentage_validation)
    if data_split_percentage_test is not None and data_split_percentage_test > 0:
        training_data, test_data = shuffle_and_split_data(dataset=training_data,
                                                          additional_dataset=dataset_sigmoid_overlap,
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
        protocol_f.write('\nTest data: ' + str(test_data_tiles) + ' tiles over ' + str(len(test_data)) + ' bags.')
    f.close()

    # Loading Hardware Device
    device = hardware.get_hardware_device(gpu_preferred=gpu_enabled)
    log.write('Selected device: ' + str(device))

    # Loader args
    loader_kwargs = {}
    data_loader_pin_memory = False
    if torch.cuda.is_available():
        # model.cuda()
        loader_kwargs = {'num_workers': data_loader_cores, 'pin_memory': data_loader_pin_memory}

    #############################
    # SETTING UP DATA LOADERS
    #############################
    # Data Generators
    test_dl = None
    if augment_train:
        train_dl = OmniSpheroAugmentedDataLoader(training_data, batch_size=1, shuffle=shuffle_data_loaders,
                                                 transform_enabled=augment_train,
                                                 transform_data_saver=False, **loader_kwargs)
    else:
        train_dl = DataLoader(training_data, batch_size=1, shuffle=shuffle_data_loaders, **loader_kwargs)

    if augment_validation:
        validation_dl = OmniSpheroAugmentedDataLoader(validation_data, batch_size=1,
                                                      transform_enabled=augment_validation,
                                                      transform_data_saver=False,
                                                      shuffle=shuffle_data_loaders, **loader_kwargs)
    else:
        validation_dl = DataLoader(validation_data, batch_size=1, shuffle=shuffle_data_loaders, **loader_kwargs)

    if data_split_percentage_test is not None:
        test_dl = DataLoader(test_data, batch_size=1, shuffle=shuffle_data_loaders, **loader_kwargs)
    del validation_data, test_data

    if sigmoid_evaluation_enabled:
        dataset_sigmoid, _ = loader.convert_bag_to_batch(bags=X_sigmoid, labels=None, y_tiles=None)
        data_loader_sigmoid = DataLoader(dataset_sigmoid, batch_size=1, shuffle=False, **loader_kwargs)
        del dataset_sigmoid, X_sigmoid

    ################
    # MODEL START
    ################
    # Setting up Model
    log.write('Setting up model.')
    accuracy_function = 'binary'
    model = models.BaselineMIL(input_dim=input_dim, device=device,
                               use_max=model_use_max,
                               enable_attention=model_enable_attention,
                               device_ordinals=device_ordinals,
                               loss_function=loss_function,
                               accuracy_function=accuracy_function)

    # Saving the raw version of this model
    untrained_model_path = out_dir + os.sep + 'model.h5'
    torch.save(model.state_dict(), out_dir + 'model.pt')
    torch.save(model, untrained_model_path)
    log.write('Saving trained model to: ' + untrained_model_path)

    model_optimizer = models.choose_optimizer(model, selection=optimizer)
    log.write('Finished loading data and model')
    log.write('Optimizer: ' + str(model_optimizer))

    # Callbacks
    callbacks = []
    hnm_callbacks = []
    if stop_when_spiking_loss:
        callbacks.append(torch_callbacks.SpikingLossCallback(loss_max=40.0))
        hnm_callbacks.append(torch_callbacks.SpikingLossCallback(loss_max=40.0))
    if early_stopping_enabled:
        hnm_callbacks.append(torch_callbacks.EarlyStopping(epoch_threshold=int(epochs / 5 + 1)))
        callbacks.append(torch_callbacks.EarlyStopping(epoch_threshold=int(epochs / 5 + 1)))

    protocol_f.write('\n\n == Model Information==')
    protocol_f.write('\nDevice Ordinals: ' + str(device_ordinals))
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
    log.write(
        'Start of training for ' + str(epochs) + ' epochs. Devices: ' + str(device_ordinals) + '. GPU enabled: ' + str(
            gpu_enabled))
    log.write('Training: "' + training_label + '"!')
    history, history_keys, model_save_path_best = models.fit(model=model,
                                                             optimizer=model_optimizer,
                                                             epochs=epochs,
                                                             training_data=train_dl,
                                                             validation_data=validation_dl,
                                                             out_dir_base=out_dir,
                                                             bag_names=bag_names,
                                                             checkpoint_interval=None,
                                                             sigmoid_video_render_enabled=sigmoid_video_render_enabled,
                                                             render_fps=render_fps,
                                                             hist_bins_override=hist_bins_override,
                                                             sigmoid_evaluation_enabled=sigmoid_evaluation_enabled,
                                                             save_sigmoid_plot_interval=save_sigmoid_plot_interval,
                                                             data_loader_sigmoid=data_loader_sigmoid,
                                                             X_metadata_sigmoid=X_metadata_sigmoid,
                                                             clamp_min=clamp_min,
                                                             clamp_max=clamp_max,
                                                             callbacks=callbacks)
    log.write('Finished training!')

    if not use_hard_negative_mining:
        # Not mining, so we can delete the training data early to save on resources
        del train_dl, training_data
        training_data = None
        train_dl = None

    # Checking how many epochs have actually passed. If more than 100, the fitted line will be printed!
    epochs_passed = len(history)
    include_line_fit = False
    if epochs_passed > 100:
        include_line_fit = True

    if writing_metrics_enabled:
        log.write('Plotting and saving loss and acc plots...')
        mil_metrics.write_history(history, history_keys, metrics_dir)
        mil_metrics.plot_losses(history, metrics_dir, include_raw=True, include_tikz=True, clamp=2.0,
                                include_line_fit=include_line_fit)
        mil_metrics.plot_accuracy(history, metrics_dir, include_raw=True, include_tikz=True,
                                  include_line_fit=include_line_fit)
        mil_metrics.plot_accuracy_tiles(history, metrics_dir, include_raw=True, include_tikz=True,
                                        include_line_fit=include_line_fit)
        mil_metrics.plot_accuracies(history, metrics_dir, include_tikz=True, include_line_fit=include_line_fit)
        mil_metrics.plot_dice_scores(history, metrics_dir, include_tikz=True, include_line_fit=include_line_fit)
        mil_metrics.plot_sigmoid_scores(history, metrics_dir, include_tikz=True, include_line_fit=include_line_fit)
        mil_metrics.plot_binary_roc_curves(history, metrics_dir, include_tikz=True)

        if model.enable_attention:
            mil_metrics.plot_attention_otsu_threshold(history, metrics_dir, label=1, include_tikz=True)
            mil_metrics.plot_attention_entropy(history, metrics_dir, label=1, include_tikz=True)
            mil_metrics.plot_attention_otsu_threshold(history, metrics_dir, label=0, include_tikz=True)
            mil_metrics.plot_attention_entropy(history, metrics_dir, label=0, include_tikz=True)
    del include_line_fit

    ##################
    # TESTING START
    ##################
    if testing_model_enabled:
        log.write('Testing best model on validation and test data to determine performance')
        test_dir = out_dir + 'metrics' + os.sep + 'performance-validation-data' + os.sep
        test_model(model, model_save_path_best, model_optimizer, data_loader=validation_dl, out_dir=test_dir,
                   bag_names=bag_names, X_raw=X_raw, y_tiles=y_tiles)
        if data_split_percentage_test > 0:
            test_dir = out_dir + 'metrics' + os.sep + 'performance-test-data' + os.sep
            test_model(model, model_save_path_best, model_optimizer, data_loader=test_dl, out_dir=test_dir, X_raw=X_raw,
                       bag_names=bag_names, y_tiles=y_tiles)

    ########################
    # HARD NEGATIVE MINING
    ########################
    if use_hard_negative_mining:
        log.write('[0/4] Hard Negative Mining: Pre-Processing')
        # Hard Negative Mining
        # train_dl.transform_enabled = False

        log.write('[1/4] Hard Negative Mining: Finding false positives')
        false_positive_bags, attention_weights_list, false_positive_bags_raw = omnisphero_mining.get_false_positive_bags(
            trained_model=model,
            train_dl=train_dl,
            X_raw=X_raw)

        log.write('[2/4] Hard Negative Mining: Finding hard negatives')
        hard_negative_instances, hard_negative_instances_raw = omnisphero_mining.determine_hard_negative_instances(
            false_positive_bags=false_positive_bags, attention_weights=attention_weights_list,
            false_positive_bags_raw=false_positive_bags_raw,
            magnitude=hnm_magnitude)
        if not len(hard_negative_instances):
            log.write('[?/?] Hard Negative Mining: No hard negative instances found!')
            return

        log.write('[3/4] Hard Negative Mining: Creating new bags')
        n_clusters = math.ceil(len(training_data) * hnm_new_bag_percentage + 1)
        new_bags, new_bags_raw, new_bag_names = omnisphero_mining.new_bag_generation(hard_negative_instances,
                                                                                     training_data,
                                                                                     hard_negative_instances_raw=hard_negative_instances_raw,
                                                                                     n_clusters=n_clusters)

        log.write('[4/4] Hard Negative Mining: Adding new bags to the dataset')
        training_data, X_raw, bag_names = omnisphero_mining.add_back_to_dataset(training_ds=training_data,
                                                                                new_bags=new_bags,
                                                                                new_bag_names=new_bag_names,
                                                                                X_raw=X_raw, new_bags_raw=new_bags_raw,
                                                                                bag_names=bag_names)

        # Fitting a new model with the mined bags
        if augment_train:
            train_dl = OmniSpheroAugmentedDataLoader(training_data, batch_size=1, shuffle=shuffle_data_loaders,
                                                     transform_enabled=augment_train, transform_data_saver=False,
                                                     **loader_kwargs)
        else:
            train_dl = DataLoader(training_data, batch_size=1, shuffle=shuffle_data_loaders, **loader_kwargs)

        mined_out_dir = out_dir + os.sep + 'hnm' + os.sep
        os.makedirs(mined_out_dir, exist_ok=True)
        epochs = math.ceil(epochs * 1.5)

        f = open(mined_out_dir + 'mining.txt', 'w')
        f.write('Hard Negative Mining parameters:')
        f.write('\nMining dir: ' + mined_out_dir)
        f.write('\nEpochs: ' + str(epochs))
        f.write('\nTraining data training bag mult: ' + str(hnm_new_bag_percentage))
        f.write('\nTraining data new bag count: ' + str(n_clusters))
        f.close()
        del f

        # Saving new bags to disk
        sample_preview.save_hnm_bags(mined_out_dir + 'bags' + os.sep, new_bags, new_bags_raw, new_bag_names)

        print('Fitting a new model using HNM bags!')
        history, history_keys, model_save_path_best = models.fit(model=model,
                                                                 optimizer=model_optimizer,
                                                                 epochs=epochs,
                                                                 training_data=train_dl,
                                                                 validation_data=validation_dl,
                                                                 out_dir_base=mined_out_dir,
                                                                 data_loader_sigmoid=data_loader_sigmoid,
                                                                 X_metadata_sigmoid=X_metadata_sigmoid,
                                                                 sigmoid_evaluation_enabled=sigmoid_evaluation_enabled,
                                                                 save_sigmoid_plot_interval=save_sigmoid_plot_interval,
                                                                 checkpoint_interval=None,
                                                                 sigmoid_video_render_enabled=sigmoid_video_render_enabled,
                                                                 render_fps=render_fps,
                                                                 clamp_min=clamp_min,
                                                                 clamp_max=clamp_max,
                                                                 bag_names=None,
                                                                 # TODO add bag names
                                                                 callbacks=hnm_callbacks)

        # Plotting HNM metrics
        log.write('Plotting HNM and saving loss and acc plots...')
        metrics_dir = mined_out_dir + os.sep + 'metrics' + os.sep
        os.makedirs(metrics_dir, exist_ok=True)

        epochs_passed = len(history)
        include_line_fit = False
        if epochs_passed > 100:
            include_line_fit = True

        mil_metrics.write_history(history, history_keys, metrics_dir)
        mil_metrics.plot_losses(history, metrics_dir, include_raw=True, include_tikz=True, clamp=2.0,
                                include_line_fit=include_line_fit)
        mil_metrics.plot_accuracy(history, metrics_dir, include_raw=True, include_tikz=True,
                                  include_line_fit=include_line_fit)
        mil_metrics.plot_accuracy_tiles(history, metrics_dir, include_raw=True, include_tikz=True,
                                        include_line_fit=include_line_fit)
        mil_metrics.plot_accuracies(history, metrics_dir, include_tikz=True, include_line_fit=include_line_fit)
        mil_metrics.plot_dice_scores(history, metrics_dir, include_tikz=True, include_line_fit=include_line_fit)
        mil_metrics.plot_sigmoid_scores(history, metrics_dir, include_tikz=True, include_line_fit=include_line_fit)
        mil_metrics.plot_binary_roc_curves(history, metrics_dir, include_tikz=True)

        if model.enable_attention:
            mil_metrics.plot_attention_otsu_threshold(history, metrics_dir, label=1, include_tikz=True)
            mil_metrics.plot_attention_entropy(history, metrics_dir, label=1, include_tikz=True)
            mil_metrics.plot_attention_otsu_threshold(history, metrics_dir, label=0, include_tikz=True)
            mil_metrics.plot_attention_entropy(history, metrics_dir, label=0, include_tikz=True)
        del include_line_fit

        # Testing HNM models on test data
        log.write('Testing HNM best model on validation and test data to determine performance')
        test_dir = mined_out_dir + 'metrics' + os.sep + 'performance-validation-data' + os.sep
        test_model(model, model_save_path_best, model_optimizer, data_loader=validation_dl, out_dir=test_dir,
                   bag_names=bag_names, X_raw=X_raw, y_tiles=y_tiles)
        if data_split_percentage_test > 0:
            test_dir = mined_out_dir + 'metrics' + os.sep + 'performance-test-data' + os.sep
            test_model(model, model_save_path_best, model_optimizer, data_loader=test_dl, out_dir=test_dir, X_raw=X_raw,
                       bag_names=bag_names, y_tiles=y_tiles)
    # Finished HARD NEGATIVE MINING at this point

    ##################################
    # FINISHED TRAINING - Cleaning up
    ##################################
    log.write("Finished training and testing for this run. Job's done!")
    del train_dl, training_data
    del validation_dl, test_dl, X_raw, y_tiles

    ############################
    # PREDICTING THE TEST DATA
    ############################
    prediction_paths: [[str]] = []
    prediction_labels: [str] = []
    if predict_training_data_afterwards:
        prediction_paths.append(source_dirs)
        prediction_labels.append('training_experiments')
    if predict_sigmoid_data_afterwards and sigmoid_evaluation_enabled and len(sigmoid_validation_dirs) > 0:
        prediction_paths.append(sigmoid_validation_dirs)
        prediction_labels.append('training_experiments')

    for (prediction_path, prediction_label) in zip(prediction_paths, prediction_labels):
        log.write('\n\n==== PREDICTING DATA TO TEST THE NEW MODEL ====', include_timestamp=False)
        log.write('\n ## ' + training_label + ' => ' + prediction_label + ' ##\n\n', include_timestamp=False)

        prediction_out_path = out_dir + os.sep + 'post_predictions_' + prediction_label + os.sep
        os.makedirs(prediction_out_path, exist_ok=True)

        # Calculating every path one at a time to save on system resources
        for path in prediction_path:
            predict_batch.predict_path(
                # (Re-) using the best model that was just calculated:
                model_save_path=untrained_model_path,
                checkpoint_file=model_save_path_best,
                bag_paths=[path],
                out_dir=prediction_out_path,
                gpu_enabled=gpu_enabled,
                channel_inclusions=channel_inclusions,

                # (Re-) using the same normalization settings and other directories. Now used for predictions.
                normalize_enum=normalize_enum,
                max_workers=max_workers,
                image_folder=image_folder,
                global_log_dir=global_log_dir,

                # Setting the tile constraints to the default prediction settings
                tile_constraints=loader.default_tile_constraints_nuclei,

                # deciding what to render:
                render_merged_predicted_tiles_activation_overlays=False,
                render_attention_histogram_enabled=True,
                render_attention_cell_distributions=False,
                render_dose_response_curves_enabled=True,

                # misc settings
                sigmoid_verbose=False,
                clear_global_logs=False
            )

        del prediction_path, prediction_label

    ##################################
    # CLEAN UP EVERYTHING ELSE
    ##################################
    log.remove_file(local_log_filename)
    if global_log_dir is not None:
        log.remove_file(global_log_filename)
    log.clear_files()
    # Run finished
    # Noting more to do beyond this point


def test_model(model: models.OmniSpheroMil, model_save_path_best: str, model_optimizer: Optimizer,
               data_loader: OmniSpheroDataLoader, X_raw: [np.ndarray], y_tiles: [int],
               bag_names: [str], out_dir: str):
    attention_out_dir = out_dir + 'attention' + os.sep
    sigmoid_out_dir = out_dir + 'sigmoid' + os.sep
    os.makedirs(attention_out_dir, exist_ok=True)
    os.makedirs(sigmoid_out_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Get best saved model from this run
    model, model_optimizer, _, _ = models.load_checkpoint(model_save_path_best, model, model_optimizer)
    y_hats, y_pred, y_true, _, y_samples_pred, y_samples_true, all_attentions, _ = models.get_predictions(model,
                                                                                                          data_loader)

    # Flattening sample predictions
    y_samples_pred = [item for sublist in y_samples_pred for item in sublist]
    y_samples_true = [item for sublist in y_samples_true for item in sublist]

    # Saving attention scores for every tile in every bag!
    log.write('Writing attention scores to: ' + attention_out_dir)
    try:
        mil_metrics.save_tile_attention(out_dir=attention_out_dir, model=model, normalized=True, bag_names=bag_names,
                                        dataset=data_loader, X_raw=X_raw, y_tiles=y_tiles)
        # mil_metrics.save_tile_attention(out_dir=attention_out_dir, model=model, normalized=False, bag_names=bag_names,
        #                                 dataset=data_loader, X_raw=X_raw, y_tiles=y_tiles)
    except Exception as e:
        log.write('Failed to save the attention score for every tile!')
        f = open(attention_out_dir + 'attention-error.txt', 'a')
        f.write('\nError: ' + str(e.__class__) + '\n' + str(e))
        f.close()
    log.write('Finished writing attention scores.')

    # Confidence Matrix
    try:
        log.write('Saving Confidence Matrix')
        mil_metrics.plot_conf_matrix(y_true, y_pred, out_dir, target_names=['Class 0', 'Class 1'],
                                     normalize=False)
        mil_metrics.plot_conf_matrix(y_true, y_pred, out_dir, target_names=['Class 0', 'class 1'], normalize=True)
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


def print_bag_metadata(X, y, y_tiles, bag_names, file_name: str):
    f = open(file_name, 'w')
    f.write(';Bag;Label;Tile Labels (Sum);Name;Memory Raw;Memory Formatted;Shape;Shape;Shape;Shape')

    y0 = len(np.where(np.asarray(y) == 0)[0])
    y1 = len(np.where(np.asarray(y) == 1)[0])

    x_size = 0
    bag_count = 0
    for i in range(len(X)):
        current_X = X[i]
        x_size = x_size + current_X.nbytes
        x_size_converted = utils.byteSizeString(current_X.nbytes)
        bag_name = bag_names[i]

        shapes = ';'.join([str(s) for s in current_X.shape])
        bag_count = bag_count + current_X.shape[0]

        # print('Bag #' + str(i + 1) + ': ', shapes, ' -> label: ', str(y[i]))
        f.write('\n;' + str(i) + ';' + str(y[i]) + ';' + str(sum(y_tiles[i])) + ';' + bag_name + ';' + str(
            current_X.nbytes) + ';' + x_size_converted + ';' + shapes)

    f.write('\n' + 'Sum;' + str(len(X)) + ';0: ' + str(y0) + ';' + str(x_size) + ';' + utils.byteSizeString(
        x_size) + ';' + str(bag_count))
    f.write('\n;;1: ' + str(y1))
    f.close()


def main(debug: bool = False):
    if sys.platform == 'win32':
        debug = True
    print('Debug mode: ' + str(debug))
    log.write('Python: ' + str(sys.version_info))

    current_epochs = 450
    current_max_workers = 35
    default_out_dir_base = default_out_dir_unix_base
    current_sources_dir = paths.curated_overlapping_source_dirs_unix
    current_gpu_enabled = True
    if debug:
        current_sources_dir = [current_sources_dir[0]]
        current_epochs = 5

    # Preparing for model loading
    current_device_ordinals = models.device_ordinals_ehrlich
    image_folder: str = None
    sigmoid_input_dirs: [str] = []

    if sys.platform == 'win32':
        image_folder = paths.nucleus_predictions_image_folder_win
        sigmoid_input_dirs = paths.default_sigmoid_validation_dirs_win

        current_global_log_dir = 'U:\\bioinfdata\\work\\OmniSphero\\Sciebo\\HCA\\00_Logs\\mil_log\\win\\'
        log.add_file('U:\\bioinfdata\\work\\OmniSphero\\Sciebo\\HCA\\00_Logs\\mil_log\\win\\all_logs.txt')

        current_max_workers = 6
        current_sources_dir = paths.curated_overlapping_source_dirs_win
        default_out_dir_base = paths.default_out_dir_win_base
        current_gpu_enabled = False
        current_device_ordinals = models.device_ordinals_local
    else:
        sigmoid_input_dirs = paths.default_sigmoid_validation_dirs_unix
        image_folder = paths.nucleus_predictions_image_folder_unix
        current_global_log_dir = '/Sciebo/HCA/00_Logs/mil_log/linux/'

    current_out_dir = default_out_dir_base + os.sep
    os.makedirs(current_out_dir, exist_ok=True)

    # Checking if all specified paths actually exist
    # assert os.path.exists(image_folder)
    # assert np.all([os.path.exists(x) for x in sigmoid_input_dirs])
    # assert np.all([os.path.exists(x) for x in current_sources_dir])
    assert check_exists([image_folder])
    assert check_exists(sigmoid_input_dirs)
    assert check_exists(current_sources_dir)

    log.write('Starting Training...')
    if debug and sys.platform == 'win32':
        training_label = 'debug-sigmoid'
        train_model(source_dirs=current_sources_dir, out_dir=current_out_dir, epochs=current_epochs,
                    max_workers=current_max_workers, gpu_enabled=current_gpu_enabled, image_folder=image_folder,
                    device_ordinals=current_device_ordinals,
                    normalize_enum=6,
                    training_label=training_label,
                    global_log_dir=current_global_log_dir,
                    save_sigmoid_plot_interval=1,
                    repack_percentage=0.0,
                    model_use_max=False,
                    model_enable_attention=True,
                    positive_bag_min_samples=0,
                    augment_validation=False,
                    augment_train=False,
                    predict_sigmoid_data_afterwards=True,
                    predict_training_data_afterwards=False,
                    stop_when_spiking_loss=False,
                    early_stopping_enabled=False,
                    tile_constraints_0=loader.default_tile_constraints_nuclei,
                    tile_constraints_1=loader.default_tile_constraints_oligos,
                    label_1_well_indices=loader.default_well_indices_debug_early,
                    label_0_well_indices=loader.default_well_indices_debug_late,
                    loss_function='mean_square_error',
                    testing_model_enabled=True,
                    writing_metrics_enabled=True,
                    use_hard_negative_mining=False,
                    reserve_sigmoid_experiments_as_test_data=True,
                    # sigmoid_validation_dirs=None
                    sigmoid_validation_dirs=paths.default_sigmoid_validation_dirs_win
                    )
    elif debug:
        log.write("Testing all source dirs, if they are trainable!")
        time.sleep(2)

        for i in range(len(paths.curated_overlapping_source_dirs_unix)):
            source_dir = paths.curated_overlapping_source_dirs_unix[i]
            log.write('Source ' + str(i) + ': ' + str(source_dir) + '. Exists: ' + str(os.path.exists(source_dir)))
            time.sleep(0.1)
            del i
        time.sleep(1)

        for c in range(len(paths.curated_overlapping_source_dirs_unix)):
            source_dir = paths.curated_overlapping_source_dirs_unix[c]

            copy_dirs = paths.curated_overlapping_source_dirs_unix.copy()
            random.shuffle(copy_dirs)
            # source_dirs = [source_dir, copy_dirs[0], copy_dirs[1]]
            source_dirs = [source_dir]
            log.write('DEBUGGING SOURCE DIRS:\n' + str(source_dirs))

            train_model(source_dirs=source_dirs,
                        device_ordinals=current_device_ordinals,
                        training_label='debug-sorce-test-' + str(c),
                        image_folder=image_folder,
                        normalize_enum=6,
                        use_hard_negative_mining=False,
                        model_enable_attention=True,
                        model_use_max=False,
                        augment_validation=False,
                        augment_train=False,
                        predict_sigmoid_data_afterwards=True,
                        predict_training_data_afterwards=False,
                        max_workers=27,
                        optimizer='adadelta',
                        loss_function='binary_cross_entropy',
                        channel_inclusions=loader.default_channel_inclusions_no_neurites,
                        tile_constraints_0=loader.default_tile_constraints_nuclei,
                        tile_constraints_1=loader.default_tile_constraints_nuclei,
                        repack_percentage=0,
                        label_1_well_indices=loader.default_well_bmc_threshold_control,
                        label_0_well_indices=loader.default_well_bmc_threshold_effect,
                        sigmoid_validation_dirs=paths.default_sigmoid_validation_dirs_win,
                        gpu_enabled=True,
                        epochs=5
                        )
    else:
        # '/mil/oligo-diff/models/linux/hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65/'
        c = 0
        for l in ['mean_square_error', 'binary_cross_entropy']:
            # best: binary_cross_entropy
            for o in ['adadelta']:  # ['adam', 'adadelta']:
                # best: adadelta
                for p in [0.0]:  # [0.3, 0.35, 0.6]:  # [0.10, 0.20, 0.3, 0.05, 0.15, 0.25, 0.3, 0.35]:
                    # best: 0.65 or 0.3
                    for i in [6, 7, 8, 4]:
                        for aug in [[True, True]]:  # , [True, False], [False, True]]:
                            augment_validation = aug[0]
                            augment_train = aug[1]
                            training_label = 'ep-overlap-' + o + '-n-' + str(i) + '-rp-' + str(
                                p) + '-l-' + l + '-BMC-wholeSphere-reserved'

                            log.write('Training label: ' + training_label)
                            if os.path.exists(current_out_dir + os.sep + training_label):
                                log.write('MODEL PATH ALREADY EXISTS!')
                                time.sleep(1)
                                print('\n')
                                log.write('SKIPPING!')
                                print('\n')
                                time.sleep(3)
                                continue

                            print('\n\n############################################################\n\n')
                            train_model(source_dirs=current_sources_dir, out_dir=current_out_dir, epochs=current_epochs,
                                        max_workers=current_max_workers, gpu_enabled=current_gpu_enabled,
                                        image_folder=image_folder,
                                        force_balanced_batch=True,
                                        normalize_enum=i,
                                        training_label=training_label,
                                        global_log_dir=current_global_log_dir,
                                        predict_sigmoid_data_afterwards=True,
                                        predict_training_data_afterwards=False,
                                        data_split_percentage_validation=0.25,
                                        data_split_percentage_test=0.15,
                                        use_hard_negative_mining=False,
                                        hnm_magnitude=5.5,
                                        hnm_new_bag_percentage=0.35,
                                        loss_function=l,
                                        repack_percentage=p,
                                        optimizer=o,
                                        channel_inclusions=loader.default_channel_inclusions_no_neurites,
                                        augment_validation=augment_validation,
                                        augment_train=augment_train,
                                        model_use_max=False,
                                        model_enable_attention=True,
                                        positive_bag_min_samples=4,
                                        reserve_sigmoid_experiments_as_test_data=True,
                                        tile_constraints_0=loader.default_tile_constraints_nuclei,
                                        tile_constraints_1=loader.default_tile_constraints_oligos,
                                        label_1_well_indices=loader.default_well_bmc_threshold_control,
                                        label_0_well_indices=loader.default_well_bmc_threshold_effect,
                                        device_ordinals=current_device_ordinals,
                                        sigmoid_validation_dirs=sigmoid_input_dirs
                                        )
    log.write('Finished every training!')


def check_exists(paths: []):
    ret = True
    for path in paths:
        if not os.path.exists(path):
            log.write('PATH DOES NOT EXIST: ' + path)
            ret = False
    return ret


if __name__ == '__main__':
    print("Training OmniSphero MIL")
    debug: bool = False

    hardware.print_gpu_status()

    main(debug=debug)
