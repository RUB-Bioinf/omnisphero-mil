# IMPORTS

import json
import math
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from sys import platform
from typing import Union
from zipfile import ZipFile

import matplotlib.pyplot as plt
import mil_metrics
import numpy as np
import r
from util import log
from util import paths
from util import utils
from util.sample_preview import z_score_to_rgb
from util.utils import gct
from util.utils import get_time_diff
from util.utils import line_print
from util.well_metadata import PlateMetadata
from util.well_metadata import TileMetadata
####
# Constants
from util.well_metadata import extract_well_info

# normalize_enum is an enum to determine normalisation as follows:
#  0 = no normalisation
#  1 = normalize every cell between 0 and 255 (8 bit)
#  2 = normalize every cell individually with every color channel independent
#  3 = normalize every cell individually with every color channel using the min / max of all three
#  4 = normalize every cell but with bounds determined by the brightest cell in the bag
#  5 = z-score every cell individually with every color channel independent
#  6 = z-score every cell individually with every color channel using the mean / std of all three
#  7 = z-score every cell individually with every color channel independent using all samples in the bag
#  8 = z-score every cell individually with every color channel using the mean / std of all three from all samples in the bag
#  9: Normalizing first, according to [4] and z-scoring afterwards according to [5]
# 10: Normalizing first, according to [4] and z-scoring afterwards according to [6]

normalize_enum_default = 1

normalize_enum_descriptions = [
    ' 0: No Normalisation',
    ' 1: Normalize samples between 0 and 255 (8 bit)',
    ' 2: Normalize samples individually with every color channel independent',
    ' 3: Normalize samples individually with every color channel using the min / max of all three',
    ' 4: Normalize samples but with bounds determined by the brightest cell in the bag',
    ' 5: z-score samples with every color channel independent',
    ' 6: z-score samples with combined mean / std of all three',
    ' 7: z-score samples with mean / std of all three over the bag',
    ' 8: z-score samples with combined mean / std over the bag',
    ' 9: Normalizing first, according to [4] and z-scoring afterwards according to [5]',
    '10: Normalizing first, according to [4] and z-scoring afterwards according to [6]'
]

default_tile_constraints_none = [0, 0, 0]
default_tile_constraints_nuclei = [1, 0, 0]
default_tile_constraints_oligos = [0, 1, 0]
default_tile_constraints_neurons = [0, 0, 1]

default_well_indices_none = []
default_well_indices_all = list(range(99))
default_well_indices_early = [0, 1, 2, 3, 4]
default_well_indices_late = [7, 8, 9]

default_well_indices_very_early = [0, 1, 2, 3]
default_well_indices_very_late = [8, 9]

default_well_bmc_threshold_control = 0.7
default_well_bmc_threshold_effect = 0.3

default_channel_inclusions_all = [True, True, True]
default_channel_inclusions_no_neurites = [True, True, False]

####

# Threading lock
global thread_lock
thread_lock = threading.Lock()

global plate_metadata_cache
plate_metadata_cache = {}


def load_bags_json_batch(batch_dirs: [str], max_workers: int, normalize_enum: int, include_raw: bool = True,
                         channel_inclusions=None, constraints_0=None, constraints_1=None, label_0_well_indices=None,
                         label_1_well_indices=None):
    if label_1_well_indices is None:
        label_1_well_indices = default_well_indices_none
    if label_0_well_indices is None:
        label_0_well_indices = default_well_indices_none
    if constraints_1 is None:
        constraints_1 = default_tile_constraints_none
    if constraints_0 is None:
        constraints_0 = default_tile_constraints_none
    if channel_inclusions is None:
        channel_inclusions = default_channel_inclusions_all

    log.write('Looking for multiple source dirs to load json data from.')
    log.write('Batch dir: ' + str(batch_dirs))
    log.write('Normalization Protocol: ' + str(normalize_enum))

    log.write('Well indices label 0: ' + str(label_0_well_indices))
    log.write('Well indices label 1: ' + str(label_1_well_indices))

    log.write('Tile constraints explained: Minimum number of x [Nuclei, Oligos, Neurons]')
    log.write('Tile Constraints label 0: ' + str(constraints_0))
    log.write('Tile Constraints label 1: ' + str(constraints_1))
    log.write('Channel Inclusions: ' + str(channel_inclusions))

    X_full = None
    X_raw_full = None
    y_full = None
    y_tiles_full = None
    X_metadata_full = []
    error_list = []
    loaded_files_list_full = []
    bag_names_full = []
    experiment_names_full = []
    well_names_full = []

    # Checking if all the paths exist
    # we do this first, as to avoid unnessesary loading just to see that a path does not exist later on
    all_paths_exist = True
    for i in range(len(batch_dirs)):
        current_dir = batch_dirs[i]

        if not os.path.exists(current_dir):
            log.write('Error! Path to load does not exist: ' + current_dir)
            all_paths_exist = False
    assert all_paths_exist

    # Now that we know that these paths do all exist, we can load them
    for i in range(len(batch_dirs)):
        current_dir = batch_dirs[i]
        log.write('Considering source directory: ' + current_dir)

        if os.path.isdir(current_dir):
            X, y, y_tiles, X_raw, X_metadata, bag_names, experiment_names, well_names, errors, loaded_files_list = load_bags_json(
                source_dir=current_dir,
                max_workers=max_workers,
                normalize_enum=normalize_enum,
                gp_current=i + 1,
                channel_inclusions=channel_inclusions,
                label_0_well_indices=label_0_well_indices,
                label_1_well_indices=label_1_well_indices,
                constraints_0=constraints_0,
                constraints_1=constraints_1,
                gp_max=len(batch_dirs),
                include_raw=include_raw)

            # Adding results to the respective lists
            loaded_files_list_full.extend(loaded_files_list)
            bag_names_full.extend(bag_names)
            error_list.extend(errors)
            well_names_full.extend(well_names)
            experiment_names_full.extend(experiment_names)
            X_metadata_full.extend(X_metadata)

            # Deleting these results to save on RAM
            if X_full is None:
                X_full = X
            else:
                X_full = np.concatenate((X_full, X), axis=0)

            if X_raw_full is None:
                X_raw_full = X_raw
            else:
                X_raw_full = np.concatenate((X_raw_full, X_raw), axis=0)

            if y is not None:
                if y_full is None:
                    y_full = y
                else:
                    y_full = np.concatenate((y_full, y), axis=0)

                if y_tiles_full is None:
                    y_tiles_full = y_tiles
                else:
                    y_tiles_full = np.concatenate((y_tiles_full, y_tiles), axis=0)

            # Deleting references to save on RAM
            del X, y, y_tiles, X_raw, X_metadata, bag_names, experiment_names, well_names, errors, loaded_files_list

            # Printing current loaded overview
            X_s_full = str(utils.byteSizeString(utils.listToBytes(X_full)))
            X_s_raw_full = str(utils.byteSizeString(utils.listToBytes(X_raw_full)))
            y_s_full = str(utils.byteSizeString(sys.getsizeof(y_full)))
            log.write("X-size in memory (after "+str(i+1)+" batches): " + str(X_s_full))
            log.write("y-size in memory (after "+str(i+1)+" batches): " + str(y_s_full))
            log.write("X-size (raw) in memory (after "+str(i+1)+" batches): " + str(X_s_raw_full))
            del X_s_full, y_s_full, X_s_raw_full

    if X_full is None:
        log.write('No files were loaded!')
        assert False

    log.write('Debug list size "X_full": ' + str(len(X_full)), print_to_console=False)
    log.write('Debug list size "y_full": ' + str(len(y_full)), print_to_console=False)
    log.write('Debug list size "y_tiles_full": ' + str(len(y_tiles_full)), print_to_console=False)
    log.write('Debug list size "X_raw_full": ' + str(len(X_raw_full)), print_to_console=False)
    log.write('Debug list size "bag_names": ' + str(len(bag_names_full)), print_to_console=False)
    log.write('Debug list size "X_metadata_full": ' + str(len(X_metadata_full)), print_to_console=False)

    assert len(X_full) == len(y_full)
    assert len(X_full) == len(y_tiles_full)
    assert len(X_full) == len(X_raw_full)
    assert len(X_full) == len(bag_names_full)
    assert len(X_full) == len(X_metadata_full)

    return X_full, y_full, y_tiles_full, X_raw_full, X_metadata_full, bag_names_full, experiment_names_full, well_names_full, error_list, loaded_files_list_full


# Main Loading function
def load_bags_json(source_dir: str, max_workers: int, normalize_enum: int, label_0_well_indices: [int],
                   channel_inclusions: [bool], label_1_well_indices: [int], constraints_1: [int], constraints_0: [int],
                   gp_current: int = 1, gp_max: int = 1, include_raw: bool = True):
    files = os.listdir(source_dir)
    log.write('[' + str(gp_current) + '/' + str(gp_max) + '] Loading from source: ' + source_dir)
    loaded_files_list = []

    terminal_columns = None
    if platform == "linux" or platform == "linux2":
        try:
            terminal_rows, terminal_columns = os.popen('stty size', 'r').read().split()
            terminal_columns = int(terminal_columns)
        except Exception as e:
            terminal_columns = None

    executor = ThreadPoolExecutor(max_workers=max_workers)
    future_list = []
    worker_verbose: bool = max_workers == 1

    for file in files:
        # print('Considering source file: ' + file)
        filepath = source_dir + os.sep + file

        if file.endswith('.json.zip'):
            loaded_files_list.append(filepath + os.sep + file)
            future = executor.submit(unzip_and_read_JSON,
                                     filepath,
                                     worker_verbose,
                                     normalize_enum,
                                     label_0_well_indices,
                                     label_1_well_indices,
                                     constraints_0,
                                     constraints_1,
                                     channel_inclusions,
                                     include_raw
                                     )
            future_list.append(future)

        if file.endswith('.json'):
            if os.path.exists(filepath + '.zip'):
                # print('A zipped version also existed. Skipping.')
                continue

            loaded_files_list.append(filepath + os.sep + file)
            future = executor.submit(read_JSON_file,
                                     filepath,
                                     worker_verbose,
                                     normalize_enum,
                                     label_0_well_indices,
                                     label_1_well_indices,
                                     constraints_0,
                                     constraints_1,
                                     channel_inclusions,
                                     include_raw
                                     )
            future_list.append(future)

    start_time = gct(raw=True)
    all_finished: bool = False
    executor.shutdown(wait=False)

    while not all_finished:
        finished_count = 0
        error_count = 0

        for future in future_list:
            if future.done():
                finished_count = finished_count + 1

                e = future.exception()
                if e is not None:
                    error_count = error_count + 1

        line_print('[' + str(gp_current) + ' / ' + str(gp_max) + '] ' + str(
            max_workers) + ' Threads running. Finished: ' + str(finished_count) + '/' + str(
            len(future_list)) + '. Errors: ' + str(
            error_count) + '. Running: ' + get_time_diff(
            start_time) + '. ' + gct(), max_width=terminal_columns, include_in_log=False)
        all_finished = finished_count == len(future_list)
        time.sleep(1)

    experiment_name = '<unknown experiment>'
    X = []
    y = []
    X_raw = []
    X_metadata = []
    bag_names = []
    well_names = []
    experiment_names = []
    y_tiles = []
    error_list = []
    print('')
    log.write('Parallel execution resulted in ' + str(len(future_list)) + ' futures.')
    print('\n')

    for i in range(len(future_list)):
        future = future_list[i]
        line_print('Extracting future: ' + str(i + 1) + '/' + str(len(future_list)), include_in_log=False)

        e = future.exception()
        if e is None:
            X_f, y_f, y_f_tiles, x_f_raw, x_f_metadata_raw, bag_name, experiment_name, well_name = future.result()
            if X_f is not None and y_f is not None and y_f_tiles is not None and x_f_raw is not None and len(
                    X_f) != 0:
                X.append(X_f)
                y.append(y_f)
                y_tiles.append(y_f_tiles)
                X_raw.append(x_f_raw)
                X_metadata.append(x_f_metadata_raw)
                experiment_names.append(experiment_name)
                well_names.append(well_name)
                bag_names.append(bag_name)
        else:
            log.write('\n' + gct() + 'Error extracting future results: ' + str(e) + '\n')
            tb = traceback.TracebackException.from_exception(e)
            for line in tb.stack:
                log.write(str(line))
            error_list.append(e)

    print('\n')
    log.write('\nFully Finished Loading Path. Files: ' + str(loaded_files_list))

    # Deleting the futures and the future list to immediately releasing the memory.
    future_count = len(future_list)
    del future_list[:]
    del future_list

    log.write('Debug list size "X": ' + str(len(X)))
    log.write('Debug list size "y": ' + str(len(y)))
    log.write('Debug list size "y_tiles": ' + str(len(y_tiles)))
    log.write('Debug list size "X_raw": ' + str(len(X_raw)))
    log.write('Debug list size "bag_names": ' + str(len(bag_names)))
    log.write('Debug list size "bag_names": ' + str(len(X_metadata)))

    X_s = str(utils.byteSizeString(utils.listToBytes(X)))
    X_s_raw = str(utils.byteSizeString(utils.listToBytes(X_raw)))
    y_s = str(utils.byteSizeString(sys.getsizeof(y)))
    log.write("X-size in memory of " + experiment_name + " alone): " + str(X_s))
    log.write("y-size in memory of " + experiment_name + " alone): " + str(y_s))
    log.write("X-size (raw) in memory of " + experiment_name + " alone): " + str(X_s_raw))

    log.write('Wells with label 0 from '+experiment_name+': ' + str(len(np.where(np.asarray(y) == 0)[0]))+' / '+str(future_count))
    log.write('Wells with label 1 from '+experiment_name+': ' + str(len(np.where(np.asarray(y) == 1)[0]))+' / '+str(future_count))
    del X_s, X_s_raw, y_s

    assert len(X) == len(y)
    assert len(X) == len(y_tiles)
    assert len(X) == len(X_raw)
    assert len(X) == len(bag_names)
    assert len(X) == len(experiment_names)
    assert len(X) == len(well_names)
    assert len(X) == len(X_metadata)

    # In case we have a R terminal running, we should keep it busy so it does not time out
    r.has_connection()

    return X, y, y_tiles, X_raw, X_metadata, bag_names, experiment_names, well_names, error_list, loaded_files_list


####


def unzip_and_read_JSON(filepath, worker_verbose, normalize_enum, label_0_well_indices: [int],
                        label_1_well_indices: [int], constraints_0: [int], constraints_1: [int],
                        channel_inclusions: [bool], include_raw: bool = True):
    if worker_verbose:
        log.write('Unzipping and reading json: ' + filepath)
    threading.current_thread().setName('Unzipping & Reading JSON: ' + filepath)

    # handling the case, if a json file has been zipped
    # The idea: Read the zip, unzip it in ram and parse the byte stream directly as a string!
    input_zip = ZipFile(filepath)
    zipped_data_name = input_zip.namelist()[0]
    data = input_zip.read(zipped_data_name)
    input_zip.close()

    data = json.loads(data)
    X, y_bag, y_tiles, X_raw, X_metadata, bag_name, experiment_name, well_name = parse_JSON(filepath,
                                                                                            str(zipped_data_name), data,
                                                                                            worker_verbose,
                                                                                            normalize_enum,
                                                                                            label_0_well_indices=label_0_well_indices,
                                                                                            label_1_well_indices=label_1_well_indices,
                                                                                            channel_inclusions=channel_inclusions,
                                                                                            constraints_1=constraints_1,
                                                                                            constraints_0=constraints_0,
                                                                                            include_raw=include_raw)

    if worker_verbose:
        log.write('File Shape: ' + filepath + ' -> ')
        log.write("X-shape: " + str(np.asarray(X).shape))
        log.write("y-shape: " + str(np.asarray(y_bag).shape))

    return X, y_bag, y_tiles, X_raw, X_metadata, bag_name, experiment_name, well_name


####

def read_JSON_file(filepath: str, worker_verbose: bool, normalize_enum: int, label_0_well_indices: [int],
                   label_1_well_indices: [int], constraints_0: [int], constraints_1: [int],
                   channel_inclusions: [bool], include_raw: bool = True):
    if worker_verbose:
        log.write('Reading json: ' + filepath)

    # Renaming the thread, so profilers can keep up
    threading.current_thread().setName('Loading JSON: ' + filepath)
    f = open(filepath)
    data = json.load(f)
    f.close()

    return parse_JSON(filepath=filepath, zipped_data_name=filepath, worker_verbose=worker_verbose,
                      normalize_enum=normalize_enum, label_0_well_indices=label_0_well_indices,
                      label_1_well_indices=label_1_well_indices, include_raw=include_raw,
                      channel_inclusions=channel_inclusions, json_data=data,
                      constraints_1=constraints_1, constraints_0=constraints_0)


####


def parse_JSON(filepath: str, zipped_data_name: str, json_data, worker_verbose: bool, normalize_enum: int,
               label_0_well_indices: [int], label_1_well_indices: [int], constraints_1: [int], constraints_0: [int],
               channel_inclusions: [bool], include_raw: bool = True) -> (np.ndarray, int, [int], np.ndarray, str):
    # Setting up arrays
    X = []
    X_raw = []
    X_metadata = []
    label = None
    y_tiles = None

    # Renaming the thread, so profilers can keep up
    threading.current_thread().setName('Parsing JSON: ' + zipped_data_name)

    assert len(channel_inclusions) == 3
    assert len(constraints_1) == 3
    assert len(constraints_0) == 3
    inclusions_count = sum([float(channel_inclusions[i]) for i in range(len(channel_inclusions))])

    # Reading JSON meta data
    width = json_data['tileWidth']
    height = json_data['tileHeight']
    bit_depth = json_data['bit_depth']
    well = json_data['well']
    experiment_name = zipped_data_name[:zipped_data_name.find('-')]
    bag_name = experiment_name + '-' + well

    # Reading metadata for the well, if it exists
    well_image_width: int = 5520
    well_image_height: int = 5520
    if 'w' in json_data.keys() and 'h' in json_data.keys():
        well_image_width = int(json_data['w'])
        well_image_height = int(json_data['h'])
    well_letter, well_number = extract_well_info(well, verbose=worker_verbose)

    # Reading metadata for the experiment
    metadata_path = paths.mil_metadata_file_linux
    if sys.platform == 'win32':
        metadata_path = paths.mil_metadata_file_win
    if not os.path.exists(metadata_path):
        log.write('Failed to locate metadata file: ' + metadata_path)
        assert False

    plate_metadata_out_path = os.path.dirname(filepath)
    if not sys.platform == 'win32':
        plate_metadata_out_path = os.path.dirname(plate_metadata_out_path)
    plate_metadata_out_path = plate_metadata_out_path + os.sep + 'bmc_metadata_preview'
    plate_metadata = get_experiment_metadata(metadata_path=metadata_path, experiment_name=experiment_name,
                                             out_dir=plate_metadata_out_path)

    # bit_max = np.info('uint' + str(bit_depth)).max
    bit_max = pow(2, bit_depth) - 1

    # Setting label to match the param
    label = -1
    if type(label_1_well_indices) == list and type(label_0_well_indices) == list:
        # Checking if the labels are in 'List' Format. If so, those bags are assigned the specific label
        # Example #1: label_1_well_indices = [0, 1, 2, 3, 4]
        if well_number in label_1_well_indices:
            label = 1
        elif well_number in label_0_well_indices:
            label = 0
    elif type(label_1_well_indices) == float and type(label_0_well_indices) == float:
        # Checking if the labels are in 'List' Format. If so, those bags are assigned the specific label
        # Example #1: label_1_well_indices = 0.2

        # First, checking if the metadata is None
        if plate_metadata is None or plate_metadata.compound_oligo_diff is None:
            log.write('\n\n== WARNING ==\n'
                      'Cannot set label based on plate BMC for: ' + zipped_data_name + '! No metadata found!')
        else:
            oligo_diff = plate_metadata.compound_oligo_diff[well_number - plate_metadata.well_control]

            if worker_verbose:
                log.write('\n\n' + zipped_data_name + ' read BMC: ' + str(oligo_diff) + '. Compare to: <=' + str(
                    label_1_well_indices) + ' | >=' + str(label_0_well_indices) + '\n\n')
            if oligo_diff >= label_1_well_indices:
                label = 1
            elif oligo_diff <= label_0_well_indices:
                label = 0

    if label == -1:
        # Labels were not set. Skipping this bag.
        if worker_verbose:
            log.write('This bag has no label assigned. Removing.')

        return X, label, y_tiles, X_raw, X_metadata, bag_name, experiment_name, well

    if worker_verbose:
        log.write('Reading JSON: ' + str(width) + 'x' + str(height) + '. Bits: ' + str(bit_depth))

    # Deciding on what constraints to use, based on label
    used_constraints = constraints_0
    if label == 1:
        used_constraints = constraints_1

    # Initializing "best" min / max values for every cell in the tile
    best_well_min_r = bit_max
    best_well_min_g = bit_max
    best_well_min_b = bit_max
    best_well_max_r = 0
    best_well_max_g = 0
    best_well_max_b = 0

    # Reading tiles
    json_data = json_data['tiles']
    keys = list(json_data.keys())
    if len(keys) == 0:
        if worker_verbose:
            print('The read bag is empty!')
        return X, label, y_tiles, X_raw, X_metadata, bag_name, experiment_name, well

    for i in range(len(keys)):
        # print('Processing tile: ' + str(i + 1) + '/' + str(len(keys)))
        key = keys[i]
        current_tile = json_data[str(key)]

        # Reading channels
        r = np.array(current_tile['r'])
        g = np.array(current_tile['g'])
        b = np.array(current_tile['b'])

        # Extracting nuclei information from the metadata
        count_nuclei: int = 0
        count_oligos: int = 0
        count_neurons: int = 0
        if 'nuclei' in current_tile.keys():
            count_nuclei = int(current_tile['nuclei'])
            count_oligos = int(current_tile['oligos'])
            count_neurons = int(current_tile['neurons'])

        # Reading metadata for the tile
        pos_x = math.nan
        pos_y = math.nan
        if 'x' in current_tile.keys() and 'y' in current_tile.keys():
            pos_x = int(current_tile['x'])
            pos_y = int(current_tile['y'])
        metadata = TileMetadata(experiment_name=experiment_name, well_letter=well_letter, well_number=well_number,
                                pos_x=pos_x, pos_y=pos_y, well_image_width=well_image_width,
                                count_nuclei=count_nuclei, count_oligos=count_oligos, count_neurons=count_neurons,
                                plate_metadata=plate_metadata,
                                well_image_height=well_image_height, read_from_source=True)
        X_metadata.append(metadata)

        # Checking the constraints...
        if count_nuclei < used_constraints[0] or count_oligos < used_constraints[1] or count_neurons < used_constraints[
            2]:
            # The constraints were not met...
            if worker_verbose:
                log.write('Tile with label ' + str(label) + ' did not meet the constraints! My values: ' + str(
                    count_nuclei) + ', ' + str(count_oligos) + ', ' + str(count_neurons) + ' - ' + str(
                    used_constraints))
            continue

        if include_raw:
            # Reshaping the color images to a 2 dimensional array
            raw_r = np.reshape(r, (width, height))
            raw_g = np.reshape(g, (width, height))
            raw_b = np.reshape(b, (width, height))

            # Normalizing r,g,b based on max bit depth to convert them to 8 bit int
            raw_r = raw_r / bit_max * 255
            raw_g = raw_g / bit_max * 255
            raw_b = raw_b / bit_max * 255

            # Concatenating the color images to a rgb image
            raw_rgb = np.dstack((raw_r, raw_g, raw_b))
            raw_rgb = raw_rgb.astype('uint8')
            X_raw.append(raw_rgb)
            del raw_r, raw_g, raw_b, raw_rgb

        # 0 = no normalisation
        # 1 = normalize every cell between 0 and 255 (8 bit)
        # 2 = normalize every cell individually with every color channel independent
        # 3 = normalize every cell individually with every color channel using the min / max of all three
        # 4 = normalize every cell but with bounds determined by the brightest cell in the same well
        # 5 = z-score every cell individually with every color channel independent
        # 6 = z-score every cell individually with every color channel using the mean / std of all three

        r_min = min(r)
        g_min = min(g)
        b_min = min(b)
        r_max = max(r)
        g_max = max(g)
        b_max = max(b)

        min_list = []
        max_list = []
        if channel_inclusions[0]:
            min_list.append(r_min)
            max_list.append(r_max)
        if channel_inclusions[1]:
            min_list.append(g_min)
            max_list.append(g_max)
        if channel_inclusions[2]:
            min_list.append(b_min)
            max_list.append(b_max)
        rgb_min = min(min_list)
        rgb_max = max(max_list)

        # Updating 'best' min / max values
        best_well_min_r = min(best_well_min_r, r_min)
        best_well_min_g = min(best_well_min_g, g_min)
        best_well_min_b = min(best_well_min_b, b_min)
        best_well_max_r = max(best_well_max_r, r_max)
        best_well_max_g = max(best_well_max_g, g_max)
        best_well_max_b = max(best_well_max_b, b_max)

        # 2 = normalize every cell individually with every color channel independent
        if normalize_enum == 2:
            r = normalize_np(r, r_min, r_max)
            g = normalize_np(g, g_min, g_max)
            b = normalize_np(b, b_min, b_max)

        # 3 = normalize every cell individually with every color channel using the min / max of all three
        if normalize_enum == 3:
            r = normalize_np(r, rgb_min, rgb_max)
            g = normalize_np(g, rgb_min, rgb_max)
            b = normalize_np(b, rgb_min, rgb_max)

        # 5 = z-score every cell individually with every color channel independent
        if normalize_enum == 5:
            r = z_score(r, axis=0)
            g = z_score(g, axis=0)
            b = z_score(b, axis=0)

        # 6 = z-score every cell individually with every color channel using the mean / std of all three
        if normalize_enum == 6:
            standardize_list = []
            if channel_inclusions[0]:
                standardize_list.append(r)
            if channel_inclusions[1]:
                standardize_list.append(g)
            if channel_inclusions[2]:
                standardize_list.append(b)

            rgb = np.concatenate(standardize_list, axis=0)
            mean = np.mean(rgb)
            std = np_std(rgb, axis=0, mean=mean)
            r = z_score(r, axis=0, std=std, mean=mean)
            g = z_score(g, axis=0, std=std, mean=mean)
            b = z_score(b, axis=0, std=std, mean=mean)
            del rgb, mean, std, standardize_list

        # Reshaping the color images to a 2 dimensional array
        r = np.reshape(r, (width, height))
        g = np.reshape(g, (width, height))
        b = np.reshape(b, (width, height))

        # Concatenating the color images to a rgb image
        rgb = np.dstack((r, g, b))

        # 1 = normalize every cell between 0 and 255 (8 bit)
        if normalize_enum == 1:
            rgb = normalize_np(rgb, 0, bit_max)

        # Appending the current tile to the list
        X.append(rgb)

        del r
        del g
        del b
        del rgb

    # 4 = normalize every cell individually with every color channel using the min / max of all three
    # 7 = z-score every cell individually with every color channel independent using all samples in the bag
    # 8 = z-score every cell individually with every color channel using the mean / std of all three from all samples in the bag
    if normalize_enum == 4 or normalize_enum == 7 or normalize_enum == 8 or normalize_enum == 9 or normalize_enum == 10:
        if normalize_enum == 7:
            bag_mean_r, bag_std_r = get_bag_mean(X, axis=0)
            bag_mean_g, bag_std_g = get_bag_mean(X, axis=1)
            bag_mean_b, bag_std_b = get_bag_mean(X, axis=2)
        if normalize_enum == 8:
            bag_mean, bag_std = get_bag_mean(X, channel_inclusions=channel_inclusions)

        for i in range(len(X)):
            current_x = X[i]
            current_r = current_x[:, :, 0]
            current_g = current_x[:, :, 1]
            current_b = current_x[:, :, 2]

            if normalize_enum == 4 or normalize_enum == 9 or normalize_enum == 10:
                current_r = normalize_np(current_r, best_well_min_r, best_well_max_r)
                current_g = normalize_np(current_g, best_well_min_g, best_well_max_g)
                current_b = normalize_np(current_b, best_well_min_b, best_well_max_b)

                if normalize_enum == 9:
                    current_r = z_score(current_r, axis=0)
                    current_g = z_score(current_g, axis=0)
                    current_b = z_score(current_b, axis=0)
                if normalize_enum == 10:
                    standardize_list = []
                    if channel_inclusions[0]:
                        standardize_list.append(current_r)
                    if channel_inclusions[1]:
                        standardize_list.append(current_g)
                    if channel_inclusions[2]:
                        standardize_list.append(current_b)

                    rgb = np.concatenate(standardize_list, axis=0)
                    mean = np.mean(rgb)
                    std = np_std(rgb, axis=0, mean=mean)
                    current_r = z_score(current_r, axis=0, std=std, mean=mean)
                    current_g = z_score(current_g, axis=0, std=std, mean=mean)
                    current_b = z_score(current_b, axis=0, std=std, mean=mean)
                    del mean, std, rgb, standardize_list
            if normalize_enum == 7:
                current_r = z_score(current_r, mean=bag_mean_r, std=bag_std_r)
                current_g = z_score(current_g, mean=bag_mean_g, std=bag_std_g)
                current_b = z_score(current_b, mean=bag_mean_b, std=bag_std_b)
            if normalize_enum == 8:
                current_r = z_score(current_r, mean=bag_mean, std=bag_std)
                current_g = z_score(current_g, mean=bag_mean, std=bag_std)
                current_b = z_score(current_b, mean=bag_mean, std=bag_std)

            current_rgb = np.dstack((current_r, current_g, current_b))
            del current_r
            del current_g
            del current_b

            X[i] = current_rgb

    # Checking the well inclusions and setting all unwanted channels to 0
    if inclusions_count > 1:
        for i in range(len(X)):
            current_x = X[i]
            current_r = current_x[:, :, 0]
            current_g = current_x[:, :, 1]
            current_b = current_x[:, :, 2]

            # Checking if channels need to be removed
            if not channel_inclusions[0]:
                # r
                current_r = np.array(np.zeros((current_g.shape[0], current_g.shape[1])))
            if not channel_inclusions[1]:
                # g
                current_g = np.array(np.zeros((current_g.shape[0], current_g.shape[1])))
            if not channel_inclusions[2]:
                # b
                current_b = np.array(np.zeros((current_b.shape[0], current_b.shape[1])))

            current_rgb = np.dstack((current_r, current_g, current_b))
            X[i] = current_rgb
            del current_x, current_r, current_g, current_rgb

    # Placing sample tiles in a list
    y_tiles = []
    for i in range(len(X)):
        # Actually writing bag labels
        y_tiles.append(label)

    # Checking for NaNs (last thing to do, after normalizing)
    for i in range(len(X)):
        current_rgb = X[i]

        # Checking if there is 'NaN' in the stack?
        if np.any(np.isnan(current_rgb)):
            num_nan = np.count_nonzero(np.isnan(current_rgb))

            nan_msg = 'Warning! Tile "' + str(X_metadata[i]) + '" has NaNs! Count: ' + str(num_nan)
            log.write(nan_msg, print_to_console=worker_verbose)

            # Setting NaNs to 0
            current_rgb[np.where(np.isnan(current_rgb))] = 0

            X[i] = current_rgb

    # Checking if there is actually something loaded
    if len(X) == 0:
        X = None
        X_raw = None
        label = None
        y_tiles = None

        log.write('JSON file has no data: ' + str(filepath))
    else:
        X = np.asarray(X)
        X_raw = np.asarray(X_raw)

        # Checking if X_raw is a 8 bit unsigned int
        assert X_raw.dtype == np.uint8

        # X = X.astype(np.float16)
        # TODO make this optional via argument

        # Saving preview (if it exists)
        if used_constraints is not None:
            try:
                save_save_bag_preview(X=X, out_dir_base=filepath, experiment_name=experiment_name, well=well,
                                      normalize_enum=normalize_enum, preview_constraints=used_constraints,
                                      channel_inclusions=channel_inclusions,
                                      bit_depth=bit_depth, X_raw=X_raw, verbose=worker_verbose)
            except Exception as e:
                # TODO handle this better
                pass

        # Checking the 'tegredy of the bag and its samples
        assert len(X) == len(X_raw)
        assert len(X) == len(y_tiles)
        assert (label == 0 or label == 1)

    # All good. Returning.
    label = int(label)
    return X, label, y_tiles, X_raw, X_metadata, bag_name, experiment_name, well


def save_save_bag_preview(X, out_dir_base, experiment_name, well, preview_constraints, normalize_enum, bit_depth,
                          channel_inclusions: [bool], X_raw, dpi: int = (1337 * 1.5), colormap_name: str = 'jet',
                          v_min: float = -3.0, v_max=3.0, outline: int = 2, verbose: bool = False):
    width = None
    height = None
    z_mode = normalize_enum > 4

    if platform == "linux" or platform == "linux2":
        # going one level up for linux dirs
        out_dir_base = os.path.dirname(out_dir_base)

    channel_inclusions_label = str(channel_inclusions).replace('[', '').replace(']', '').replace(',', '-').replace(
        ' ', '').lower()
    preview_constraints_label = str(preview_constraints).replace('[', '').replace(']', '').replace(',', '-').replace(
        ' ', '')

    out_dir = os.path.dirname(out_dir_base)
    out_dir = out_dir + os.sep + 'bag_previews' + os.sep + 'normalize-' + str(
        normalize_enum) + os.sep + 'channels-' + channel_inclusions_label + os.sep + 'constraints-' + preview_constraints_label + os.sep + experiment_name + os.sep
    os.makedirs(out_dir, exist_ok=True)
    bit_max = pow(2, bit_depth) - 1

    out_file_name = out_dir + experiment_name + '-' + well + '.png'
    out_file_name_raw = out_dir + experiment_name + '-' + well + '-raw.png'
    # print('debug out fname: ' + out_file_name)
    if os.path.exists(out_file_name):
        # Preview already exists. Nothing to do.
        if verbose:
            log.write('Preview file already exists. Skipping: ' + out_file_name)
        return

    rgb_samples = []
    rgb_samples_raw = []
    z_r_samples = []
    z_g_samples = []
    z_b_samples = []
    z_out_dir = out_dir + 'z_scores' + os.sep
    z_out_file_name_r = z_out_dir + experiment_name + '-' + well + '_r.png'
    z_out_file_name_g = z_out_dir + experiment_name + '-' + well + '_g.png'
    z_out_file_name_b = z_out_dir + experiment_name + '-' + well + '_b.png'

    # Collecting every samples in the bag to save them on the device
    assert len(X) > 0
    global thread_lock

    for i in range(len(X)):
        sample = X[i]
        sample_raw = X_raw[i]
        width, height, _ = sample.shape

        # Storing raw samples
        sample_raw = mil_metrics.outline_rgb_array(sample_raw, None, None, outline=outline,
                                                   override_colormap=[255, 255, 255])
        rgb_samples_raw.append(sample_raw)

        # Storing the actual sample, based if it's z-scored or normalized
        if z_mode:
            z_score_channels = z_score_to_rgb(img=sample, colormap=colormap_name, a_min=v_min, a_max=v_max)

            sample_r = mil_metrics.outline_rgb_array(z_score_channels[0], None, None, outline=outline,
                                                     override_colormap=[255, 255, 255])
            sample_g = mil_metrics.outline_rgb_array(z_score_channels[1], None, None, outline=outline,
                                                     override_colormap=[255, 255, 255])
            sample_b = mil_metrics.outline_rgb_array(z_score_channels[2], None, None, outline=outline,
                                                     override_colormap=[255, 255, 255])
            z_r_samples.append(sample_r)
            z_g_samples.append(sample_g)
            z_b_samples.append(sample_b)
        else:
            if normalize_enum == 0:
                sample = (sample / bit_max) * 255
            else:
                sample = sample * 255

            sample = sample.astype(np.uint8)
            sample = mil_metrics.outline_rgb_array(sample, None, None, outline=outline,
                                                   override_colormap=[255, 255, 255])
            rgb_samples.append(sample)

    # Saving the image
    if z_mode:
        os.makedirs(z_out_dir, exist_ok=True)
        fused_image_r = mil_metrics.fuse_image_tiles(images=z_r_samples, image_width=width, image_height=height)
        fused_image_g = mil_metrics.fuse_image_tiles(images=z_g_samples, image_width=width, image_height=height)
        fused_image_b = mil_metrics.fuse_image_tiles(images=z_b_samples, image_width=width, image_height=height)

        # Saving single chanel z-score rgb images
        fused_images = [fused_image_r, fused_image_g, fused_image_b]
        fused_width, fused_height, _ = fused_image_r.shape
        plt.imsave(z_out_file_name_r, fused_image_r)
        plt.imsave(z_out_file_name_g, fused_image_g)
        plt.imsave(z_out_file_name_b, fused_image_b)

        # saving the raw included version
        out_image_raw = mil_metrics.fuse_image_tiles(images=rgb_samples_raw, image_width=width, image_height=height)
        out_image_raw = mil_metrics.fuse_image_tiles(
            images=[fused_image_r, fused_image_g, fused_image_b, out_image_raw],
            image_width=fused_width, image_height=fused_height)
        plt.imsave(out_file_name_raw, out_image_raw)

        # Blocking all other threads so pyplot doesn't overwrite itself
        thread_lock.acquire(blocking=True)

        # saving a well formatted version
        plt.clf()
        for i in range(3):
            current_channel = fused_images[i]
            plt.subplot(1, 3, i + 1, adjustable='box', aspect=1)
            plt.xticks([], [])
            plt.yticks([], [])

            # Creating a dummy image for the color bar to fit
            img = plt.imshow(np.array([[v_min, v_max]]), cmap=colormap_name)
            img.set_visible(False)
            c_bar = plt.colorbar(orientation='vertical', fraction=0.046)
            if i == 2:
                c_bar.ax.set_ylabel('z-score', rotation=270)

            plt.imshow(current_channel)
            plt.title(['r', 'g', 'b'][i])

            plt.suptitle(
                'z-scores: ' + experiment_name + ' - ' + well + '\n\n' + normalize_enum_descriptions[normalize_enum])
            plt.tight_layout()
            plt.autoscale()
            plt.savefig(out_file_name, dpi=dpi, bbox_inches="tight")

        # Releasing them other threads
        thread_lock.release()
    else:
        out_image = mil_metrics.fuse_image_tiles(images=rgb_samples, image_width=width, image_height=height)
        plt.imsave(out_file_name, out_image)

        fused_width, fused_height, _ = out_image.shape
        out_image_raw = mil_metrics.fuse_image_tiles(images=rgb_samples_raw, image_width=width, image_height=height)
        out_image_raw = mil_metrics.fuse_image_tiles(images=[out_image, out_image_raw], image_width=fused_width,
                                                     image_height=fused_height)
        plt.imsave(out_file_name_raw, out_image_raw)


def get_bag_mean(n: [np.ndarray], axis: int = None, channel_inclusions=default_channel_inclusions_all):
    combined_x = np.zeros(0)
    for i in range(len(n)):
        current_x = n[i]
        dim_x = current_x.shape[0]
        dim_y = current_x.shape[1]

        if axis is None:
            current_r = current_x[:, :, 0].reshape(dim_x * dim_y)
            current_g = current_x[:, :, 1].reshape(dim_x * dim_y)
            current_b = current_x[:, :, 2].reshape(dim_x * dim_y)

            if channel_inclusions[0]:
                combined_x = np.append(combined_x, current_r)
            if channel_inclusions[1]:
                combined_x = np.append(combined_x, current_g)
            if channel_inclusions[2]:
                combined_x = np.append(combined_x, current_b)
            del current_r, current_g, current_b
        else:
            if channel_inclusions[axis]:
                current_axis = current_x[:, :, axis].reshape(dim_x * dim_y)
                combined_x = np.append(combined_x, current_axis)
                del current_axis

    mean = np.mean(combined_x)
    std = np_std(n=combined_x, mean=mean)
    return mean, std[0]


def convert_bag_to_batch(bags: [np.ndarray], labels: [int] = None, y_tiles: [[int]] = None):
    ''' Convert bag and label pairs into batch format
    Inputs:
        a list of bags and a list of bag-labels

    Outputs:
        Returns a dataset (list) containing (stacked tiled instance data, bag label)
    '''
    dataset = []
    input_dim = None

    if labels is None:
        labels = [math.nan for i in range(len(bags))]
    if y_tiles is None:
        y_tiles = [[math.nan for i in range(bag.shape[0])] for bag in bags]
        # y_tiles = []
        # for bag in bags:
        #    y_tiles.append([math.nan for i in range(bag.shape[0])])

    for index, (bag, bag_label, tile_labels) in enumerate(zip(bags, labels, y_tiles)):
        batch_data = np.asarray(bag, dtype='float32')
        batch_label = np.asarray(bag_label, dtype='float32')
        batch_label_tiles = np.asarray(tile_labels, dtype='float32')
        batch_original_index = np.asarray(index, dtype='float32')
        dataset.append((batch_data, batch_label, batch_label_tiles, batch_original_index))

        input_dim = batch_data.shape[1:]

    return dataset, input_dim


def build_bags(tiles, labels):
    ''' Builds bags suited for MIL problems: A bag is a collection of a variable number of instances. The instance-level labels are not known.
    These instances are combined into a single bag, which is then given a supervised label eg. patient diagnosis label when the instances are multiple tissue instances from that same patient.

    Inputs:
        Data tiled from images with expanded dimensionality, see preprocessing.tile_wsi and .expand_dimensionality

    Outputs:
        Returns two arrays: bags, labels where each label is sorted to a bag. Number of bags == number of labels
        bag shape is [n (tiles,x,y,z) ]
    '''
    result_bags = tiles
    result_labels = []
    count = 0

    log.write(str(len(result_bags)))
    log.write(str(len(result_bags[0])))
    log.write(str(result_bags[0][0].shape))

    # check number of bags against labels
    if len(result_bags) == len(labels):
        pass

    else:
        raise ValueError(
            'Number of Bags is not equal to the number of labels that can be assigned.\nCheck your input data!')

    # this step seems to be necessary in Tensorflow... it is not possible to use one bag - one label
    for j in labels:
        number_of_instances = result_bags[count].shape[0]
        tiled_instance_labels = np.tile(labels[count], (number_of_instances, 1))
        result_labels.append(tiled_instance_labels)
        count += 1

    return result_bags, result_labels, labels


####

def normalize_np(n: np.ndarray, lower: float = None, upper: float = None) -> np.ndarray:
    """
    Using linear normalization, every entry in a given numpy array between a lower and upper bound.
    The shape of the array can be arbitrary.

    If the lower and upper bound are equal, the reulting array will contain only zeros.

    Created by Nils Förster.

    :param n: An arbitrary numpy array
    :param lower: The lower bound for normalization
    :param upper: The upper bound for normalization

    :type n: np.ndarray
    :type lower: float
    :type upper: float

    :returns: Returns a copy of the array normalized between 0 and 1, relative to the lower and upper bound
    :rtype: np.ndarray

    Examples
    ----------
    Use this example to generate ten random integers between 0 and 100. Then normalize them using this function.

    >>> n=np.random.randint(0,100,10)
    >>> normalize_np(n,0,100)

    """
    if lower is None:
        lower = np.min()
    if upper is None:
        upper = np.max()

    nnv = np.vectorize(_normalize_np_worker)
    return nnv(n, lower, upper)


####

def _normalize_np_worker(x: float, lower: float, upper: float):
    if lower == upper:
        return 0

    lower = float(lower)
    upper = float(upper)
    return (x - lower) / (upper - lower)


def z_score(n: np.ndarray, axis=None,
            mean: Union[np.ndarray, float] = None,
            std: Union[np.ndarray, float] = None) -> np.ndarray:
    """ Also often called standardization, which transforms the data into a
    distribution with a mean of 0 and a standard deviation of 1.
    Each standardized value is computed by subtracting the mean of the corresponding feature
    and then dividing by the std dev.
    X_zscr = (x-mu)/std
    """

    if mean is None:
        mean = np.mean(n, axis=axis, keepdims=True)
    if std is None:
        std = np_std(n=n, axis=axis, mean=mean)
    return (n - mean) / std


####

def np_std(n: np.ndarray, axis=None, mean: float = None) -> np.ndarray:
    if mean is None:
        mean = np.mean(n, axis=axis, keepdims=True)

    if len(n) == 0:
        log.write('RuntimeWarning: Mean of empty slice!')
        # TODO check if this is caught and handle this case!

    return np.sqrt(((n - mean) ** 2).mean(axis=axis, keepdims=True))


####


# Takes the read data and labes and creates new bags, so that bags with label 1 contain PERCENTAGE% tiles with label
# 1, while the rest is label 0. This is done by merging two adjacent input bags together.
# Yes, this discards a lot of tiles
def repack_bags_merge(X: [np.ndarray], X_raw: [np.ndarray], y: [int], bag_names: [str],
                      repack_percentage: float = 0.05, positive_bag_min_samples: int = None):
    new_x = []
    new_x_r = []
    new_y = []
    new_y_tiles = []
    new_bag_names = []

    negative_indices = np.where(np.asarray(y) == 0)[0]
    positive_indices = np.where(np.asarray(y) == 1)[0]
    random.shuffle(negative_indices)
    random.shuffle(positive_indices)
    assert len(positive_indices) > 0
    assert len(negative_indices) > 0
    assert repack_percentage > 0

    if positive_bag_min_samples is None:
        positive_bag_min_samples = 0

    log.write(
        'Starting to repack ' + str(len(X)) + ' bags. Positive: ' + str(len(positive_indices)) + '. Negative: ' + str(
            len(negative_indices)))
    log.write('Minimum positive sample size: ' + str(positive_bag_min_samples))
    print('')
    k = 0
    for i in range(len(negative_indices)):
        line_print('Trying to repack negative bag index ' + str(i + 1) + '/' + str(
            len(negative_indices)) + ' and pairing it with positive bag index ' + str(k + 1) + '/' + str(
            len(positive_indices)), include_in_log=True)
        if k == len(positive_indices):
            # cannot repack further, positive bags have been exhausted
            log.write('Exhaustion continues. This bag is carried over.')
            new_x.append(X[negative_indices[i]])
            new_x_r.append(X_raw[negative_indices[i]])
            new_y.append(0)
            new_y_tiles.append([0 for i in range(X[negative_indices[i]].shape[0])])
            new_bag_names.append(bag_names[negative_indices[i]])
            continue

        # Setting up data, extracting bags and labels
        current_label = y[negative_indices[i]]
        partner_label = y[positive_indices[k]]

        # Checking if labels match. If they do, we have no alternating inner-outer migration ring pattern!
        if current_label == partner_label:
            raise Exception('Bag partners hd the same label!')

        negative_bag = X[negative_indices[i]]
        positive_bag = X[positive_indices[k]]
        negative_bag_raw = X_raw[negative_indices[i]]
        positive_bag_raw = X_raw[positive_indices[k]]
        negative_bag_name = bag_names[negative_indices[i]]
        positive_bag_name = bag_names[positive_indices[k]]
        del partner_label, current_label

        if i % 2 == 0:
            log.write('Repacking negative bag "' + negative_bag_name + '" (Index ' + str(i + 1) + '/' + str(
                len(negative_indices)) + ') by leaving it unchanged.')

            # This new entry is all label 0. Thus, nothing needs to change.
            new_x.append(negative_bag)
            new_x_r.append(negative_bag_raw)
            new_y.append(0)
            new_bag_names.append(negative_bag_name)
            new_y_tiles.append([0 for i in range(negative_bag.shape[0])])
        else:
            log.write('Repacking negative bag "' + negative_bag_name + '" (Index ' + str(i + 1) + '/' + str(
                len(negative_indices)) + ') and pairing it with positive bag "' + positive_bag_name + '" (Index ' + str(
                k + 1) + '/' + str(len(positive_indices)) + ').')

            # This bag will have label 1.
            # As such, we will take all zero-label samples from the current bag and add random PARAM-% samples from the positive bag.
            original_count = negative_bag.shape[0]
            repack_count = math.ceil(original_count * repack_percentage + 1)
            repack_count = min(repack_count, positive_bag.shape[0] - 2)
            repack_count = max(repack_count, 0)

            negative_bag_original = np.array(negative_bag, copy=True)
            negative_bag_raw_original = np.array(negative_bag_raw, copy=True)

            repacked_tiles = 0
            for j in range(repack_count):
                if repack_count == 0 or positive_bag.shape[0] <= 0:
                    # There are no more tiles in the positive bag. Not repacking.
                    continue
                elif repack_count == 1 or positive_bag.shape[0] == 1:
                    # There is exactly one tile in the positive bag left.
                    move_index = 0
                else:
                    # There is more than one tile in the positive bag. Picking one at random.
                    move_index = random.randrange(0, positive_bag.shape[0])

                positive_sample = positive_bag[move_index]
                positive_sample_raw = positive_bag_raw[move_index]
                positive_bag = np.delete(positive_bag, move_index, axis=0)
                positive_bag_raw = np.delete(positive_bag_raw, move_index, axis=0)

                positive_sample = np.expand_dims(positive_sample, 0)
                positive_sample_raw = np.expand_dims(positive_sample_raw, 0)
                negative_bag = np.append(negative_bag, positive_sample, axis=0)
                negative_bag_raw = np.append(negative_bag_raw, positive_sample_raw, axis=0)

                repacked_tiles = repacked_tiles + 1

            current_tile_labels = [0 for _ in range(original_count)]
            current_tile_labels.extend([1 for _ in range(repacked_tiles)])

            if 1 in current_tile_labels and repacked_tiles > 0 and repacked_tiles >= positive_bag_min_samples:
                # The new bag was successfully mixed and constructed
                new_y.append(1)
                new_x.append(negative_bag)
                new_x_r.append(negative_bag_raw)
                new_y_tiles.append(current_tile_labels)
                new_bag_names.append('[' + negative_bag_name + ', ' + positive_bag_name + ']')
            else:
                # Despite trying to repack the data, there were no tiles repacked actually!
                # This bag is discarded, just to be safe!
                if repacked_tiles < positive_bag_min_samples:
                    log.write('Wanted to repack ' + str(repacked_tiles) + ', but minimum positive tiles is ' + str(
                        positive_bag_min_samples))

                log.write('Warning! Positive bag #' + str(
                    k) + ' ("' + positive_bag_name + '") failed to repack! This bag is discarded!')

                # Yet, the (unchanged) negative bag can be appended to the list. Nice.
                new_x.append(negative_bag_original)
                new_x_r.append(negative_bag_raw_original)
                new_y.append(0)
                new_y_tiles.append([0 for i in range(negative_bag_original.shape[0])])
                new_bag_names.append(negative_bag_name)

            # incrementing k, so the next iteration can pick the next positive bag
            k = k + 1
            if k == len(positive_indices):
                # All positive bags exhausted. Returning.
                log.write('All positive bags have been exhausted.')
                break

    log.write('Finished repacking. Repacked bags: ' + str(len(new_x)))
    # Checking if data is correctly packed
    assert len(new_x) == len(new_y)
    assert len(new_x) == len(new_x_r)
    assert len(new_x) == len(new_bag_names)

    del X, X_raw, y
    return new_x, new_x_r, new_y, new_y_tiles, new_bag_names


def repack_bags(X: [np.array], y: [int], repack_percentage: float = 0.2):
    raise Exception("Deprecated")

    y = np.asarray(y)
    y0 = np.where(y == 0)[0]
    y1 = np.where(y == 1)[0]

    for i in range(len(y0)):
        bag_index = y0[i]
        current_x = X[bag_index]
        x_length = current_x.shape[0]
        repack_count = int(x_length * repack_percentage + 1)

        log.write(
            'Moving ' + str(repack_count) + ' out of ' + str(x_length) + ' elements from bag index ' + str(bag_index))
        for j in range(repack_count):
            move_index = random.randrange(1, x_length - j)
            target_bag_index = np.random.choice(y1)

            move_tile = current_x[move_index]
            move_tile = np.expand_dims(move_tile, 0)
            current_x = np.delete(current_x, move_index, axis=0)

            target_bag = X[target_bag_index]
            target_bag = np.append(target_bag, move_tile, axis=0)
            X[target_bag_index] = target_bag

        X[bag_index] = current_x

    return X


####


def get_experiment_metadata(metadata_path: str, experiment_name: str, out_dir: str, verbose: bool = False):
    assert os.path.exists(metadata_path)
    os.makedirs(out_dir, exist_ok=True)

    f = open(metadata_path, 'r')
    lines = f.readlines()
    f.close()
    del f

    plate_metadata = None
    found_experiment = False
    for l in lines:
        l = l.strip()
        l = l.split(';')

        line_experiment = l[1].lower()
        if line_experiment.startswith(experiment_name.lower()):
            assert not found_experiment
            found_experiment = True

            # checking if the metadata is in the right format
            assert len(l) == 18
            experiment_id = l[1]

            global plate_metadata_cache
            if experiment_id in plate_metadata_cache.keys():
                # This experiment ID has already been processed and can be read to the cache!
                plate_metadata = plate_metadata_cache[experiment_id]
            else:
                # extracting the CSV values
                compound_name = str(l[2])
                compound_cas = str(l[3])
                compound_concentration_max = float(l[4])
                compound_concentration_dilution = float(l[5])
                well_max_compound_concentration = int(l[6])
                well_control = int(l[7])
                plate_bmc30 = float(l[8])
                compound_bmc30 = float(l[9])

                # reading normalized oligo diff
                compound_oligo_diff = [float(l[10]) / 100.0,
                                       float(l[11]) / 100.0,
                                       float(l[12]) / 100.0,
                                       float(l[13]) / 100.0,
                                       float(l[14]) / 100.0,
                                       float(l[15]) / 100.0,
                                       float(l[16]) / 100.0,
                                       float(l[17]) / 100.0]

                plate_metadata = PlateMetadata(compound_name=compound_name,
                                               compound_cas=compound_cas,
                                               compound_concentration_max=compound_concentration_max,
                                               compound_concentration_dilution=compound_concentration_dilution,
                                               well_max_compound_concentration=well_max_compound_concentration,
                                               well_control=well_control,
                                               plate_bmc30=plate_bmc30,
                                               compound_oligo_diff=compound_oligo_diff,
                                               experiment_id=experiment_id,
                                               compound_bmc30=compound_bmc30
                                               )
                plate_metadata_cache[experiment_id] = plate_metadata

                # baking the BMC30 for this plate.
                # But since we use an R connection, it's best to do this via thread lock.
                global thread_lock
                thread_lock.acquire(blocking=True)
                try:
                    plate_metadata.bake_plate_bmc30(out_dir=out_dir, bmc_30_compound_concentration=compound_bmc30,
                                                    verbose=verbose)
                except Exception as e:
                    log.write('Failed to bake plate BMC30 for compound: ' + compound_name)
                    log.write(str(e), print_to_console=verbose)
                thread_lock.release()

            log.write('Comparing experiments: ' + experiment_name + ' VS ' + experiment_id, print_to_console=verbose)

    if not found_experiment:
        log.write('WARNING: ' + experiment_name + ' has no compound metadata in: ' + metadata_path)

    return plate_metadata


####

if __name__ == '__main__':
    print("This is a loader helper function that has no original main.")
