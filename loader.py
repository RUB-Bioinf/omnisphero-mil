# IMPORTS

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from sys import platform
from typing import Union
from zipfile import ZipFile

import numpy as np

from util import log
from util.utils import gct
from util.utils import get_time_diff
from util.utils import line_print

####
# Constants

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
normalize_enum_default = 1


####

def load_bags_json_batch(batch_dirs: [str], max_workers: int, normalize_enum: int):
    log.write('Looking for multiple source dirs to load json data from.')
    log.write('Batch dir: ' + str(batch_dirs))
    log.write('Normalization Protocol: ' + str(normalize_enum))

    X_full = None
    y_full = None
    error_list = []
    loaded_files_list = []

    for i in range(len(batch_dirs)):
        current_dir = batch_dirs[i]
        log.write('Considering source directory: ' + current_dir)

        if os.path.isdir(current_dir):
            X, y, errors, loaded_files_list = load_bags_json(source_dir=current_dir, max_workers=max_workers,
                                                             normalize_enum=normalize_enum, gp_current=i + 1,
                                                             gp_max=len(batch_dirs))

            error_list.extend(errors)
            if X_full is None:
                X_full = X
            else:
                X_full = np.concatenate((X_full, X), axis=0)

            if y is not None:
                if y_full is None:
                    y_full = y
                else:
                    y_full = np.concatenate((y_full, y), axis=0)

    return X_full, y_full, error_list, loaded_files_list


# Main Loading function
def load_bags_json(source_dir: str, max_workers: int, normalize_enum: int, gp_current: int = 1, gp_max: int = 1):
    files = os.listdir(source_dir)
    log.write('Loading from source: ' + source_dir)
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
                                     normalize_enum
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
                                     normalize_enum
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

    X = []
    y = []
    error_list = []
    log.write('\n')

    for i in range(len(future_list)):
        future = future_list[i]
        line_print('Extracting future: ' + str(i) + '/' + str(len(future_list)))

        e = future.exception()
        if e is None:
            X_f, y_f = future.result()
            if X_f is not None:
                X.append(X_f)
            if y_f is not None:
                y.append(y_f)
        else:
            log.write('\n' + gct() + 'Error extracting future results: ' + str(e) + '\n')
            error_list.append(e)

    print('\n')
    log.write('Fully Finished Loading Path.')

    # Deleting the futures and the future list to immediately releasing the memory.
    del future_list[:]
    del future_list

    return X, y, error_list, loaded_files_list


####


def unzip_and_read_JSON(filepath, worker_verbose, normalize_enum):
    if worker_verbose:
        print('Unzipping and reading json: ' + filepath)

    # handling the case, if a json file has been zipped
    # The idea: Read the zip, unzip it in ram and parse the byte stream directly as a string!
    input_zip = ZipFile(filepath)
    zipped_data_name = input_zip.namelist()[0]
    data = input_zip.read(zipped_data_name)
    input_zip.close()

    data = json.loads(data)
    X, y = parse_JSON(data, worker_verbose, normalize_enum)

    if worker_verbose:
        log.write('File Shape: ' + filepath + ' -> ')
        log.write("X-shape: " + str(np.asarray(X).shape))
        log.write("y-shape: " + str(np.asarray(y).shape))

    return X, y


####

def read_JSON_file(filepath, worker_verbose, normalize_enum):
    if worker_verbose:
        log.write('Reading json: ' + filepath)

    f = open(filepath)
    data = json.load(f)
    f.close()

    return parse_JSON(data, worker_verbose, normalize_enum)


####

def parse_JSON(json_data, worker_verbose, normalize_enum) -> (np.array, int):
    # Setting up arrays
    X = []
    y = None

    # Reading meta data
    width = json_data['tileWidth']
    height = json_data['tileHeight']
    bit_depth = json_data['bit_depth']

    # bit_max = np.iinfo('uint' + str(bit_depth)).max
    bit_max = pow(2, bit_depth) - 1

    # Reading label, if it exists
    if 'label' in json_data:
        label = json_data['label']
        y = int(label)

    if worker_verbose:
        log.write('Reading JSON: ' + str(width) + 'x' + str(height), '. Bits: ' + str(bit_depth))

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
    for i in range(len(keys)):
        # print('Processing tile: ' + str(i + 1) + '/' + str(len(keys)))
        key = keys[i]
        current_tile = json_data[str(key)]

        # Reading channels
        r = np.array(current_tile['r'])
        g = np.array(current_tile['g'])
        b = np.array(current_tile['b'])

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
        rgb_min = min(r_min, g_min, b_min)
        rgb_max = max(r_max, g_max, b_max)

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
            rgb = np.concatenate((r, g, b), axis=0)
            mean = np.mean(rgb)
            std = np_std(rgb, axis=0, mean=mean)
            r = z_score(r, axis=0, std=std, mean=mean)
            g = z_score(g, axis=0, std=std, mean=mean)
            b = z_score(b, axis=0, std=std, mean=mean)
            del rgb, mean, std

        # Reshaping the color images to a 2 dimensional array
        r = np.reshape(r, (width, height))
        g = np.reshape(g, (width, height))
        b = np.reshape(b, (width, height))

        # Concatenating the color images to a rgb image
        rgb = np.dstack((r, g, b))

        # 1 = normalize every cell between 0 and 255 (8 bit)
        if normalize_enum == 1:
            rgb = normalize_np(rgb, 0, bit_max)

        X.append(rgb)
        del r
        del g
        del b
        del rgb

    # 4 = normalize every cell individually with every color channel using the min / max of all three
    # 7 = z-score every cell individually with every color channel independent using all samples in the bag
    # 8 = z-score every cell individually with every color channel using the mean / std of all three from all samples in the bag
    if normalize_enum == 4 or normalize_enum == 7 or normalize_enum == 8:
        if normalize_enum == 7:
            bag_mean_r, bag_std_r = get_bag_mean(X, axis=0)
            bag_mean_g, bag_std_g = get_bag_mean(X, axis=1)
            bag_mean_b, bag_std_b = get_bag_mean(X, axis=2)
        if normalize_enum == 8:
            bag_mean, bag_std = get_bag_mean(X)

        for i in range(len(X)):
            current_x = X[i]
            current_r = current_x[:, :, 0]
            current_g = current_x[:, :, 1]
            current_b = current_x[:, :, 2]

            if normalize_enum == 4:
                current_r = normalize_np(current_r, best_well_min_r, best_well_max_r)
                current_g = normalize_np(current_g, best_well_min_g, best_well_max_g)
                current_b = normalize_np(current_b, best_well_min_b, best_well_max_b)
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

    X = np.asarray(X)
    return X, y


def get_bag_mean(n: [np.ndarray], axis: int = None):
    combined_x = np.zeros(0)
    for i in range(len(n)):
        current_x = n[i]
        dim_x = current_x.shape[0]
        dim_y = current_x.shape[1]

        if axis is None:
            current_r = current_x[:, :, 0].reshape(dim_x * dim_y)
            current_g = current_x[:, :, 1].reshape(dim_x * dim_y)
            current_b = current_x[:, :, 2].reshape(dim_x * dim_y)

            combined_x = np.append(combined_x, current_r)
            combined_x = np.append(combined_x, current_g)
            combined_x = np.append(combined_x, current_b)
            del current_r, current_g, current_b
        else:
            current_axis = current_x[:, :, axis].reshape(dim_x * dim_y)
            combined_x = np.append(combined_x, current_axis)
            del current_axis

    mean = np.mean(combined_x)
    std = np_std(n=combined_x, mean=mean)
    return mean, std[0]


def convert_bag_to_batch(bags, labels):
    ''' Convert bag and label pairs into batch format
    Inputs:
        a list of bags and a list of bag-labels

    Outputs:
        Returns a dataset (list) containing (stacked tiled instance data, bag label)
    '''
    dataset = []
    input_dim = None

    for index, (bag, bag_label) in enumerate(zip(bags, labels)):
        batch_data = np.asarray(bag, dtype='float32')
        batch_label = np.asarray(bag_label, dtype='float32')
        dataset.append((batch_data, batch_label))

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

    return np.sqrt(((n - mean) ** 2).mean(axis=axis, keepdims=True))


####

def repack_pags(X: [np.array], y: [int], repack_percentage: float = 0.2):
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

if __name__ == '__main__':
    print("This is a loader helper function that has no original main.")
