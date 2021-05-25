# IMPORTS

import os
import time
from zipfile import ZipFile
import json
import numpy as np
from sys import platform

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from lib.utils import gct
from lib.utils import get_time_diff
from lib.utils import line_print


####

def load_bags_json_batch(batch_dirs: [str], max_workers: int):
    print('Looking for multiple source dirs to load json data from.')
    print('Batch dir: ', batch_dirs)

    X_full = None
    y_full = None
    error_list = []

    for i in range(len(batch_dirs)):
        current_dir = batch_dirs[i]
        print('Considering source directory: ' + current_dir)

        if os.path.isdir(current_dir):
            X, y, errors = load_bags_json(source_dir=current_dir, max_workers=max_workers, gp_current=i,
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

    return X_full, y_full, error_list


# Main Loading function
def load_bags_json(source_dir: str, max_workers: int, gp_current: int = 1, gp_max: int = 1):
    files = os.listdir(source_dir)
    print('Loading from source: ' + source_dir)

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
        print('Considering source file: ' + file)
        filepath = source_dir + os.sep + file

        X_j = []
        y_j = []

        if file.endswith('.json.zip'):
            future = executor.submit(unzip_and_read_JSON,
                                     filepath,
                                     worker_verbose
                                     )
            # [X_j, y_j] = unzip_and_read_JSON(filepath,worker_verbose)
            # X.extend(X_j)
            # y.extend(y_j)
            future_list.append(future)

        if file.endswith('.json'):
            if os.path.exists(file + '.zip'):
                print('A zipped version also existed. Skipping.')
                continue

            future = executor.submit(read_JSON_file,
                                     filepath,
                                     worker_verbose
                                     )
            future_list.append(future)
            # [X_j, y_j] = read_JSON_file(filepath,worker_verbose)
            # X.extend(X_j)
            # y.extend(y_j)

    start_time = gct(raw=True)
    all_finished: bool = False
    executor.shutdown(wait=False)

    while not all_finished:
        finished_count = 0
        predicted_count = 0
        error_count = 0

        for future in future_list:
            if future.done():
                finished_count = finished_count + 1

                e = future.exception()
                if e is not None:
                    error_count = error_count + 1

        line_print('[' + str(gp_current) + ' / ' + str(gp_max) + '] ' + str(
            max_workers) + ' Threads running. Finished: ' + str(finished_count) + '/' + str(
            len(future_list)) + '. Already predicted: ' + str(predicted_count) + '. Errors: ' + str(
            error_count) + '. Running: ' + get_time_diff(
            start_time) + '. ' + gct(), max_width=terminal_columns)
        all_finished = finished_count == len(future_list)
        time.sleep(1)

    X = []
    y = []
    error_list = []

    for i in range(len(future_list)):
        future = future_list[i]
        line_print('Extracting future: ' + str(i) + '/' + str(len(future_list)))

        e = future.exception()
        if e is None:
            X_f, y_f = future.result()
            if X_f is not None:
                X.extend(X_f)
            if y_f is not None:
                y.extend(y_f)
        else:
            print('\n' + gct() + 'Error extracting future results: ' + str(e) + '\n')
            error_list.append(e)

    print(gct() + ' Fully Finished Loading Path.')

    # Deleting the futures and the future list to immediately releasing the memory.
    del future_list[:]
    del future_list

    X = np.asarray(X)
    y = np.asarray(y)
    return X, y, error_list


####


def unzip_and_read_JSON(filepath, worker_verbose):
    if worker_verbose:
        print('Unzipping and reading json: ' + filepath)

    # handling the case, if a json file has been zipped
    # The idea: Read the zip, unzip it in ram and parse the byte stream directly as a string!
    input_zip = ZipFile(filepath)
    zipped_data_name = input_zip.namelist()[0]
    data = input_zip.read(zipped_data_name)
    input_zip.close()

    data = json.loads(data)
    X, y = parse_JSON(data, worker_verbose)

    if worker_verbose:
        print('File Shape: ' + filepath + ' -> ')
        print("X-shape: " + str(np.asarray(X).shape))
        print("y-shape: " + str(np.asarray(y).shape))

    return X, y


####

def read_JSON_file(filepath, worker_verbose):
    if worker_verbose:
        print('Reading json: ' + filepath)

    X = []
    y = []

    f = open(filepath)
    data = json.load(f)
    f.close()

    return parse_JSON(data, worker_verbose)


####

def parse_JSON(json_data, worker_verbose):
    # Setting up arrays
    X = []
    y = None

    # Reading meta data
    width = json_data['tileWidth']
    height = json_data['tileHeight']
    bit_depth = json_data['bit_depth']

    # Reading label, if it exists
    if 'label' in json_data:
        label = json_data['label']
        label = int(label)
        y = [label]

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

        r = np.reshape(r, (width, height))
        g = np.reshape(g, (width, height))
        b = np.reshape(b, (width, height))

        rgb = np.dstack((r, g, b))
        X.append(rgb)
        del r
        del g
        del b

    return X, y


####

if __name__ == '__main__':
    print("This is a loader helper function that has no original main.")
