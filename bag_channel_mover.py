import zipfile
from zipfile import ZipFile
import sys
import os
import time
from concurrent.futures.thread import ThreadPoolExecutor

from util import log
from util import paths
from util.utils import gct
from util.utils import get_time_diff
from util.utils import line_print


def channel_mover(input_dirs: [str], output_dir: str, move_pattern: str = 'rgb', max_workers: int = 4):
    assert len(move_pattern) == 3
    os.makedirs(output_dir, exist_ok=True)

    log.write('Input dirs:')
    log.write(input_dirs, include_timestamp=False)
    log.write('Output dir: ' + output_dir)

    replace_r = str(move_pattern[0])
    replace_g = str(move_pattern[1])
    replace_b = str(move_pattern[2])
    assert replace_r in ['r', 'g', 'b']
    assert replace_g in ['r', 'g', 'b']
    assert replace_b in ['r', 'g', 'b']

    time.sleep(1)
    log.write('Replacing color chanel: "r" -> ' + replace_r)
    log.write('Replacing color chanel: "g" -> ' + replace_g)
    log.write('Replacing color chanel: "b" -> ' + replace_b)
    time.sleep(1)

    for i in range(len(input_dirs)):
        log.write('\n\n\t===== [PROCESSING ' + str(i + 1) + '/' + str(len(input_dirs)) + '] =====',
                  include_timestamp=False)
        current_input_dir = input_dirs[i]
        log.write(current_input_dir)
        dir_name = os.path.basename(current_input_dir)
        log.write('Directory: ' + dir_name)
        time.sleep(1)

        executor = ThreadPoolExecutor(max_workers=max_workers)
        future_list = []
        files = os.listdir(current_input_dir)

        for file in files:
            file = str(file)
            file_path = current_input_dir + os.sep + file

            if file.endswith('.json'):
                future = executor.submit(process_file,
                                         file_path,
                                         replace_r,
                                         replace_g,
                                         replace_b,
                                         dir_name,
                                         output_dir
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

            line_print('[' + str(i + 1) + ' / ' + str(len(input_dirs)) + '] ' + str(
                max_workers) + ' Threads running. Finished: ' + str(finished_count) + '/' + str(
                len(future_list)) + '. Errors: ' + str(
                error_count) + '. Running: ' + get_time_diff(
                start_time) + '. ' + gct(), include_in_log=False)
            all_finished = finished_count == len(future_list)
            time.sleep(1)

        log.write('All done for: ' + dir_name)
    log.write('Finished all conversions.')


def process_file(file_path,
                 replace_r,
                 replace_g,
                 replace_b,
                 dir_name, output_dir):
    output_dir = output_dir + os.sep + dir_name
    os.makedirs(output_dir, exist_ok=True)

    data = ''
    with open(file_path, 'r') as file:
        data = file.read()

    # replacing the channels with temp values so nothing will be cross-replaced
    data = data.replace('"r"', '"temp_replace_r"')
    data = data.replace('"g"', '"temp_replace_g"')
    data = data.replace('"b"', '"temp_replace_b"')

    # now replacing the temp values with the actual values
    data = data.replace('"temp_replace_r"', '"' + replace_r + '"')
    data = data.replace('"temp_replace_g"', '"' + replace_g + '"')
    data = data.replace('"temp_replace_b"', '"' + replace_b + '"')

    # writing .json file
    input_file_name = os.path.basename(file_path)
    output_file_name = output_dir + os.sep + input_file_name
    output_file_name_zip = output_file_name + '.zip'
    f = open(output_file_name, "w")
    f.write(data)
    f.close()

    # zipping it
    zipf = ZipFile(output_file_name_zip, "w", )
    zipf.write(filename=output_file_name, arcname=input_file_name, compress_type=zipfile.ZIP_DEFLATED)
    zipf.close()


def main():
    if sys.platform == 'win32':
        current_global_log_dir = 'U:\\bioinfdata\\work\\OmniSphero\\Sciebo\\HCA\\00_Logs\\mil_log\\win\\'
        input_dir = paths.all_prediction_dirs_win
        max_workers = 4
    else:
        current_global_log_dir = '/Sciebo/HCA/00_Logs/mil_log/linux/'
        input_dir = paths.all_prediction_dirs_unix
        max_workers = 25

    if current_global_log_dir is not None:
        global_log_filename = current_global_log_dir + os.sep + 'log-channel_mover.txt'
        os.makedirs(current_global_log_dir, exist_ok=True)
        log.add_file(global_log_filename)
    log.diagnose()

    # Running the mover now
    move_pattern = 'rbg'
    parent_dir = os.path.abspath(os.path.join(input_dir[0], os.pardir))
    output_dir = parent_dir + os.sep + 'channel-transformed-' + move_pattern

    channel_mover(input_dirs=input_dir, output_dir=output_dir, max_workers=max_workers, move_pattern=move_pattern)


if __name__ == '__main__':
    main()
