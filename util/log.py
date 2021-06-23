import os
import sys

from util.utils import gct

_log_files = []


def add_file(filename: str):
    global _log_files
    if filename in _log_files:
        return

    _log_files.append(filename)


def remove_file(filename: str):
    global _log_files

    if filename in _log_files:
        _log_files.remove(filename)


def clear_files():
    global _log_files
    _log_files.clear()


def set_file(filename: str):
    clear_files()
    add_file(filename=filename)


def write(output, print_to_console: bool = True, include_timestamp: bool = True):
    global _log_files
    output = str(output)

    if include_timestamp:
        timestamp = gct()
        output = '[' + timestamp + '] ' + output

    if print_to_console:
        print(output)

    for current_out_file in _log_files:
        if os.path.exists(current_out_file):
            f = open(current_out_file, 'a')
            f.write('\n')
        else:
            f = open(current_out_file, 'w')

        f.write(output)
        f.close()


def main():
    global _log_files
    write('Testing the "log" functions.')

    if len(_log_files) == 0:
        add_file('test_log.txt')
    write('This is a test log.')


if __name__ == "__main__":
    main()
