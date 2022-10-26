import os
from pathlib import Path
from util.utils import gct
import torch

# Just for pytorch: Setting the print-options so pytorch tensors are displayed in more detail
torch.set_printoptions(sci_mode=False, precision=8)

global _log_files
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


def write(output: str, print_to_console: bool = True, include_timestamp: bool = True, include_in_files: bool = True):
    try:
        if output is None:
            output = '<none>'

        output = str(output)
        _write(output=output, print_to_console=print_to_console, include_timestamp=include_timestamp,
               include_in_files=include_in_files)
    except Exception as e:
        print('Failed to log: "' + str(output).strip() + '"!')
        print(str(e))
        pass
        # TODO: Better log error


def _write(output, print_to_console: bool = True, include_timestamp: bool = True, include_in_files: bool = True):
    global _log_files
    output = str(output)

    if include_timestamp:
        timestamp = gct()
        output = '[' + timestamp + '] ' + output

    if print_to_console:
        print(output)

    if include_in_files:
        for current_out_file in _log_files:
            try:
                if os.path.exists(current_out_file):
                    f = open(current_out_file, 'a')
                    f.write('\n')
                else:
                    # Creating the parent path
                    parent_path = Path(current_out_file)
                    parent_path = parent_path.parent.absolute()
                    os.makedirs(parent_path, exist_ok=True)
                    f = open(current_out_file, 'w')

                f.write(output)
                f.close()
            except Exception as e:
                print('Failed to log to: ' + str(current_out_file))
                print(str(e))
                # TODO: Better log error


def diagnose():
    global _log_files
    # Diagnosing log files
    write('Diagnosing log files.')
    write('Number of log files: ' + str(len(_log_files)))
    for file in _log_files:
        write('Logging to: ' + str(file))


def main():
    global _log_files
    write('Testing the "log" functions.')

    if len(_log_files) == 0:
        add_file('test_log.txt')
    write('This is a test log.')
    diagnose()
    clear_files()


if __name__ == "__main__":
    main()
