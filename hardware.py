import numpy as np
import torch

from util import log


def get_hardware_device(gpu_preferred: bool = True):
    ''' Pick GPU if available, else run on CPU.
    Returns the corresponding device.
    '''
    if gpu_preferred:
        init()
        print_gpu_status()

        if torch.cuda.is_available():
            print('Running on GPU.')
            return torch.device('cuda')
        else:
            print('  =================')
            print('Wanted to run on GPU but it is not available!!')
            print('  =================')

    log.write('Running on CPU.')
    return torch.device('cpu')


def print_gpu_status():
    print('Torch: is Cuda available: ' + str(torch.cuda.is_available()))
    print('Torch: Visible Devices: ' + str(torch.cuda.device_count()))
    print('Torch: Compiled Torch Architecture: ' + str(torch.cuda.get_arch_list()))


def init():
    torch.cuda.init()


def main():
    print('Running GPU Hardware diagnostics...')
    init()
    print_gpu_status()


if __name__ == '__main__':
    main()
