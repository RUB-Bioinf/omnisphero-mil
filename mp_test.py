from multiprocessing import Process
from time import sleep
from random import random
from time import sleep
from multiprocessing import Value
from multiprocessing import Process
from util import log
from multiprocessing import Pool
import os
import loader


def mp_test():
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def f(name):
    info('function f')
    print('hello', name)


# function to execute in a child process
def task(variable):
    # generate some data
    data = random()
    # block, to simulate computational effort
    print(f'Generated {data}', flush=True)
    sleep(data)
    # return data via value
    variable.value = data


# protect the entry point
if __name__ == '__main__':
    # create shared variable
    variable = Value('f', 0.0)
    # create a child process process
    process = Process(target=task, args=(variable,))
    # start the process
    process.start()
    # wait for the process to finish
    process.join()
    # report return value
    print(f'Returned: {variable.value}')
