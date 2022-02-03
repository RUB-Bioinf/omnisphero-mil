import math
import os
import random
from datetime import datetime
from sys import platform
import sys
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import numpy as np

# ###############################
# OTHER UTIL FUNCTIONS
# ###############################
from util import log


def gct(raw: bool = False) -> [str, datetime]:
    """
    Gets the current time as a formated string or datetime object.
    Shortcut function.

    Created by Nils Förster.

    :param raw: An optional parameter. If True, the time will be returned as a datetime object. Default: False.

    :type raw: bool

    :returns: The current time. Either as a formated string or datetime object.
    :rtype: datetime,str

    .. note:: Required packages: datetime
    """
    n = datetime.now()
    if raw:
        return n
    return n.strftime("%d/%m/%Y %H:%M:%S")


def get_time_diff(start_time: datetime) -> str:
    """
    Calculates the time difference from a given datetime object and the current time this function is being called.

    Created by Nils Förster.

    :param start_time: The timestamp to calculate the difference from

    :type start_time: datetime

    :returns: The time difference as a formated string.
    :rtype: str

    .. note:: Required packages: datetime
    """

    diff = datetime.now() - start_time
    minutes = divmod(diff.total_seconds(), 60)

    m: str = str(int(minutes[0]))
    s: str = str(int(minutes[1]))
    if minutes[1] < 10:
        s = '0' + s
    return m + ':' + s


# ###############################
# API TO LaTeX TIKZ
# ###############################

def create_tikz_axis(title: str, label_y: str, label_x: str = 'Epoch', max_x: float = 1.0, min_x: float = 0.0,
                     max_y: float = 1.0, min_y: float = 0.0, tick_count: int = 10,
                     legend_pos: str = 'north west') -> str:
    """
    Sets up a basic tikz plot environment to be used in a LaTeX document.
    This is a helper function; For the true function try #get_plt_as_tex.
    That function fills the tikz plot with graphs.

    Created by Nils Förster.

    :param title: The title to be used in the plot.
    :param label_y: The label for the y axis.
    :param label_x: Optional argument. The label for the x axis. Default: 'Epoch'.
    :param max_x: Optional argument. The maximum span for the x axis. Default: 1.0
    :param min_x: Optional argument. The minimum span for the x axis. Default: 0.0
    :param max_y: Optional argument. The maximum span for the y axis. Default: 1.0
    :param min_y: Optional argument. The maximum span for the y axis. Default: 0.0
    :param tick_count: Optional argument. In how many 'ticks' should the plot be partitioned? Default: 10.
    :param legend_pos: Optional argument. The position of the legend. Default: 'north-west'.

    :type title: str
    :type label_y: str
    :type label_x: str
    :type max_x: float
    :type min_x: float
    :type max_y: float
    :type min_y: float
    :type tick_count: int
    :type legend_pos: str

    :returns: A template for a tikz plot as a string.
    :rtype: str
    """

    max_x = float(max_x)
    max_y = float(max_y)
    tick_count = float(tick_count)

    tick_x = max_x / tick_count
    tick_y = max_y / tick_count
    if min_x + max_x > 10:
        tick_x = int(tick_x)
    if min_y + max_y > 10:
        tick_y = int(tick_y)

    axis_text: str = '\\begin{center}\n\t\\begin{tikzpicture}\n\t\\begin{axis}[title={' + title + '},xlabel={' + label_x + '},ylabel={' + label_y + '},xtick distance=' + str(
        tick_x) + ',ytick distance=' + str(tick_y) + ',xmin=' + str(min_x) + ',xmax=' + str(
        max_x) + ',ymin=' + str(min_y) + ',ymax=' + str(
        max_y) + ',major grid style={line width=.2pt,draw=gray!50},grid=both,height=8cm,width=8cm'
    if legend_pos is not None:
        axis_text = axis_text + ', legend pos=' + legend_pos
    axis_text = axis_text + ']'
    return axis_text


def get_plt_as_tex(data_list_y: [[float]], plot_colors: [str], title: str, label_y: str, data_list_x: [[float]] = None,
                   plot_titles: [str] = None, label_x: str = 'Epoch', max_x: float = 1.0, min_x: float = 0.0,
                   max_y: float = 1.0,
                   min_y: float = 0.0, max_entries: int = 4000, tick_count: int = 10, legend_pos: str = 'north west'):
    """
    Formats a list of given plots in a single tikz axis to be compiled in LaTeX.
    
    This function respects the limits of the tikz compiler.
    That compiler can only use a limited amount of virtual memory that is (to my knowledge not changeable).
    Hence this function can limit can limit the line numbers for the LaTeX document.
    Read this function's parameters for more info.

    This function is designed to be used in tandem with the python library matplotlib.
    You can plot multiple graphs in a single axis by providing them in a list.
    Make sure that the lengths of the lists (see parameters below) for the y and x coordinates, colors and legend labels match.

    Created by Nils Förster.

    :param data_list_y: A list of plots to be put in the axis. Each entry in this list should be a list of floats with the y position of every node in the graph.
    :param plot_colors: A list of strings descibing colors for every plot. Make sure len(data_list_y) matches len(plot_colors).
    :param title: The title to be used in the plot.
    :param label_y: The label for the y axis.
    :param plot_titles: Optional argument. A list of strings containing the legend entries for every graph. If None, no entries are written in the legend. Make sure len(data_list_y) matches len(plot_titles).
    :param data_list_x: Optional argument. A list of plots to be put in the axis. Each entry in this list should be a list of floats with the x position of every node in the graph. If this argument is None, the entries in the argument data_list_y are plotted as nodes in sequential order.
    :param label_x: Optional argument. The label for the x axis. Default: 'Epoch'.
    :param max_x: Optional argument. The maximum span for the x axis. Default: 1.0
    :param min_x: Optional argument. The minimum span for the x axis. Default: 0.0
    :param max_y: Optional argument. The maximum span for the y axis. Default: 1.0
    :param min_y: Optional argument. The maximum span for the y axis. Default: 0.0
    :param tick_count: Optional argument. In how many 'ticks' should the plot be partitioned? Default: 10.
    :param legend_pos: Optional argument. The position of the legend. Default: 'north-west'.
    :param max_entries: Limits the amount of nodes for the plot to this number. This does not cut of the data, but increases scanning offsets. Use a smaller number for faster compile times in LaTeX. Default: 4000.

    :type data_list_y: [[float]]
    :type plot_colors: [str]
    :type title: str
    :type label_y: str
    :type plot_titles: [str]
    :type label_x: str
    :type max_x: float
    :type min_x: float
    :type max_y: float
    :type min_y: float
    :type tick_count: int
    :type legend_pos: str
    :type max_entries: int

    :returns: A fully formated string containing a tikz plot with multiple graphs and and legends. Save this to your device and compile in LaTeX to render your plot.
    :rtype: str

    Examples
    ----------
    Use this example to plot a graph in matplotlib (as plt) as well as tikz:

    >>> history = model.fit(...)
    >>> loss = history.history['loss']
    >>> plt.plot(history_all.history[hist_key]) # plotting the loss using matplotlib
    >>> f = open('example.tex')
    >>> tex = get_plt_as_tex(data_list_y=[loss], title='Example Plot', label_y='Loss', label_x='Epoch', plot_colors=['blue']) # plotting the same data as a tikz axis
    >>> f.write(tex)
    >>> f.close()

    When you want to plot multiple graphs into a single axis, expand the example above like this:

    >>> get_plt_as_tex(data_list_y=[loss, val_loss], title='Example Plot', label_y='Loss', label_x='Epoch', plot_colors=['blue'], plot_titles=['Loss','Validation Loss'])

    When trying to render the tikz plot, make sure to import these LaTeX packages:

    >>> \\usepackage{tikz,amsmath, amssymb,bm,color,pgfplots}

    .. note:: Some manual adjustments may be required to the tikz axis. Try using a wysisyg tikz / LaTeX editor for that. For export use, read the whole tikz user manual ;)
    """

    out_text = create_tikz_axis(title=title, label_y=label_y, label_x=label_x, max_x=max_x, min_x=min_x, max_y=max_y,
                                min_y=min_y, tick_count=tick_count, legend_pos=legend_pos) + '\n'
    line_count = len(data_list_y[0])
    data_x = None
    steps = int(max(len(data_list_y) / max_entries, 1))

    for j in range(0, len(data_list_y), steps):
        data_y = data_list_y[j]
        if data_list_x is not None:
            data_x = data_list_x[j]

        color = plot_colors[j]

        out_text = out_text + '\t\t\\addplot[color=' + color + '] coordinates {' + '\n'
        for i in range(line_count):
            y = data_y[i]

            x = i + 1
            if data_x is not None:
                x = data_x[i]

            out_text = out_text + '\t\t\t(' + str(x) + ',' + str(y) + ')\n'
        out_text = out_text + '\t\t};\n'

        if plot_titles is not None:
            plot_title = plot_titles[j]
            out_text = out_text + '\t\t\\addlegendentry{' + plot_title + '}\n'

    out_text = out_text + '\t\\end{axis}\n\t\\end{tikzpicture}\n\\end{center}'
    return out_text


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def line_print(text: str, max_width: int = None, cutoff_too_large_text: bool = True, use_new_line: bool = True,
               include_in_log: bool = True):
    text = str(text)

    if max_width is None:
        if platform == "linux" or platform == "linux2":
            try:
                terminal_rows, terminal_columns = os.popen('stty size', 'r').read().split()
                terminal_rows = int(terminal_rows)
                terminal_columns = int(terminal_columns)
                # print('Did you know your terminal is ' + str(terminal_columns) + 'x' + str(terminal_rows) + ' characters big.')
            except Exception as e:
                print(text)
                return
        else:
            max_width = 50
            terminal_columns = 50
            cutoff_too_large_text = False
            use_new_line = False
    else:
        terminal_columns = max_width

    out_s = ''
    if cutoff_too_large_text:  # and len(text) < terminal_columns:
        for i in range(terminal_columns - 1):
            if i < len(text):
                out_s = out_s + text[i]
            else:
                out_s = out_s + ' '
        # out_s = out_s + ' ' + str(len(out_s))
    else:
        out_s = str(text)

    if include_in_log:
        log.write(out_s, print_to_console=False)

    if use_new_line:
        print(out_s, end="\r")
    else:
        print(out_s)


###########################
# PROJECT UTILS
###########################

# shuffle data and split into training and validation set
def shuffle_and_split_data(dataset, split_percentage: float, _recursion_depth: int = 0):
    '''
    Takes a dataset that was converted from bags to batches and shuffles and splits it into two splits (train/val)
    '''
    split_percentage_index = math.ceil(split_percentage * len(dataset))
    indices = np.arange(len(dataset))
    random.shuffle(indices)
    test_ind, train_ind = np.asarray(indices[:split_percentage_index]), np.asarray(indices[split_percentage_index:])

    training_ds = [dataset[i] for i in train_ind]
    validation_ds = [dataset[j] for j in test_ind]

    num_positive = sum([int(dataset[i][1]) for i in range(len(dataset))])
    num_negative = len(dataset) - num_positive
    if not (num_positive == 0 or num_negative == 0) and (
            num_positive >= 2 and num_negative >= 2 and split_percentage_index >= 2) and (
            _recursion_depth < sys.getrecursionlimit() - 10):
        # This if branch tests if there are no positive or negative bags.
        # So if there are none, that means there are not enough bags to suffle and end up with at least one pos / negative bag after the shuffle

        # Also checks if there are at least two pos and negative bags to fairly distribute them in the first place!

        training_num_positive = sum([int(training_ds[i][1]) for i in range(len(training_ds))])
        training_num_negative = len(training_ds) - training_num_positive

        validation_num_positive = sum([int(validation_ds[i][1]) for i in range(len(validation_ds))])
        validation_num_negative = len(validation_ds) - validation_num_positive

        if training_num_positive == 0 or training_num_negative == 0 or validation_num_positive == 0 or validation_num_negative == 0:
            # At this point, there is at least one dataset without positive or negative bags
            log.write(
                '[' + str(_recursion_depth) + '/' + str(sys.getrecursionlimit()) + '] Data shuffle resulted in inequal '
                                                                                   'distribution. Trying again.')

            del training_ds, validation_ds
            return shuffle_and_split_data(dataset, split_percentage, _recursion_depth + 1)

    del dataset
    return training_ds, validation_ds


def lecture_shannon_entropy(n: np.ndarray) -> float:
    # Only works for 1d arrays
    assert len(n.shape) == 1
    counts, values = sparse_hist(n)

    # entropy = 0.0
    # for i in range(n.min(), n.max()):
    #     if i in values:
    #         j = values.index(i)
    #         p = counts[j] / sum(counts)
    #         entropy = entropy + p * math.log(p, 2)

    entropy = 0.0
    for v in values:
        j = values.index(v)
        p = counts[j] / sum(counts)
        entropy = entropy + p * math.log(p, 2)

    entropy = entropy * -1
    return entropy


def lecture_otsu(n: np.ndarray) -> int:
    if len(n.shape) == 1:
        pixel_number = n.shape[0]
    else:
        pixel_number = n.shape[0] * n.shape[1]
    mean_weight = 1.0 / pixel_number

    n = n.astype(np.float64)
    p = n / max(n)

    indices = list(range(len(n)))
    best_std = float("inf")
    best_k = float("nan")
    for k in indices[1:-1]:
        weight_0 = sum(p[indices[:k]])
        mean_0 = (1 / weight_0) * sum([i * p[i] for i in indices[0:k]])
        variance_0 = (1 / weight_0) * sum([(i - mean_0) ** 2 * p[i] for i in indices[0:k]])

        weight_1 = sum(p[indices[k + 1:]])
        mean_1 = (1 / weight_1) * sum([i * p[i] for i in indices[k + 1:]])
        variance_1 = (1 / weight_1) * sum([(i - mean_1) ** 2 * p[i] for i in indices[k + 1:]])

        weighted_std = weight_0 * variance_0 + weight_1 * variance_1
        if weighted_std < best_std:
            best_std = weighted_std
            best_k = k

    return best_k


def sparse_hist(a: np.ndarray) -> ([np.float32], [int]):
    x = np.copy(a)
    x.astype(np.float64)
    x.sort()

    h = {}
    for i in range(len(x)):
        current_value = x[i]

        if current_value not in h.keys():
            h[current_value] = 0

        current_count = h[current_value]
        h[current_value] = current_count + 1

        del current_value, current_count

    plt.clf()
    bins = list(h.keys())
    n = list(h.values())
    return n, bins


def otsu_8bit_image(gray):
    if len(gray.shape) == 1:
        pixel_number = gray.shape[0]
    else:
        pixel_number = gray.shape[0] * gray.shape[1]
    mean_weight = 1.0 / pixel_number
    his, bins = np.histogram(gray, bins=np.arange(0, 257))
    final_thresh = -1
    final_value = -1
    intensity_arr = np.arange(256)
    for t in bins[1:-1]:
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])
        Wb = pcb * mean_weight
        Wf = pcf * mean_weight

        mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
        # print mub, muf
        value = Wb * Wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value
    return final_thresh


if __name__ == "__main__":
    print('There are some util functions for everyone to use within this file. Enjoy. :)')
