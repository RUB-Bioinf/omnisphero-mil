import math

import numpy as np
from scipy.optimize import curve_fit

from imports.frechet import frechet_dist
from util import log


def curve_fit_ideal(y_data_count: int, approximation_points_mult: int = 100):
    x_data = np.linspace(0, y_data_count, (y_data_count + 1) * approximation_points_mult)
    y_data = np.linspace(1, 0, (y_data_count + 1) * 100)
    p_opt, p_cov = sigmoid_curve_fit(x_data, y_data)

    y = sigmoid(x_data, *p_opt)
    return y, x_data


def curve_fit_prediction(prediction_dict: dict, approximation_points_mult: int = 100) -> (np.ndarray, np.ndarray):
    """ Maps a given prediction to a sigmoid curve.
    Returns the x and y position of the curve as lists to be plotted.
    """
    # Preparing x & y data
    x_data = np.array(list(range(len(prediction_dict))))
    y_data = [np.mean(prediction_dict[p]) for p in prediction_dict]

    # Getting the Fit
    try:
        p_opt, p_cov = sigmoid_curve_fit(x_data, y_data)
    except Exception as e:
        log.write('Curve fit failed!')
        log.write(str(e))
        return None, None

    # Getting the x & y fitting coordinates
    x = np.linspace(0, len(prediction_dict) - 1, len(prediction_dict) * approximation_points_mult)
    y = sigmoid(x, *p_opt)

    return y, x


def curve_fit_prediction_accuracy(prediction_dict: dict, approximation_points_mult: int = 100) -> ([float], float):
    """ Maps a given prediction to a sigmoid curve.
    The result is then compared to the ideal sigmoid curve for the number of data-points provided.
    """
    y1, x1 = curve_fit_prediction(prediction_dict=prediction_dict, approximation_points_mult=approximation_points_mult)
    y2, x2 = curve_fit_ideal(len(prediction_dict) - 1)

    if x1 is None or x2 is None:
        return [math.nan for i in range(len(y2))], math.nan

    d = compare_curve_distances(y1=y1, y2=y2)
    p = np.array([list(y1), list(range(len(y1)))])
    q = np.array([list(y2), list(range(len(y2)))])

    frechet = frechet_dist(p, q)
    return d, frechet


def compare_curve_distances(y1: [float], y2: [float]) -> [float]:
    assert len(y1) == len(y2)

    d = []
    for i in range(len(y1)):
        d.append(_cmp(y1[i], y2[i]))

    return d


def _cmp(a: float, b: float) -> float:
    if a == b:
        return 0

    x = min(a, b)
    y = max(a, b)
    del a, b
    return math.sqrt(math.pow((y - x), 2))


def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k * (x - x0)))
    return y


def sigmoid_curve_fit(x_data: np.array, y_data: np.array):
    """ Internal function. Remaps two given arrays to a sigmoid function.
    """
    return curve_fit(sigmoid, x_data, y_data)
