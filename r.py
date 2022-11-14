import math
import os
import socket
import numpy as np
import multiprocessing

from util import log

# local r test file inside this project
r_test_file = 'imports' + os.sep + 'r' + os.sep + 'sigmoid_evaluation.R'
assert os.path.exists(r_test_file)

# r test file on the prodi drive, available to unix devices
prodi_r_test_file = '/mil/sigmoid_evaluation.R'
prodi_r_test_file = r_test_file
if not os.name == 'nt':
    assert os.path.exists(prodi_r_test_file)

# Getting the whole path(s)
r_test_file = os.path.abspath(r_test_file)
prodi_r_test_file = os.path.abspath(prodi_r_test_file)


def pooled_sigmoid_evaluation(doses: [float], responses: [float], out_image_filename: str, global_bmc_30: bool = True,
                              testing_connection: bool = False, save_sigmoid_plot: bool = False, verbose: bool = False):
    assert len(doses) == len(responses)
    assert len(doses) > 1
    import pyRserve

    if testing_connection:
        save_sigmoid_plot = False
        # verbose = False

    # Setting up the out dir
    if save_sigmoid_plot:
        out_dir = os.path.dirname(out_image_filename)
        os.makedirs(out_dir, exist_ok=True)
        del out_dir

    # Establishing connection
    conn = pyRserve.connect()

    # Testing if we can receive pi from R correctly
    pi = conn.eval('pi')
    if verbose:
        log.write('According to Rserve, "PI" = ' + str(pi))

    doses_parsed = str(doses)[1:-1]
    responses_parsed = str(responses)[1:-1]
    dose_instruction = None
    resp_instruction = None

    # Testing the external file
    # Needing to sanitize paths before, just in case. Thanks windows.
    if os.name == 'nt':
        out_image_filename = out_image_filename.replace('\\', '\\\\')
        r_test_file_local = r_test_file.replace('\\', '\\\\')
    else:
        r_test_file_local = prodi_r_test_file

    if verbose:
        log.write('Executing file: ' + r_test_file_local)

    try:
        # Setting input variables

        # === EXAMPLE FORMAT ===
        # conn.eval('dose <- c(0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4,0.1,0.2,0.3,0.4)')
        # conn.eval('resp <- c(100,100,50,0,100,100,50,0,100,100,50,0)')

        dose_instruction = 'dose <- c(' + doses_parsed + ')'
        resp_instruction = 'resp <- c(' + responses_parsed + ')'
        log.write('dose instruction: ' + dose_instruction, print_to_console=False)
        log.write('resp instruction: ' + resp_instruction, print_to_console=False)

        conn.eval(dose_instruction)
        conn.eval(resp_instruction)
        if save_sigmoid_plot:
            conn.eval('plot_curve <- T')
            if verbose:
                log.write('Plotting R curve to: ' + out_image_filename)
        else:
            conn.eval('plot_curve <- F')
            if verbose:
                log.write('Not saving the R plot.')

        conn.eval('filename <- \'' + out_image_filename + '\'')

        # Running the R based code
        conn.eval('source("' + r_test_file_local + '")')

        # extracting results
        final_score = conn.eval('final_Score')
        fitted_plot = conn.eval('plot_data')
        estimate_plot = conn.eval('estimate_data')
        score_data = conn.eval('score_data')

        # Extracting sigmoid fitted curve
    except Exception as e:
        final_score = float('nan')
        fitted_plot = None
        estimate_plot = None
        score_data = None
        log.write(' == FATAL ERROR! ==')
        log.write('R failed to evaluate the sigmoid evaluation!')
        log.write(str(e))

    # Extracting estimate data
    if estimate_plot is not None:
        try:
            estimate_plot = np.asarray(estimate_plot, dtype=np.float64)
            estimate_plot = np.asarray([e[0] for e in estimate_plot])

            fitted_plot = np.asarray(fitted_plot, dtype=np.float64)
            # min(myList, key=lambda x: abs(x - myNumber))
            # fitted_plot[1][int(len(fitted_plot) / 2)]
        except Exception as e:
            estimate_plot = None
            log.write(' == FATAL ERROR! ==')
            log.write('Failed to extract the curve fitted values!')
            log.write(str(e))

    # Getting BMC30
    bmc_30 = float('NaN')
    if estimate_plot is not None:
        try:
            well_curve = fitted_plot[0]
            dose_curve = fitted_plot[1]

            if global_bmc_30:
                # When trying to find BMC30 at a 'global' scale, aka at 30% inhibitation
                bmc_point = 0.7
            else:
                # When trying to find BMC30 at a 'local' scale, based on normalized curve points
                bmc_point = (dose_curve.min() + dose_curve.max()) / 2

            mid_point_value = min(dose_curve, key=lambda x: abs(x - bmc_point))
            mid_point_index = np.where(dose_curve == mid_point_value)[0]
            bmc_30 = float(well_curve[mid_point_index])

            # Checking if the BMC is equal to min or max of the curve.
            # If so, that means, the BMC is actually NOT on the curve
            if not well_curve.min() < bmc_30 < well_curve.max():
                bmc_30 = float('NaN')
        except Exception as e:
            bmc_30 = float('NaN')
            log.write(' == Failed to estimate BMC30 ==')
            # TODO check on these errors later!
            log.write(str(e))
            log.write_exception(e)

    # Checking if final score is nan, so the plot data is set to None
    if math.isnan(final_score):
        estimate_plot = None
        fitted_plot = None
        score_data = None
    else:
        score_data = dict(zip(score_data.keys, score_data.values))

    if verbose:
        log.write('Sigmoid score: ' + str(final_score))

    try:
        conn.close()
    except Exception as e:
        # Failed to close connection. That's okay, script can continue running
        log.write('Warning: Failed to close R connection.')
        log.write(str(e))

    instructions = [dose_instruction, resp_instruction]
    return final_score, score_data, estimate_plot, fitted_plot, instructions, bmc_30


def prediction_sigmoid_evaluation(X_metadata, y_pred: [np.ndarray], out_dir: str,
                                  save_sigmoid_plot: bool = False, file_name_suffix: str = None, verbose: bool = False):
    if verbose:
        log.write('Running sigmoid prediction on predictions')
    os.makedirs(out_dir, exist_ok=True)
    assert len(X_metadata) == len(y_pred)

    # Remapping predictions so they can be evaluated
    experiment_prediction_map = {}
    for (X_metadata_current, y_pred_current) in zip(X_metadata, y_pred):
        y_pred_current: float = float(y_pred_current)
        metadata = X_metadata_current[0]

        experiment_name = metadata.experiment_name
        well_letter = metadata.well_letter
        well_number = metadata.well_number
        well = metadata.get_formatted_well()

        if experiment_name not in experiment_prediction_map.keys():
            experiment_prediction_map[experiment_name] = {}

        well_index_map = experiment_prediction_map[experiment_name]
        if well_number not in well_index_map.keys():
            well_index_map[well_number] = []

        # Adding the current prediction to the list
        well_index_map[well_number].append(y_pred_current)
        del X_metadata_current, y_pred_current
    del X_metadata, y_pred

    # Iterating over the experiment metadata, so we can run the sigmoid evaluations
    sigmoid_score_map = {}
    sigmoid_plot_estimation_map = {}
    sigmoid_instructions_map = {}
    sigmoid_fitted_plot_map = {}
    sigmoid_plot_score_detail_map = {}
    sigmoid_bmc30_map = {}

    for experiment_name in experiment_prediction_map.keys():
        well_index_map = experiment_prediction_map[experiment_name]

        doses = []
        responses = []
        for well_index in well_index_map.keys():
            index_responses = well_index_map[well_index]
            for response in index_responses:
                doses.append(well_index)
                responses.append(response)

        log.write(
            'Running sigmoid evaluation for ' + experiment_name + ' with ' + str(len(responses)) + ' total responses.')
        out_image_filename = out_dir + os.sep + experiment_name + '-sigmoid'
        if file_name_suffix is not None:
            out_image_filename = out_image_filename + file_name_suffix
        out_image_filename = out_image_filename + '.png'

        sigmoid_score, score_detail, estimate_plot, fitted_plot, instructions, bmc_30 = pooled_sigmoid_evaluation(
            doses=doses,
            responses=responses,
            verbose=verbose,
            save_sigmoid_plot=False,
            out_image_filename=out_image_filename)

        sigmoid_score_map[experiment_name] = sigmoid_score
        sigmoid_plot_estimation_map[experiment_name] = estimate_plot
        sigmoid_plot_score_detail_map[experiment_name] = score_detail
        sigmoid_fitted_plot_map[experiment_name] = fitted_plot
        sigmoid_instructions_map[experiment_name] = instructions
        sigmoid_bmc30_map[experiment_name] = bmc_30
        log.write('Sigmoid score for ' + experiment_name + ': ' + str(sigmoid_score) + '. BMC30: ' + str(bmc_30))

    return sigmoid_score_map, sigmoid_plot_score_detail_map, sigmoid_plot_estimation_map, sigmoid_fitted_plot_map, sigmoid_instructions_map, sigmoid_bmc30_map


def has_connection(also_test_script: bool = False, verbose: bool = False) -> bool:
    try:
        import pyRserve
        conn = pyRserve.connect()
        conn.close()

        if also_test_script:
            response = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]
            dose = [100, 100, 50, 0, 100, 100, 50, 0, 100, 100, 50, 0]
            pooled_sigmoid_evaluation(doses=dose, responses=response, out_image_filename='test.png',
                                      save_sigmoid_plot=False, testing_connection=True, verbose=verbose)
    except Exception as e:
        # Failed to close connection.
        log.write('Cannot connect to Rserve: ' + str(e))
        return False
    return True


def main():
    log.write('This function is used to evaluate predictions using R.')
    log.write('You are now on: ' + str(socket.gethostname()))
    log.write('Testing the connection:')
    connected = has_connection(also_test_script=True, verbose=True)

    log.write('Testing done. Test succeeded: ' + str(connected))


if __name__ == '__main__':
    main()
