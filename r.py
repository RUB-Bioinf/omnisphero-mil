import os
import numpy as np
from util import log

# local r test file inside this project
from util.well_metadata import TileMetadata

r_test_file = 'imports' + os.sep + 'r' + os.sep + 'sigmoid_evaluation.R'
assert os.path.exists(r_test_file)

# r test file on the prodi drive, available to unix devices
prodi_r_test_file = '/mil/sigmoid_evaluation.R'
if not os.name == 'nt':
    assert os.path.exists(prodi_r_test_file)

# Getting the whole path(s)
r_test_file = os.path.abspath(r_test_file)
prodi_r_test_file = os.path.abspath(prodi_r_test_file)


def pooled_sigmoid_evaluation(doses: [float], responses: [float], out_image_filename: str,
                              save_sigmoid_plot: bool = True, verbose: bool = False):
    assert len(doses) == len(responses)
    assert len(doses) > 1
    import pyRserve

    out_dir = os.path.dirname(out_image_filename)
    os.makedirs(out_dir, exist_ok=True)
    conn = pyRserve.connect()

    # Testing if we can receive pi from R correctly
    pi = conn.eval('pi')
    if verbose:
        log.write('According to Rserve, "PI" = ' + str(pi))

    doses_parsed = str(doses)[1:-1]
    responses_parsed = str(responses)[1:-1]

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

        conn.eval('dose <- c(' + doses_parsed + ')')
        conn.eval('resp <- c(' + responses_parsed + ')')
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
        final_score = conn.eval('final_Score ')
    except Exception as e:
        final_score = float('nan')
        log.write(' == FATAL ERROR! ==')
        log.write('R failed to evaluate the sigmoid evaluation!')
        log.write(str(e))

    if verbose:
        log.write('Sigmoid score: ' + str(final_score))

    try:
        conn.close()
    except Exception as e:
        # Failed to close connection. That's okay, script can continue running
        log.write('Warning: Failed to close R connection.')
        log.write(str(e))

    return final_score


def prediction_sigmoid_evaluation(X_metadata: [TileMetadata], y_pred: [np.ndarray], out_dir: str,
                                  save_sigmoid_plot: bool = False,
                                  file_name_suffix: str = None, verbose: bool = False):
    if verbose:
        log.write('Running sigmoid prediction on predictions')
    os.makedirs(out_dir, exist_ok=True)
    assert len(X_metadata) == len(y_pred)

    # Remapping predictions so they can be evaluated
    experiment_prediction_map = {}
    for (X_metadata_current, y_pred_current) in zip(X_metadata, y_pred):
        y_pred_current: float = float(y_pred_current)
        metadata: TileMetadata = X_metadata_current[0]

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

    # Iterating over the experiment metadata so we can run the sigmoid evaluations
    sigmoid_score_map = {}
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

        sigmoid_score = pooled_sigmoid_evaluation(doses=doses, responses=responses, verbose=verbose,
                                                  save_sigmoid_plot=save_sigmoid_plot,
                                                  out_image_filename=out_image_filename)
        sigmoid_score_map[experiment_name] = sigmoid_score
        log.write('Sigmoid score for ' + experiment_name + ': ' + str(sigmoid_score))

    return sigmoid_score_map


def has_connection() -> bool:
    try:
        import pyRserve
        conn = pyRserve.connect()
        conn.close()
    except Exception as e:
        # Failed to close connection.
        log.write('Cannot connect to Rserve: ' + str(e))
        return False
    return True


def main():
    print('This function is used to evaluate predictions using R.')


if __name__ == '__main__':
    main()
