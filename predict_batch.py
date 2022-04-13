import os
import shutil
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import hardware
import loader
import mil_metrics
import models
import r
from util import data_renderer
from util import log
from util import paths
from util import utils
from util.omnisphero_data_loader import OmniSpheroDataLoader
from util.paths import curated_overlapping_source_dirs_unix
from util.paths import debug_prediction_dirs_win
from util.paths import default_out_dir_unix_base
from util.utils import line_print
from util.well_metadata import TileMetadata
from util.well_metadata import extract_well_info

model_debug_path = "U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\production\\hnm-early_inverted-O1-adadelta-NoNeuron2-wells-normalize-6repack-0.55-BEST"


def predict_path(model_save_path: str, checkpoint_file: str, bag_paths: [str], normalize_enum: int, out_dir: str,
                 max_workers: int, image_folder: str, channel_inclusions=loader.default_channel_inclusions_all,
                 tile_constraints=loader.default_tile_constraints_nuclei, global_log_dir: str = None,
                 sigmoid_verbose: bool = False, render_attention_spheres_enabled: bool = True,
                 render_dose_response_curves_enabled: bool = True, hist_bins_override=None,
                 render_merged_predicted_tiles_activation_overlays: bool = False, gpu_enabled: bool = False,
                 render_attention_histogram_enabled: bool = False, data_loader_data_saver: bool = False):
    start_time = datetime.now()
    log_label = str(start_time.strftime("%d-%m-%Y-%H-%M-%S"))

    # Setting up log
    global_log_filename = None
    local_log_filename = out_dir + os.sep + 'log_predictions.txt'
    log.add_file(local_log_filename)
    if global_log_dir is not None:
        global_log_filename = global_log_dir + os.sep + 'log-predictions-' + log_label + '.txt'
        os.makedirs(global_log_dir, exist_ok=True)
        log.add_file(global_log_filename)
    log.diagnose()

    # Setting up dirs & paths
    log.write('Predicting from paths: ' + str(bag_paths))
    if not model_save_path.endswith(".h5"):
        model_save_path = model_save_path + "model.h5"
    model_state_dict_file = model_save_path[:-8] + 'model.pt'
    os.makedirs(out_dir, exist_ok=True)
    log.write('Predicting to: ' + out_dir)

    # Setting the device calculations will take part in
    device = hardware.get_hardware_device(gpu_preferred=gpu_enabled)
    log.write('Prediction device: ' + str(device))
    time.sleep(2)

    # Loading model
    log.write('Loading model: ' + model_save_path)
    log.write('Assumed model state dict: ' + model_state_dict_file)

    model = None
    loading_error = False
    if os.path.exists(model_save_path):
        try:
            model = torch.load(model_save_path, map_location='cpu')
            if not isinstance(model, models.OmniSpheroMil):
                log.write('Warning! Loaded data is the wrong type! Loaded Type: ' + str(type(model)))
                loading_error = True
        except Exception as e:
            log.write("Error loading model!")
            log.write(str(e))
            loading_error = True

    if (model is None or loading_error) and os.path.exists(model_state_dict_file):
        loss_function = 'binary_cross_entropy'
        accuracy_function = 'binary'
        log.write('Creating a new untrained model and feeding it the state dict')
        log.write('Selected device: ' + str(device))
        model = models.BaselineMIL((3, 150, 150), device=device, loss_function=None, accuracy_function=None)
        model.load_state_dict(torch.load(model_state_dict_file, map_location='cpu'))
    elif model is None or loading_error or not os.path.exists(model_state_dict_file):
        log.write('Model is None: ' + str(model is None))
        log.write('Has loading error: ' + str(loading_error))
        log.write('Exists state dict: ' + str(os.path.exists(model_state_dict_file)))
        log.write("\n\n ==== WARNING! INCORRECT MODEL PATH. YOUR MODEL MAY NOT HAVE BEEN LOADED ====\n\n")

    # Checking if the right data has been loaded
    assert model is not None
    assert isinstance(model, models.OmniSpheroMil)
    model = model.eval()

    # Loading best checkpoint
    model, _, _, _ = models.load_checkpoint(load_path=checkpoint_file, model=model, optimizer=None, map_location='cpu')
    log.write('Finished loading the model.')

    # Updating the model device
    model.device = device

    # Loading the data
    # X_full, y_full, y_tiles_full, X_raw_full, X_metadata, bag_names_full, experiment_names_full, well_names_full, error_list, loaded_files_list_full
    X, y, y_tiles, X_raw, X_metadata, bag_names, experiment_names, well_names, errors, loaded_files_list = loader.load_bags_json_batch(
        batch_dirs=bag_paths,
        max_workers=max_workers,
        include_raw=True,
        channel_inclusions=channel_inclusions,
        constraints_0=tile_constraints,
        constraints_1=tile_constraints,
        label_0_well_indices=loader.default_well_indices_all,
        label_1_well_indices=loader.default_well_indices_all,
        normalize_enum=normalize_enum)
    X = [np.einsum('bhwc->bchw', bag) for bag in X]

    # Printing the size in memory
    X_size = 0
    X_size_raw = 0
    for i in range(len(X)):
        X_size = X_size + X[i].nbytes
        X_size_raw = X_size_raw + X_raw[i].nbytes

    X_s = utils.convert_size(X_size)
    X_s_raw = utils.convert_size(X_size_raw)
    y_s = utils.convert_size(sys.getsizeof(y))
    log.write("X-size in memory (after loading all data): " + str(X_s))
    log.write("y-size in memory (after loading all data): " + str(y_s))
    log.write("X-size (raw) in memory (after loading all data): " + str(X_s_raw))
    del X_s, y_s, X_s_raw, X_size, X_size_raw

    dataset, input_dim = loader.convert_bag_to_batch(bags=X, labels=None, y_tiles=None)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                             pin_memory=False,
                             num_workers=max_workers)
    del X, dataset, y, y_tiles

    # TODO react to loading errors
    log.write('Number of files loaded: ' + str(len(loaded_files_list)))
    log.write('Number of loading errors: ' + str(len(errors)))
    del errors, loaded_files_list

    norm = True
    sparse = True
    predict_data(model=model, data_loader=data_loader, X_raw=X_raw, X_metadata=X_metadata,
                 experiment_names=experiment_names, input_dim=input_dim, sparse_hist=sparse,
                 hist_bins_override=hist_bins_override,
                 normalized_attention=norm, clear_old_data=False, image_folder=image_folder,
                 sigmoid_verbose=sigmoid_verbose, render_attention_histogram_enabled=render_attention_histogram_enabled,
                 render_merged_predicted_tiles_activation_overlays=render_merged_predicted_tiles_activation_overlays,
                 render_attention_spheres_enabled=render_attention_spheres_enabled,
                 render_dose_response_curves_enabled=render_dose_response_curves_enabled,
                 well_names=well_names, out_dir=out_dir + 'predictions')

    del data_loader, X_raw, X_metadata

    # All done and finishing up the logger
    log.write('Batch prediction finished.')
    log.remove_file(local_log_filename)
    if global_log_dir is not None:
        log.remove_file(global_log_filename)
    log.clear_files()


def predict_data(model: models.BaselineMIL, data_loader: OmniSpheroDataLoader, X_raw: [np.ndarray],
                 X_metadata: [TileMetadata], experiment_names: [str], well_names: [str], image_folder: str,
                 input_dim: (int), out_dir: str, sparse_hist: bool = True, hist_bins_override=None,
                 normalized_attention: bool = True, save_sigmoid_plot: bool = True, sigmoid_verbose: bool = False,
                 render_dose_response_curves_enabled: bool = True, histogram_bootstrap_replications: int = 1000,
                 histogram_resample_size: int = 100, attention_normalized_metrics: bool = True,
                 render_attention_spheres_enabled: bool = False, render_attention_histogram_enabled: bool = False,
                 render_merged_predicted_tiles_activation_overlays: bool = False, clear_old_data: bool = False,
                 dpi: int = 250):
    os.makedirs(out_dir, exist_ok=True)

    log.write('Using image folder: ' + image_folder)
    log.write('Exists image folder: ' + str(os.path.exists(image_folder)))
    log.write('R - Is pyRserve connection available: ' + str(r.has_connection(also_test_script=True)))

    log.write('Predicting ' + str(len(X_metadata)) + ' bags.')
    log.write('Saving predictions to: ' + out_dir)
    y_hats, y_preds, _, _, y_samples_pred, _, all_attentions, original_bag_indices = models.get_predictions(
        model, data_loader)
    assert len(X_metadata) == len(all_attentions)
    assert len(X_metadata) == len(X_raw)
    del data_loader, model

    # Running bayesian bootstrap
    if os.path.exists('IGNORE-DEPRECATED.txt'):
        # DEPRECATED FUNCTIONS
        bootstrap_list, bootstrap_threshold_indices_list, bootstrap_metadata_list = mil_metrics.bayesian_bootstrap_attention_batch(
            all_attentions=all_attentions, X_metadata=X_metadata, n_replications=histogram_bootstrap_replications,
            resample_size=histogram_resample_size)
        log.write('Finished predictions.')
        data_renderer.render_bootstrapped_histograms(out_dir=out_dir, bootstrap_list=bootstrap_list,
                                                     n_replications=histogram_bootstrap_replications,
                                                     bootstrap_threshold_indices_list=bootstrap_threshold_indices_list,
                                                     metadata_list=bootstrap_metadata_list)
        del bootstrap_list, bootstrap_threshold_indices_list, bootstrap_metadata_list

    # Evaluating attention metrics (histogram, entropy, etc.)
    attention_metadata_list, attention_n_list, attention_bins_list, attention_otsu_index_list, attention_otsu_threshold_list, attention_entropy_attention_list, attention_entropy_hist_list, error_list = mil_metrics.attention_metrics_batch(
        all_attentions=all_attentions,
        X_metadata=X_metadata,
        hist_bins_override=hist_bins_override,
        normalized=attention_normalized_metrics)

    log.write(" === START ERROR LIST ===")
    for error in error_list:
        log.write("ERROR WHILE PREDICTING: " + str(error))
        # TODO handle errors better
    log.write(" === END ERROR LIST ===")

    data_renderer.render_attention_histograms(out_dir=out_dir, metadata_list=attention_metadata_list,
                                              n_list=attention_n_list,
                                              bins_list=attention_bins_list, otsu_index_list=attention_otsu_index_list,
                                              otsu_threshold_list=attention_otsu_threshold_list,
                                              entropy_attention_list=attention_entropy_attention_list,
                                              entropy_hist_list=attention_entropy_hist_list, dpi=dpi * 2)

    # Setting up result directories and file handles
    experiment_names_unique = list(dict.fromkeys(experiment_names))
    well_letters_unique_candidates = []
    well_numbers_unique_candidates = []

    handles = {}
    for exp in experiment_names_unique:
        handles[exp] = {}
        # Removing previously existing paths, if they exist
        if os.path.exists(out_dir + os.sep + exp + os.sep) and clear_old_data:
            try:
                shutil.rmtree(out_dir + os.sep + exp + os.sep)
            except Exception as e:
                log.write("Cannot remove path for " + exp + ". Msg: " + str(e))
    del exp

    # Setting up the instructions out file
    sigmoid_instructions_file = out_dir + os.sep + 'sigmoid_instructions.csv'
    if not os.path.exists(sigmoid_instructions_file):
        f = open(sigmoid_instructions_file, 'w')
        f.write('Experiment;Instructions: Dose;Instructions: Response')
        f.close()

    # Evaluating sigmoid performance using R
    sigmoid_score_map = None
    sigmoid_plot_estimation_map = None
    sigmoid_plot_data_map = None
    sigmoid_instructions_map = None
    sigmoid_score_detail_map = None
    if r.has_connection(also_test_script=True) and render_dose_response_curves_enabled:
        sigmoid_score_map, sigmoid_score_detail_map, sigmoid_plot_estimation_map, sigmoid_plot_data_map, sigmoid_instructions_map = r.prediction_sigmoid_evaluation(
            X_metadata=X_metadata, y_pred=y_hats, out_dir=out_dir, verbose=sigmoid_verbose,
            save_sigmoid_plot=save_sigmoid_plot)
    else:
        log.write('Not running r evaluation. No connection.')

    # Logging the instructions
    if sigmoid_instructions_map is not None:
        f = open(sigmoid_instructions_file, 'a')
        for key in sigmoid_instructions_map.keys():
            sigmoid_instructions = sigmoid_instructions_map[key]
            f.write('\n' + key + ';' + sigmoid_instructions[0] + ';' + sigmoid_instructions[1])
            del key, sigmoid_instructions
        f.close()

    # Rendering basic response curves
    if render_dose_response_curves_enabled:
        log.write('Rendering dose response curves.')
        data_renderer.render_response_curves(X_metadata=X_metadata, y_pred=y_hats, out_dir=out_dir,
                                             sigmoid_plot_estimation_map=sigmoid_plot_estimation_map,
                                             sigmoid_plot_fit_map=sigmoid_plot_data_map,
                                             sigmoid_score_detail_map=sigmoid_score_detail_map,
                                             sigmoid_score_map=sigmoid_score_map, dpi=int(dpi * 1.337))

    # Rendering attention scores
    if render_attention_spheres_enabled:
        log.write('Rendering spheres and predictions.')
        data_renderer.renderAttentionSpheres(X_raw=X_raw, X_metadata=X_metadata, input_dim=input_dim,
                                             y_attentions=all_attentions, image_folder=image_folder, y_pred=y_hats,
                                             render_merged_predicted_tiles_activation_overlays=render_merged_predicted_tiles_activation_overlays,
                                             y_pred_binary=y_preds, out_dir=out_dir)
    else:
        log.write('Not rendering spheres and predictions.')

    del X_raw
    log.write('Finished rendering spheres.')

    bars_width_mod = 10000
    if sparse_hist:
        bars_width_mod = 1000
    print('\n')  # new line so linux systems can write a single line
    # Iterating over the predictions to save them to disc & dict
    for i in range(len(y_hats)):
        # Extracting predictions
        # X_raw_current = X_raw[i]
        X_metadata_current: [TileMetadata] = X_metadata[i]
        experiment_names_current = experiment_names[i]
        well_names_current = well_names[i]
        all_attentions_current = all_attentions[i]
        y_samples_pred_current = y_samples_pred[i]
        y_hat_current = y_hats[i]
        y_pred_current = y_preds[i]
        line_print('[' + str(i + 1) + '/' + str(len(y_preds)) + '] Evaluating predictions for: ' +
                   experiment_names_current + ' - ' + well_names_current, include_in_log=True)

        # Setting up handles & folders
        current_exp_handle = handles[experiment_names_current]
        if well_names_current not in current_exp_handle:
            current_exp_handle[well_names_current] = {}
        current_well_handle = current_exp_handle[well_names_current]
        well_letter, well_number = extract_well_info(well_names_current, verbose=False)
        well_letters_unique_candidates.append(well_letter)
        well_numbers_unique_candidates.append(well_number)

        # More Setting up handles & folders
        all_attentions_current_normalized = all_attentions_current / max(all_attentions_current)
        current_well_handle['attention'] = all_attentions_current
        current_out_dir = out_dir + os.sep + experiment_names_current + os.sep + well_names_current + os.sep
        os.makedirs(current_out_dir, exist_ok=True)

        # Checking if metadata matches
        assert X_metadata_current[0].experiment_name == experiment_names_current
        assert X_metadata_current[0].well_letter.lower() in well_names_current.lower()
        assert str(X_metadata_current[0].well_number) in well_names_current.lower()

        # Choosing sparse / normalized data
        all_attentions_used = all_attentions_current
        if normalized_attention:
            all_attentions_used = all_attentions_current_normalized

        # Saving histogram
        plt.clf()
        # if render_attention_histogram_enabled:
        #    plt.clf()
        #    if sparse_hist:
        #        n, bins = utils.sparse_hist(a=all_attentions_used)
        #        plt.bar(bins, n, width=(float(len(n)) / bars_width_mod), align='center', color='blue',
        #                edgecolor='white')
        #        threshold_index = utils.lecture_otsu(n=np.array(n))
        #        threshold = bins[threshold_index]
        #    else:
        #        n, bins, _ = plt.hist(x=all_attentions_used, bins=len(all_attentions_used), color='blue')
        #        plt.clf()
        #        n = list(n)
        #        bins = list(bins[:-1])
        #        plt.bar(bins, n, width=(float(len(n)) / bars_width_mod), align='center', color='blue',
        #                edgecolor='white')
        #        threshold_index = utils.lecture_otsu(n=np.array(n))
        #        threshold = bins[threshold_index]
        #
        #    plt.axvline(x=threshold, color='orange')
        #    plt.ylabel('Count (' + str(len(all_attentions_used)) + ' tiles total)')
        #    plt.xlabel('Attention Scores (Normalized)')
        #    plt.title(
        #        'Histogram of Normalized Attention Scores of: ' + experiment_names_current + ' - ' + well_names_current +
        #        '\nPrediction: ' + str(y_hat_current) + ' -> ' + str(y_pred_current))
        #    plt.legend(['Otsu Threshold: ' + str(int(threshold * 1000) / 1000)])
        #    plt.xlim([0, max(bins) * 1.05])
        #    plt.ylim([0, max(n) * 1.05])
        #    plt.grid(True)
        #    plt.autoscale()
        #    plt.savefig(current_out_dir + 'attention_hist.png', dpi=int(dpi * 1.337), bbox_inches='tight')
        #    plt.savefig(current_out_dir + 'attention_hist.svg', dpi=int(dpi * 1.337), bbox_inches='tight')
        #    plt.savefig(current_out_dir + 'attention_hist.pdf', dpi=int(dpi * 1.337), bbox_inches='tight')

        # Saving raw data as CSV
        f = open(current_out_dir + experiment_names_current + '-' + well_names_current + '-attention.csv', 'w')
        f.write('Index;Attention;Attention (Normalized)\n')
        for j in range(len(all_attentions_current)):
            f.write(str(j) + ';')
            f.write(str(all_attentions_current[j]) + ';' + str(all_attentions_used[j]))
            f.write('\n')
        f.write('Sum;' + str(sum(all_attentions_current)) + ';' + str(sum(all_attentions_used)))
        f.close()

        # Saving histogram
        # if render_attention_histogram_enabled:
        #    f = open(current_out_dir + experiment_names_current + '-' + well_names_current + '-histogram.csv', 'w')
        #    f.write('Index;n;bin\n')
        #    for j in range(len(n)):
        #        f.write(str(j) + ';')
        #        f.write(str(n[j]) + ';' + str(bins[j]))
        #        f.write('\n')
        #    f.close()
        #    current_well_handle['n'] = n
        #    current_well_handle['bins'] = bins
        #    current_well_handle['otsu'] = threshold
        #    current_well_handle['y_hat'] = y_hat_current
        #    current_well_handle['y_pred'] = y_pred_current
        #    del n, bins, threshold, y_hat_current, y_pred_current

        # writing the current handles back
        current_exp_handle[well_names_current] = current_well_handle
        handles[experiment_names_current] = current_exp_handle
        del experiment_names_current, current_exp_handle
    print('Done evaluating.')

    well_letters_unique: [int] = list(dict.fromkeys(well_letters_unique_candidates))
    well_numbers_unique: [str] = list(dict.fromkeys(well_numbers_unique_candidates))
    well_letters_unique.sort()
    well_numbers_unique.sort()
    del well_letters_unique_candidates, well_numbers_unique_candidates

    # if render_attention_histogram_enabled:
    #    print('\n')  # new line so linux systems can write a single line
    #    for e in range(len(experiment_names_unique)):
    #        exp = experiment_names_unique[e]
    #        line_print('[' + str(e + 1) + '/' + str(
    #            len(experiment_names_unique)) + '] Writing pooled histogram results for experiment: ' + exp,
    #                   include_in_log=True)
    #
    #        current_handle = handles[exp]
    #        current_out_dir = out_dir + os.sep + exp + os.sep
    #
    #        plt.clf()
    #        plt.title('Attention Comparisons: ' + exp)
    #        f, ax = plt.subplots(nrows=len(well_letters_unique), ncols=len(well_numbers_unique), figsize=(30, 30))
    #        for j in range(len(well_letters_unique)):
    #            well_letter = well_letters_unique[j]
    #            for i in range(len(well_numbers_unique)):
    #                well_number = well_numbers_unique[i]
    #                well_key = well_letter + '0' + str(well_number)
    #
    #                # subplot_index = str(len(well_letters_unique)) + str(len(well_numbers_unique)) + str(i + 1)
    #                axis_found = True
    #                try:
    #                    a = ax[j, i]
    #                except Exception as e:
    #                    log.write(' = WARNING = Failed to locate axis at ' + str(j) + ',' + str(i) + '!')
    #                    axis_found = False
    #                    continue
    #
    #                if well_key in current_handle.keys() and axis_found:
    #                    a = ax[j, i]
    #                    current_well_handle = current_handle[well_key]
    #                    threshold = current_well_handle['otsu']
    #                    bins = current_well_handle['bins']
    #                    n = current_well_handle['n']
    #                    attention = current_well_handle['attention']
    #                    y_hat = current_well_handle['y_hat']
    #                    y_pred = current_well_handle['y_pred']
    #                    attention_normalized = attention / max(attention)
    #
    #                    # a.hist(x=attention_normalized, bins=len(attention_normalized), color='blue')
    #                    # a.bar(bins, n, width=(float(len(n)) / 1000), color='blue', edgecolor='white')
    #                    a.bar(bins, n, width=(float(len(n)) / bars_width_mod), color='blue', edgecolor='white')
    #                    # a.plot([bins])
    #                    a.grid(True)
    #                    a.autoscale()
    #                    a.set_xlim([0, max(bins) * 1.05])
    #                    a.set_ylim([0, max(n) * 1.05])
    #                    a.axvline(x=threshold, color='orange')
    #                    a.legend(['Histogram - Bag Tiles: ' + str(len(attention_normalized)),
    #                              'Otsu Threshold: ' + str(int(threshold * 1000) / 1000)])
    #                    a.title.set_text(exp + ' - ' + well_key + '\nPrediction: ' + str(y_hat) + ' -> ' + str(y_pred))
    #                else:
    #                    ax[j, i].remove()  # remove Axes from fig
    #                    ax[j, i] = None  # make sure that there are no 'dangling' references.
    #
    #        plt.tight_layout()
    #        plt.autoscale()
    #        plt.savefig(current_out_dir + exp + '-hist.png', dpi=dpi, bbox_inches='tight')
    #        plt.savefig(current_out_dir + exp + '-hist.pdf', dpi=dpi, bbox_inches='tight')
    #        plt.savefig(current_out_dir + exp + '-hist.svg', dpi=dpi, bbox_inches='tight')
    #    log.write('All pooled results saved.')
    #    del y_hats, y_preds, y_samples_pred, _, all_attentions, original_bag_indices

    log.write('Finished predictions. Your results are here: ' + out_dir)


def generate_experiment_prediction_holders(X: [np.ndarray], experiment_names: [str], well_names: [str]):
    # Reordering bags based on metadata
    all_well_numbers = []
    all_well_letters = []
    experiment_holders = {}
    for (bag, experiment_name, well_name) in zip(X, experiment_names, well_names):
        if experiment_name not in experiment_holders.keys():
            experiment_holders[experiment_name] = {}

        current_holder = experiment_holders[experiment_name]
        well_letter, well_number = loader.extract_well_info(well_name, verbose=False)
        if well_number not in current_holder:
            current_holder[well_number] = {}

        if well_number not in all_well_numbers:
            all_well_numbers.append(well_number)
        if well_letter not in all_well_letters:
            all_well_letters.append(well_letter)

        current_holder[well_number][well_letter] = bag
        experiment_holders[experiment_name] = current_holder
        del current_holder
    all_well_numbers.sort()
    all_well_letters.sort()

    # Iterating over all experiments and evaluating one at a time
    experiment_names_unique = list(experiment_holders.keys())
    experiment_names_unique.sort()

    return experiment_holders, all_well_letters, all_well_numbers, experiment_names_unique


def main():
    print('Predicting and creating a Dose Response curve for a whole bag.')
    print('Platform:' + str(sys.platform))
    debug = False
    data_saver = True
    render_attention_spheres_enabled = True
    model_path = None
    image_folder = None

    if sys.platform == 'win32':
        debug = True

    normalize_enum = 6
    model_path = default_out_dir_unix_base + os.sep + 'hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65/'
    if sys.platform == 'win32':
        image_folder = paths.nucleus_predictions_image_folder_win
        current_global_log_dir = 'U:\\bioinfdata\\work\\OmniSphero\\Sciebo\\HCA\\00_Logs\\mil_log\\win\\'
        log.add_file('U:\\bioinfdata\\work\\OmniSphero\\Sciebo\\HCA\\00_Logs\\mil_log\\win\\all_logs.txt')
        model_path = 'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65\\'
    else:
        # good one:
        model_path = '/mil/oligo-diff/models/linux/ep-aug-overlap-adadelta-endpoints-wells-normalize-6no-repack-round1/'

        # testing:
        model_path = '/mil/oligo-diff/models/linux/ep-aug-overlap-adadelta-endpoints-wells-normalize-8repack-0.5-round1/'
        model_path = '/mil/oligo-diff/models/linux/ep-aug-overlap-adadelta-endpoints-wells-normalize-6repack-0.3-round1/'


        current_global_log_dir = '/Sciebo/HCA/00_Logs/mil_log/linux/'
        image_folder = paths.nucleus_predictions_image_folder_unix

    assert os.path.exists(model_path)
    if debug and sys.platform == 'win32':
        predict_path(
            model_save_path=model_path,
            global_log_dir=current_global_log_dir,
            checkpoint_file='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65\\hnm\\model_best.h5',
            bag_paths=debug_prediction_dirs_win,
            image_folder=image_folder,
            data_loader_data_saver=True,
            render_attention_spheres_enabled=render_attention_spheres_enabled,
            render_merged_predicted_tiles_activation_overlays=False,
            render_attention_histogram_enabled=False,
            render_dose_response_curves_enabled=True,
            hist_bins_override=50,
            sigmoid_verbose=True,
            out_dir='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\debug_predictions-linux\\',
            gpu_enabled=False, normalize_enum=normalize_enum, max_workers=4)
    elif sys.platform == 'win32':
        for prediction_dir in paths.all_prediction_dirs_win:
            predict_path(model_save_path=model_path,
                         global_log_dir=current_global_log_dir,
                         render_attention_spheres_enabled=render_attention_spheres_enabled,
                         render_attention_histogram_enabled=False,
                         render_merged_predicted_tiles_activation_overlays=False,
                         checkpoint_file='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65\\hnm\\model_best.h5',
                         out_dir=model_path + 'predictions-sigmoid\\',
                         bag_paths=[prediction_dir],
                         image_folder=image_folder,
                         hist_bins_override=50,
                         sigmoid_verbose=True,
                         gpu_enabled=False, normalize_enum=normalize_enum, max_workers=6)
    else:
        print('Predicting linux batches')

        checkpoint_file = model_path + os.sep + 'hnm' + os.sep + 'model_best.h5'
        if not os.path.exists(checkpoint_file):
            checkpoint_file = model_path + os.sep + 'model_best.h5'
        assert os.path.exists(checkpoint_file)

        prediction_dirs_used = [curated_overlapping_source_dirs_unix]
        if debug:
            prediction_dirs_used = [prediction_dirs_used[0][0:3]]
        if data_saver:
            prediction_dirs_used = [[d] for d in prediction_dirs_used[0]]

        prediction_dirs_used.sort()
        for i in range(len(prediction_dirs_used)):
            prediction_dir = prediction_dirs_used[i]
            log.write(str(i) + '/' + str(len(prediction_dirs_used)) + ' - Predicting: ' + str(prediction_dir))
            try:
                predict_path(checkpoint_file=checkpoint_file, model_save_path=model_path, bag_paths=prediction_dir,
                             out_dir='/mil/oligo-diff/debug_predictions/endpoint-sigmoid-linux-norm6/',
                             global_log_dir=current_global_log_dir,
                             render_attention_spheres_enabled=render_attention_spheres_enabled,
                             render_merged_predicted_tiles_activation_overlays=False,
                             render_attention_histogram_enabled=False,
                             hist_bins_override=50,
                             sigmoid_verbose=False,
                             image_folder=image_folder,
                             # tile_constraints=loader.default_tile_constraints_nuclei,
                             channel_inclusions=loader.default_channel_inclusions_no_neurites,
                             gpu_enabled=False, normalize_enum=normalize_enum, max_workers=20)
            except Exception as e:
                # TODO handle this exception better
                log.write('\n\n============================================================')
                log.write('Fatal error during HT predictions: "' + str(e) + '"!')
                log.write(str(e.__class__.__name__) + ': "' + str(e) + '"')
                raise e

    # Finishing up the logging process
    log.write('Finished predicting & Dose-Response for all bags.')


if __name__ == '__main__':
    main()
