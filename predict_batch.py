import math
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

import hardware
import loader
import models
from util import log
from util.omnisphero_data_loader import OmniSpheroDataLoader

model_debug_path = "U:\bioinfdata\work\OmniSphero\mil\oligo-diff\models\production\hnm-early_inverted-O1-adadelta-NoNeuron2-wells-normalize-6repack-0.55-BEST"


def predict_path(model_save_path: str, checkpoint_file: str, bag_path: str, normalize_enum: int, out_dir: str,
                 max_workers: int, channel_inclusions=loader.default_channel_inclusions_all,
                 tile_constraints=loader.default_tile_constraints_nuclei,
                 gpu_enabled: bool = True, model_optimizer=None):
    log.write('Predicting path: ' + bag_path)
    if not model_save_path.endswith(".h5"):
        model_save_path = model_save_path + "model.h5"
    model_state_dict_file = model_save_path[:-8] + 'model.pt'
    os.makedirs(out_dir, exist_ok=True)
    device = hardware.get_hardware_device(gpu_preferred=gpu_enabled)

    # Loading model
    log.write('Loading model: ' + model_save_path)
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
        log.write('Selected device: ' + str(device))
        model = models.BaselineMIL((3, 150, 150), device=device, loss_function=None, accuracy_function=None)
        model.load_state_dict(torch.load(model_state_dict_file, map_location='cpu'))
    else:
        log.write("WARNING! INCORRECT MODEL PATH")

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
    X, y, y_tiles, X_raw, bag_names, experiment_names, well_names, errors, loaded_files_list = loader.load_bags_json_batch(
        batch_dirs=[bag_path],
        max_workers=max_workers,
        include_raw=True,
        channel_inclusions=channel_inclusions,
        constraints_0=tile_constraints,
        constraints_1=tile_constraints,
        label_0_well_indices=loader.default_well_indices_all,
        label_1_well_indices=loader.default_well_indices_all,
        normalize_enum=normalize_enum)
    X = [np.einsum('bhwc->bchw', bag) for bag in X]
    X_raw = [np.einsum('bhwc->bchw', bag) for bag in X_raw]
    del X_raw

    # TODO react to loading errors
    log.write('Number of files loaded: ' + str(len(loaded_files_list)))
    log.write('Number of loading errors: ' + str(len(errors)))

    # Reordering bags based on metadata
    # experiment_names_unique = list(set(experiment_names))
    # well_names_unique = list(set(well_names))
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
    for current_experiment in experiment_names_unique:
        current_holder = experiment_holders[current_experiment]
        log.write('Predicting: ' + current_experiment)

        # Predicting all wells of this experiment
        all_wells, prediction_dict_bags, prediction_dict_samples, prediction_dict_well_names = predict_dose_response(
            experiment_holder=current_holder, experiment_name=current_experiment, model=model,
            max_workers=max_workers)

        # Writing the predictions to disk
        out_file_base = out_dir + os.sep + current_experiment
        save_prediction_csv(experiment_name=current_experiment, all_well_letters=all_well_letters,
                            prediction_dict=prediction_dict_samples,
                            file_path=out_file_base + '-samples.csv',
                            prediction_dict_well_names=prediction_dict_well_names)
        save_prediction_csv(experiment_name=current_experiment, all_well_letters=all_well_letters,
                            prediction_dict=prediction_dict_bags,
                            file_path=out_file_base + '-bags.csv',
                            prediction_dict_well_names=prediction_dict_well_names)

        # Saving preview dose response graph
        x_ticks_angle = 10
        x_ticks_font_size = 4
        save_prediction_img(out_file_base + '-bags.png', prediction_dict=prediction_dict_bags,
                            title='Dose Response: ' + current_experiment + ': ' + 'Whole Well',
                            prediction_dict_well_names=prediction_dict_well_names,
                            x_ticks_angle=x_ticks_angle, x_ticks_font_size=x_ticks_font_size)
        save_prediction_img(out_file_base + '-samples.png', prediction_dict=prediction_dict_samples,
                            title='Dose Response: ' + current_experiment + ': ' + 'Samples',
                            prediction_dict_well_names=prediction_dict_well_names,
                            x_ticks_angle=x_ticks_angle, x_ticks_font_size=x_ticks_font_size)


def save_prediction_img(file_path: str, title: str, prediction_dict: dict, prediction_dict_well_names: dict,
                        dpi: int = 900, x_ticks_angle: int = 30, x_ticks_font_size: int = 4, verbose: bool = False):
    # Writing the results as dose-response png images
    if verbose:
        log.write('Rendering dose response curve: "' + title + '" at ' + file_path)

    plt.clf()
    x_labels = [str(l) for l in prediction_dict_well_names.values()]
    ticks = list(range(len(x_labels)))
    plt.xticks(ticks=ticks, labels=x_labels, rotation=x_ticks_angle, fontsize=x_ticks_font_size)

    x = list(range(len(prediction_dict.keys())))
    y = [np.mean(prediction_dict[p]) for p in prediction_dict]
    plt.plot(x, y)

    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.xlabel('Wells')
    plt.ylabel('Dose Response Predictions')
    plt.title(title)
    plt.autoscale()
    
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    plt.savefig(file_path, dpi=dpi, bbox_inches="tight")


def save_prediction_csv(experiment_name: str, file_path: str, all_well_letters: [str], prediction_dict: dict,
                        prediction_dict_well_names: dict, verbose: bool = False):
    if verbose:
        log.write('Writing prediction CSV: ' + file_path)

    # Saving the results to a CSV file
    f = open(file_path, 'w')
    f.write(experiment_name + ';')
    [f.write(letter + ';') for letter in all_well_letters]
    f.write('Mean')
    for well_index in prediction_dict.keys():
        f.write('\n' + str(well_index) + ';')
        current_prediction = prediction_dict[well_index]
        current_well_names = prediction_dict_well_names[well_index]

        for letter in all_well_letters:
            for (prediction_value, predicted_well) in zip(current_prediction, current_well_names):
                if predicted_well == letter + str(well_index):
                    f.write(str(prediction_value))
            f.write(';')
        f.write(str(np.mean(current_prediction)))
    f.close()


def predict_dose_response(experiment_holder: dict, experiment_name: str, model: models.OmniSpheroMil,
                          max_workers: int = 1, verbose: bool = False):
    if verbose:
        log.write('Predicting experiment: ' + experiment_name)
    well_indices = list(experiment_holder.keys())
    well_indices.sort()

    prediction_dict_bags = {}
    prediction_dict_samples = {}
    prediction_dict_well_names = {}
    all_wells = []

    # Iterating over all well numbers
    for current_well_index in well_indices:
        well_letters = list(experiment_holder[current_well_index])
        well_letters.sort()

        if verbose:
            log.write(
                experiment_name + ' - Discovered ' + str(len(well_letters)) + ' replicas(s) for well index ' + str(
                    current_well_index) + ': ' + str(well_letters))

        replica_predictions_bags = []
        replica_predictions_samples = []
        replica_well_names = []

        # Iterating over all replcas for this concentration and adding raw predictions to the list
        for current_well_letter in well_letters:
            current_well = current_well_letter + str(current_well_index)
            if verbose:
                log.write('Predicting: ' + experiment_name + ' - ' + current_well)

            if current_well not in all_wells:
                all_wells.append(current_well)

            current_x = [experiment_holder[current_well_index][current_well_letter]]
            dataset, bag_input_dim = loader.convert_bag_to_batch(bags=current_x, labels=None, y_tiles=None)
            predict_dl = OmniSpheroDataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False,
                                              num_workers=max_workers,
                                              transform_enabled=False, transform_data_saver=False)
            del current_x, dataset
            if verbose:
                log.write('Model input dim: ' + str(model.input_dim))
                log.write('Loaded data input dim: ' + str(bag_input_dim))
            assert bag_input_dim == model.input_dim

            all_y_hats, all_predictions, all_true, all_y_tiles, all_y_tiles_binarized, all_tiles_true, all_attentions, original_bag_indices = models.get_predictions(
                model, predict_dl, verbose=False)
            del predict_dl

            if verbose:
                log.write('Finished predicting: ' + experiment_name + ' - ' + current_well)

            # Taking the means of the predictions and adding them to the list
            replica_predictions_bags.append(np.mean(all_y_hats))
            replica_predictions_samples.append(np.mean(all_y_tiles))
            replica_well_names.append(current_well)
            del all_y_hats, all_predictions, all_true, all_y_tiles, all_y_tiles_binarized, all_tiles_true, all_attentions, original_bag_indices

        # Adding the lists to the dict, so the dict will include means of the replicas
        prediction_dict_bags[current_well_index] = replica_predictions_bags
        prediction_dict_samples[current_well_index] = replica_predictions_samples
        prediction_dict_well_names[current_well_index] = replica_well_names
    del experiment_holder
    all_wells.sort()

    return all_wells, prediction_dict_bags, prediction_dict_samples, prediction_dict_well_names


def main():
    print('Predicting and creating a Dose Response curve for a whole bag.')
    debug = False
    model = None
    if sys.platform == 'win32':
        model = 'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65\\'

    if debug:
        predict_path(
            model_save_path=model,
            checkpoint_file='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65\\hnm\\model_best.h5',
            bag_path='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_win\\prediction',
            out_dir=model + 'predictions\\',
            gpu_enabled=False, normalize_enum=7, max_workers=4)
    elif sys.platform == 'win32':
        predict_path(
            model_save_path=model,
            checkpoint_file='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65\\hnm\\model_best.h5',
            bag_path='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux\\ELS517',
            out_dir=model + 'predictions\\',
            gpu_enabled=False, normalize_enum=7, max_workers=4)
        predict_path(
            model_save_path=model,
            checkpoint_file='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65\\hnm\\model_best.h5',
            bag_path='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux\\ELS411',
            out_dir=model + 'predictions\\',
            gpu_enabled=False, normalize_enum=7, max_workers=4)
        predict_path(
            model_save_path=model,
            checkpoint_file='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65\\hnm\\model_best.h5',
            bag_path='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux\\ELS682',
            out_dir=model + 'predictions\\',
            gpu_enabled=False, normalize_enum=7, max_workers=4)
        predict_path(
            model_save_path=model,
            checkpoint_file='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65\\hnm\\model_best.h5',
            bag_path='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux\\ELS719',
            out_dir=model + 'predictions\\',
            gpu_enabled=False, normalize_enum=7, max_workers=4)
        predict_path(
            model_save_path=model,
            checkpoint_file='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65\\hnm\\model_best.h5',
            bag_path='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux\\ELS637',
            out_dir=model + 'predictions\\',
            gpu_enabled=False, normalize_enum=7, max_workers=4)

    print('Finished predicting & Dose-Response for all bags.')


if __name__ == '__main__':
    main()
