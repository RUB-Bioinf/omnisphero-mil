import os
import sys

import numpy as np
import torch

import hardware
import loader
import models
import omnisphero_mil
from mil_metrics import save_sigmoid_prediction_csv
from mil_metrics import save_sigmoid_prediction_img
from util import log
from models import predict_dose_response
from util.paths import default_out_dir_unix_base

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
    log.write('Predicting to: ' + out_dir)

    # Setting the device calculations will take part in
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
    X, _, _, _, _, experiment_names, well_names, errors, loaded_files_list = loader.load_bags_json_batch(
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

    # TODO react to loading errors
    log.write('Number of files loaded: ' + str(len(loaded_files_list)))
    log.write('Number of loading errors: ' + str(len(errors)))

    # Repacking loaded experiments into holders for prediction
    experiment_holders, all_well_letters, all_well_numbers, experiment_names_unique = generate_experiment_prediction_holders(
        X=X, experiment_names=experiment_names, well_names=well_names)

    for current_experiment in experiment_names_unique:
        current_holder = experiment_holders[current_experiment]
        log.write('Predicting: ' + current_experiment)

        # Predicting all wells of this experiment
        all_wells, prediction_dict_bags, prediction_dict_samples, prediction_dict_well_names = predict_dose_response(
            experiment_holder=current_holder, experiment_name=current_experiment, model=model,
            max_workers=max_workers)

        # Writing the predictions to disk
        out_file_base = out_dir + os.sep + current_experiment
        save_sigmoid_prediction_csv(experiment_name=current_experiment, all_well_letters=all_well_letters,
                                    prediction_dict=prediction_dict_samples,
                                    file_path=out_file_base + '-samples.csv',
                                    prediction_dict_well_names=prediction_dict_well_names)
        save_sigmoid_prediction_csv(experiment_name=current_experiment, all_well_letters=all_well_letters,
                                    prediction_dict=prediction_dict_bags,
                                    file_path=out_file_base + '-bags.csv',
                                    prediction_dict_well_names=prediction_dict_well_names)

        # Saving preview dose response graph
        x_ticks_angle = 15
        x_ticks_font_size = 10
        save_sigmoid_prediction_img(out_file_base + '-bags.png', prediction_dict=prediction_dict_bags,
                                    title='Dose Response: ' + current_experiment + ': ' + 'Whole Well',
                                    prediction_dict_well_names=prediction_dict_well_names,
                                    x_ticks_angle=x_ticks_angle, x_ticks_font_size=x_ticks_font_size)
        save_sigmoid_prediction_img(out_file_base + '-samples.png', prediction_dict=prediction_dict_samples,
                                    title='Dose Response: ' + current_experiment + ': ' + 'Samples',
                                    prediction_dict_well_names=prediction_dict_well_names,
                                    x_ticks_angle=x_ticks_angle, x_ticks_font_size=x_ticks_font_size)


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
    debug = False
    model = None

    model = default_out_dir_unix_base + os.sep + 'hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65/'
    if sys.platform == 'win32':
        # debug = True
        model = 'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65\\'

    if debug:
        predict_path(
            model_save_path=model,
            checkpoint_file='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65\\hnm\\model_best.h5',
            bag_path='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_win\\',
            out_dir=model + 'predictions\\',
            gpu_enabled=False, normalize_enum=7, max_workers=4)
    elif sys.platform == 'win32':
        for data_dir in omnisphero_mil.all_source_dirs_win:
            predict_path(model_save_path=model,
                         checkpoint_file='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65\\hnm\\model_best.h5',
                         out_dir=model + 'predictions-win\\', bag_path=data_dir,
                         gpu_enabled=False, normalize_enum=7, max_workers=6)
    else:
        print('Predicting linux batches')
        # checkpoint_file = '/bph/puredata4/bioinfdata/work/OmniSphero/mil/oligo-diff/models/linux/hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65/'
        checkpoint_file = model + 'hnm/model_best.h5'
        for data_dir in omnisphero_mil.default_source_dirs_unix:
            predict_path(checkpoint_file=checkpoint_file, model_save_path=model, bag_path=data_dir,
                         out_dir=model + 'predictions-linux/',
                         gpu_enabled=False, normalize_enum=7, max_workers=20)

    print('Finished predicting & Dose-Response for all bags.')


if __name__ == '__main__':
    main()
