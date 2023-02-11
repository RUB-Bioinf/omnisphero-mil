import os
import shutil
import sys
import time
from datetime import datetime
from typing import Union

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
from util.paths import debug_prediction_dirs_win
from util.paths import default_out_dir_unix_base
from util.utils import line_print
from util.well_metadata import TileMetadata
from util.well_metadata import extract_well_info


def predict_path(model_save_path: str, checkpoint_file: str, bag_paths: [str], normalize_enum: int, out_dir: str,
                 max_workers: int, image_folder: str, channel_inclusions=loader.default_channel_inclusions_all,
                 tile_constraints=loader.default_tile_constraints_nuclei, global_log_dir: str = None,
                 sigmoid_verbose: bool = False,
                 render_attention_spheres_enabled: bool = True,
                 render_attention_instance_range_min: float = None,
                 render_attention_instance_range_max: float = None,
                 render_dose_response_curves_enabled: bool = True, hist_bins_override=None,
                 render_merged_predicted_tiles_activation_overlays: bool = False, gpu_enabled: bool = False,
                 render_attention_cell_distributions: bool = False,
                 render_attention_histogram_enabled: bool = False,
                 render_attention_cytometry_prediction_distributions_enabled: bool = False,
                 oligo_positive_z_score_scale: float = 1.0, oligo_z_score_max_kernel_size: int = 1,
                 predict_samples_as_bags: bool = False,
                 data_loader_data_saver: bool = False,
                 used_tile_quartiles=None,
                 clear_global_logs: bool = True,
                 out_image_dpi: int = 300,
                 ):
    start_time = datetime.now()
    log_label = str(start_time.strftime("%d-%m-%Y-%H-%M-%S"))
    print(out_dir)

    if used_tile_quartiles is None:
        used_tile_quartiles = loader.default_used_tile_quartiles.copy()
    used_tile_quartiles_enum = utils.boolean_to_integer(used_tile_quartiles[0],
                                                        used_tile_quartiles[1],
                                                        used_tile_quartiles[2],
                                                        used_tile_quartiles[3])

    # Setting up log
    global_log_filename = None
    local_log_filename = out_dir + os.sep + 'log_predictions.txt'
    print('Local Log file: ' + str(local_log_filename))
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

    log.write('z-score max kernel size: ' + str(oligo_z_score_max_kernel_size))
    log.write('z-score positive scale: ' + str(oligo_positive_z_score_scale))
    time.sleep(5)

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
    time.sleep(0.69)

    # Notifying about the quartiles to be predicted
    log.write('Using neurosphere tile quartiles: ' + str(used_tile_quartiles_enum) + ' (Enum)')
    log.write('Using neurosphere tile quartiles: ' + str(used_tile_quartiles))
    time.sleep(1)

    # Loading the data
    X, y, y_tiles, X_raw, X_metadata, bag_names, experiment_names, _, well_names, errors, loaded_files_list = loader.load_bags_json_batch(
        batch_dirs=bag_paths,
        max_workers=max_workers,
        include_raw=True,
        force_balanced_batch=False,
        channel_inclusions=channel_inclusions,
        used_tile_quartiles=used_tile_quartiles,
        constraints_0=tile_constraints,
        constraints_1=tile_constraints,
        label_0_well_indices=loader.default_well_indices_all,
        label_1_well_indices=loader.default_well_indices_all,
        positive_z_score_scale=oligo_positive_z_score_scale,
        z_score_max_kernel_size=oligo_z_score_max_kernel_size,
        normalize_enum=normalize_enum)
    X = [np.einsum('bhwc->bchw', bag) for bag in X]

    # Printing the size in memory
    X_s = str(utils.byteSizeString(utils.listToBytes(X)))
    X_s_raw = str(utils.byteSizeString(utils.listToBytes(X_raw)))
    y_s = str(utils.byteSizeString(sys.getsizeof(y)))
    log.write("X-size in memory (after loading all data): " + str(X_s))
    log.write("y-size in memory (after loading all data): " + str(y_s))
    log.write("X-size (raw) in memory (after loading all data): " + str(X_s_raw))
    del X_s, y_s, X_s_raw

    dataset, input_dim = loader.convert_bag_to_batch(bags=X, labels=None, y_tiles=None)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                             pin_memory=False,
                             num_workers=max_workers)
    del X, dataset, y, y_tiles

    # TODO react to loading errors
    log.write('Number of files loaded: ' + str(len(loaded_files_list)))
    log.write('Number of loading errors: ' + str(len(errors)))
    if len(errors) > 1:
        log.write("[!!]\n[ERRORS WHILE LOADING EXPERIMENTS!]\n[!!]", include_timestamp=False)
        for i in range(len(errors)):
            e = errors[i]
            log.write('Error #' + str(i) + ': "' + str(e) + '".')
            log.write_exception(e)
            del e
            del i
    del errors, loaded_files_list

    norm = True
    sparse = True
    predict_data(model=model, data_loader=data_loader, X_raw=X_raw, X_metadata=X_metadata,
                 experiment_names=experiment_names, input_dim=input_dim, sparse_hist=sparse,
                 hist_bins_override=hist_bins_override, out_image_dpi=out_image_dpi,
                 normalized_attention=norm, clear_old_data=False, image_folder=image_folder,
                 sigmoid_verbose=sigmoid_verbose, render_attention_histogram_enabled=render_attention_histogram_enabled,
                 render_merged_predicted_tiles_activation_overlays=render_merged_predicted_tiles_activation_overlays,
                 render_attention_spheres_enabled=render_attention_spheres_enabled,
                 render_dose_response_curves_enabled=render_dose_response_curves_enabled,
                 render_attention_cell_distributions=render_attention_cell_distributions,
                 render_attention_instance_range_min=render_attention_instance_range_min,
                 render_attention_instance_range_max=render_attention_instance_range_max,
                 predict_samples_as_bags=predict_samples_as_bags,
                 num_workers=max_workers,
                 render_attention_cytometry_prediction_distributions_enabled=render_attention_cytometry_prediction_distributions_enabled,
                 well_names=well_names, out_dir=out_dir)

    del data_loader, X_raw, X_metadata

    # All done and finishing up the logger
    log.write('Batch prediction finished.')
    log.remove_file(local_log_filename)
    if global_log_dir is not None:
        log.remove_file(global_log_filename)

    if clear_global_logs:
        log.clear_files()


def _predict_samples_as_bags(model, data_loader, X_raw: [np.ndarray], num_workers: int,
                             X_metadata: [TileMetadata], experiment_names: [str], well_names: [str], image_folder: str,
                             input_dim: (int), out_dir: str, dpi: 500):
    # TODO update arguments
    data_set = data_loader.dataset

    # Making sure all's well that predicts well
    assert len(X_raw) == len(X_metadata)
    assert len(experiment_names) == len(X_metadata)
    assert len(well_names) == len(X_metadata)
    assert len(well_names) == len(data_set)

    log.write('Predicting the samples of very bag as a bag.')
    for i in range(len(X_metadata)):
        dataset_current = data_set[i]
        X_metadata_current = X_metadata[i]
        X_raw_current = X_raw[i]
        experiment_name_current = experiment_names[i]
        well_name_current = well_names[i]

        sample_count = X_raw_current.shape[0]
        assert len(X_metadata_current) == sample_count
        assert len(X_metadata_current) == dataset_current[0].shape[0]
        metadata = X_metadata_current[0]
        log.write('[' + str(i + 1) + '/' + str(
            len(X_metadata)) + '] Predicting sample as bag for: ' + experiment_name_current + ' - ' + well_name_current)

        # iterating over every sample
        y_hat_samples = []
        out_image_localized_positive_raw = np.zeros((metadata.well_image_height, metadata.well_image_width, 3),
                                                    dtype=np.uint8)
        out_image_localized_overlay = np.zeros((metadata.well_image_height, metadata.well_image_width, 3),
                                               dtype=np.uint8)
        out_image_localized_overlay_map_r = np.ones((metadata.well_image_height, metadata.well_image_width),
                                                    dtype=np.uint8)
        out_image_localized_overlay_map_g = np.ones((metadata.well_image_height, metadata.well_image_width),
                                                    dtype=np.uint8)
        out_image_localized_overlay_map_b = np.ones((metadata.well_image_height, metadata.well_image_width),
                                                    dtype=np.uint8)
        out_image_localized_overlay_map_mask = np.zeros((metadata.well_image_height, metadata.well_image_width),
                                                        dtype=np.uint8)
        positive_samples = []
        positive_sample_count = 0
        task_time_durations = []
        for j in range(sample_count):
            task_start_time = datetime.now()
            log.write('[' + str(i + 1) + '/' + str(
                len(X_metadata)) + '] Predicting sample as bag for: ' + experiment_name_current + ' - ' + well_name_current + ': ' + str(
                j + 1) + '/' + str(sample_count), include_in_files=False)

            X_raw_sample = X_raw_current[j]
            X_metadata_sample = X_metadata_current[j]
            X_sample = dataset_current[0][j]

            # Expanding sample so it can be used for predictions
            X_sample = X_sample.copy()
            X_sample = np.copy(X_sample)
            X_sample = np.expand_dims(X_sample, axis=0)

            data_set_sample, _ = loader.convert_bag_to_batch(bags=[X_sample], labels=None, y_tiles=None)
            data_loader = DataLoader(data_set_sample, batch_size=1, shuffle=False,
                                     pin_memory=False,
                                     num_workers=num_workers)

            y_hats, y_preds, _, _, y_samples_pred, _, all_attentions, original_bag_indices = models.get_predictions(
                model, data_loader)
            y_pred = bool(y_preds[0])
            del y_hats, y_samples_pred, all_attentions, original_bag_indices, y_preds

            # Saving the results
            y_hat_samples.append(y_pred)
            X_raw_sample = np.copy(X_raw_sample)
            X_raw_sample = X_raw_sample.copy()
            width, height, _ = X_raw_sample.shape
            pos_x: int = int(X_metadata_sample.pos_x)
            pos_y: int = int(X_metadata_sample.pos_y)
            out_image_localized_overlay[pos_y:pos_y + height, pos_x:pos_x + width] = X_raw_sample
            if y_pred:
                positive_sample_count = positive_sample_count + 1
                positive_samples.append(X_raw_sample)

                out_image_localized_positive_raw[pos_y:pos_y + height, pos_x:pos_x + width] = X_raw_sample
                out_image_localized_overlay_map_r[pos_y:pos_y + height, pos_x:pos_x + width] = 1
                out_image_localized_overlay_map_b[pos_y:pos_y + height, pos_x:pos_x + width] = 0
                out_image_localized_overlay_map_g[pos_y:pos_y + height, pos_x:pos_x + width] = 0
                out_image_localized_overlay_map_mask[pos_y:pos_y + height, pos_x:pos_x + width] = 1

                out_image_localized_overlay_rgb = out_image_localized_overlay[pos_y:pos_y + height, pos_x:pos_x + width]
                out_image_localized_overlay_r = out_image_localized_overlay_rgb[:, :, 0] * 1.25
                out_image_localized_overlay_g = out_image_localized_overlay_rgb[:, :, 1] * 0.75
                out_image_localized_overlay_b = out_image_localized_overlay_rgb[:, :, 2] * 0.75

                out_image_localized_overlay_r = out_image_localized_overlay_r * 1.0
                out_image_localized_overlay_r = out_image_localized_overlay_r.astype(np.uint8)
                out_image_localized_overlay_g = out_image_localized_overlay_g.astype(np.uint8)
                out_image_localized_overlay_b = out_image_localized_overlay_b.astype(np.uint8)

                out_image_localized_overlay_rgb = np.dstack(
                    (out_image_localized_overlay_r, out_image_localized_overlay_g, out_image_localized_overlay_b))
                out_image_localized_overlay_rgb = out_image_localized_overlay_rgb.astype('uint8')

            task_end_time = datetime.now()
            task_time_diff = task_end_time - task_start_time
            task_time_durations.append(task_time_diff)
            mean_task_time = np.mean(task_time_durations)
            current_time = datetime.now()
            eta_time = current_time + (mean_task_time * (sample_count - (j + 1)))
            log.write('Task finished ETA: ' + str(eta_time.strftime("%d/%m/%Y, %H:%M:%S")), include_in_files=False,
                      include_timestamp=False)

        log.write('FINISHED ALL SAMPLES FOR THIS BAG!')
        log.write('Number of positive bags: ' + str(positive_sample_count))
        out_dir_current = out_dir + os.sep + experiment_name_current + os.sep + well_name_current + os.sep
        os.makedirs(out_dir_current, exist_ok=True)
        log.write('Saving to: ' + out_dir_current)

        # Updating the masks
        out_image_localized_overlay_map_r = out_image_localized_overlay_map_r * 255
        out_image_localized_overlay_map_g = out_image_localized_overlay_map_g * 255
        out_image_localized_overlay_map_b = out_image_localized_overlay_map_b * 255
        out_image_localized_overlay_r = out_image_localized_overlay[:, :, 0]
        out_image_localized_overlay_g = out_image_localized_overlay[:, :, 1]
        out_image_localized_overlay_b = out_image_localized_overlay[:, :, 2]

        out_image_localized_overlay_r = out_image_localized_overlay_map_r * 0.5 + out_image_localized_overlay_r * 0.5
        out_image_localized_overlay_g = out_image_localized_overlay_map_g * 0.5 + out_image_localized_overlay_g * 0.5
        out_image_localized_overlay_b = out_image_localized_overlay_map_b * 0.5 + out_image_localized_overlay_b * 0.5
        out_image_localized_overlay_map_mask = out_image_localized_overlay_map_mask.astype(np.bool)
        out_image_localized_overlay[:, :, 0][out_image_localized_overlay_map_mask] = out_image_localized_overlay_r[
            out_image_localized_overlay_map_mask]
        out_image_localized_overlay[:, :, 1][out_image_localized_overlay_map_mask] = out_image_localized_overlay_g[
            out_image_localized_overlay_map_mask]
        out_image_localized_overlay[:, :, 2][out_image_localized_overlay_map_mask] = out_image_localized_overlay_b[
            out_image_localized_overlay_map_mask]

        # TODO move all this rendering to the data renderer file
        # Rendering the data
        if positive_sample_count > 0:
            for j in range(len(positive_samples)):
                sample = positive_samples[j]
                # detail_image_name = out_dir_current_detail + os.sep + current_experiment_name + '-' + current_well + '-' + str(
                #     j) + '.png'
                # plt.imsave(detail_image_name, sample)
                positive_samples[j] = mil_metrics.outline_rgb_array(sample, None, None, bright_mode=True,
                                                                    override_colormap=[255, 255, 255])

            matching_samples_fused = mil_metrics.fuse_image_tiles(images=positive_samples, light_mode=False)
            fused_image_name = out_dir_current + os.sep + experiment_name_current + '-' + well_name_current + '_fused.png'
            fused_image_name_detail = out_dir_current + os.sep + experiment_name_current + '-' + well_name_current + '_fused-detail.png'
            plt.imsave(fused_image_name, matching_samples_fused)

            localized_image_name = out_dir_current + os.sep + experiment_name_current + '-' + well_name_current + '_localized.png'
            localized_image_name_overlay = out_dir_current + os.sep + experiment_name_current + '-' + well_name_current + '_localized-overlay.png'
            plt.imsave(localized_image_name, out_image_localized_positive_raw)
            plt.imsave(localized_image_name_overlay, out_image_localized_overlay)

            plt.clf()
            plt.imshow(out_image_localized_overlay)
            plt.title(experiment_name_current + ' - ' + well_name_current + '\nPositive Samples: ' + str(
                positive_sample_count) + '/' + str(sample_count))
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.autoscale()
            plt.savefig(fused_image_name_detail + '.png', dpi=dpi)
            plt.savefig(fused_image_name_detail + '.svg', dpi=dpi, transparent=True)
            plt.savefig(fused_image_name_detail + '.pdf', dpi=dpi)

    log.write('Done: Predicting the samples of very bag as a bag.')


def predict_data(model: models.BaselineMIL, data_loader: Union[OmniSpheroDataLoader, DataLoader], X_raw: [np.ndarray],
                 X_metadata: [TileMetadata], experiment_names: [str], well_names: [str], image_folder: str,
                 num_workers: int,
                 input_dim: (int), out_dir: str, sparse_hist: bool = True, hist_bins_override=None,
                 normalized_attention: bool = True, save_sigmoid_plot: bool = True, sigmoid_verbose: bool = False,
                 render_dose_response_curves_enabled: bool = True, histogram_bootstrap_replications: int = 1000,
                 histogram_resample_size: int = 100, attention_normalized_metrics: bool = True,
                 render_attention_spheres_enabled: bool = False, render_attention_histogram_enabled: bool = False,
                 render_attention_cytometry_prediction_distributions_enabled: bool = False,
                 render_attention_instance_range_min: float = None,
                 render_attention_instance_range_max: float = None,
                 predict_samples_as_bags: bool = False,
                 render_merged_predicted_tiles_activation_overlays: bool = False, clear_old_data: bool = False,
                 render_attention_cell_distributions: bool = False, out_image_dpi: int = 250):
    os.makedirs(out_dir, exist_ok=True)

    if not type(render_attention_instance_range_min) == type(render_attention_instance_range_max):
        if not (type(render_attention_instance_range_min) == type(None) or type(
                render_attention_instance_range_min) == type(float) or type(
            render_attention_instance_range_max) == type(None) or type(render_attention_instance_range_max) == type(
            float)):
            assert False

    log.write('Using image folder: ' + image_folder)
    log.write('Exists image folder: ' + str(os.path.exists(image_folder)))
    log.write('R - Is pyRserve connection available: ' + str(r.has_connection(also_test_script=True)))

    log.write('Predicting ' + str(len(X_metadata)) + ' bags.')
    log.write('Saving predictions to: ' + out_dir)
    y_hats, y_preds, _, _, y_samples_pred, _, all_attentions, original_bag_indices = models.get_predictions(
        model, data_loader)
    assert len(X_metadata) == len(all_attentions)
    assert len(X_metadata) == len(X_raw)

    #####################################################
    # PREDICTING SAMPLES AS IF THEY WERE BAGS
    #####################################################
    if predict_samples_as_bags:
        out_dir_samples_as_bags = out_dir + os.sep + 'sample_predictions'
        os.makedirs(out_dir_samples_as_bags, exist_ok=True)
        _predict_samples_as_bags(model=model, data_loader=data_loader, X_raw=X_raw, X_metadata=X_metadata,
                                 experiment_names=experiment_names, well_names=well_names, image_folder=image_folder,
                                 num_workers=num_workers, input_dim=input_dim, dpi=out_image_dpi,
                                 out_dir=out_dir_samples_as_bags)
        del out_dir_samples_as_bags

    del data_loader, model
    #####################################################
    # APPLYING THE MODEL, DOING THE PREDICTIONS
    #####################################################
    # Evaluating attention metrics (histogram, entropy, etc.)
    attention_metadata_list, attention_n_list, attention_bins_list, attention_otsu_index_list, attention_otsu_threshold_list, attention_entropy_attention_list, attention_entropy_hist_list, error_list = mil_metrics.attention_metrics_batch(
        all_attentions=all_attentions,
        X_metadata=X_metadata,
        hist_bins_override=hist_bins_override,
        normalized=attention_normalized_metrics)

    # WRITING ERRORS (if they exist)
    log.write(" === START ERROR LIST ===")
    for error in error_list:
        log.write("ERROR WHILE PREDICTING: " + str(error))
        log.write_exception(error)
    log.write(" === END ERROR LIST ===")

    #####################################################
    # RENDERING SAMPLES IN A GIVEN ATTENTION INTERVAL
    #####################################################
    if render_attention_instance_range_min is not None and render_attention_instance_range_max is not None:
        render_attention_instance_range_min = float(render_attention_instance_range_min)
        render_attention_instance_range_max = float(render_attention_instance_range_max)
        assert render_attention_instance_range_min <= render_attention_instance_range_max

        out_dir_attention_range = out_dir + os.sep + 'significant-attentions-' + str(
            render_attention_instance_range_min) + '-' + str(render_attention_instance_range_max) + os.sep
        os.makedirs(out_dir_attention_range, exist_ok=True)
        data_renderer.render_attention_instance_range(out_dir=out_dir_attention_range,
                                                      X_metadata=X_metadata,
                                                      y_preds=y_preds,
                                                      all_attentions=all_attentions,
                                                      X_raw=X_raw,
                                                      render_attention_instance_range_min=render_attention_instance_range_min,
                                                      render_attention_instance_range_max=render_attention_instance_range_max
                                                      )
        del out_dir_attention_range

    #####################################################
    # RENDERING CYTOMETRY PAPER PREDICTIONS PER SAMPLE
    #####################################################
    if render_attention_cytometry_prediction_distributions_enabled:
        out_dir_cytometry = out_dir + os.sep

        # Oligos
        try:
            data_renderer.render_attention_cytometry_prediction_distributions(out_dir=out_dir_cytometry,
                                                                              X_metadata=X_metadata,
                                                                              title_suffix=' (Oligodendrocytes)',
                                                                              y_preds=y_preds,
                                                                              all_attentions=all_attentions,
                                                                              filename_suffix='_oligo',
                                                                              include_oligo=True,
                                                                              include_neuron=False,
                                                                              include_nucleus=False,
                                                                              dpi=out_image_dpi)
            data_renderer.render_attention_cytometry_prediction_distributions_partitioned(out_dir=out_dir_cytometry,
                                                                                          X_metadatas=X_metadata,
                                                                                          partitions=[2, 4],
                                                                                          title_suffix=' (Oligodendrocytes)',
                                                                                          y_preds=y_preds,
                                                                                          all_attentions=all_attentions,
                                                                                          filename_suffix='_oligo',
                                                                                          include_oligo=True,
                                                                                          include_neuron=False,
                                                                                          include_nucleus=False,
                                                                                          dpi=out_image_dpi)
        except Exception as e:
            log.write_exception(e)

        # Neurons
        try:
            data_renderer.render_attention_cytometry_prediction_distributions(out_dir=out_dir_cytometry,
                                                                              X_metadata=X_metadata,
                                                                              title_suffix=' (Neurons)',
                                                                              y_preds=y_preds,
                                                                              all_attentions=all_attentions,
                                                                              filename_suffix='_neurons',
                                                                              include_oligo=False,
                                                                              include_neuron=True,
                                                                              include_nucleus=False,
                                                                              dpi=out_image_dpi)
            data_renderer.render_attention_cytometry_prediction_distributions_partitioned(out_dir=out_dir_cytometry,
                                                                                          X_metadatas=X_metadata,
                                                                                          partitions=[2, 4],
                                                                                          title_suffix=' (Neurons)',
                                                                                          y_preds=y_preds,
                                                                                          all_attentions=all_attentions,
                                                                                          filename_suffix='_neurons',
                                                                                          include_oligo=False,
                                                                                          include_neuron=True,
                                                                                          include_nucleus=False,
                                                                                          dpi=out_image_dpi)
        except Exception as e:
            log.write_exception(e)

        # Nuclei
        try:
            data_renderer.render_attention_cytometry_prediction_distributions(out_dir=out_dir_cytometry,
                                                                              X_metadata=X_metadata,
                                                                              title_suffix=' (Nuclei)',
                                                                              y_preds=y_preds,
                                                                              all_attentions=all_attentions,
                                                                              filename_suffix='_nuclei',
                                                                              include_oligo=False,
                                                                              include_neuron=False,
                                                                              include_nucleus=True,
                                                                              dpi=out_image_dpi)
            data_renderer.render_attention_cytometry_prediction_distributions_partitioned(out_dir=out_dir_cytometry,
                                                                                          X_metadatas=X_metadata,
                                                                                          partitions=[2, 4],
                                                                                          title_suffix=' (Nuclei)',
                                                                                          y_preds=y_preds,
                                                                                          all_attentions=all_attentions,
                                                                                          filename_suffix='_nuclei',
                                                                                          include_oligo=False,
                                                                                          include_neuron=False,
                                                                                          include_nucleus=True,
                                                                                          dpi=out_image_dpi)
        except Exception as e:
            log.write_exception(e)

        # All
        try:
            data_renderer.render_attention_cytometry_prediction_distributions(out_dir=out_dir_cytometry,
                                                                              X_metadata=X_metadata,
                                                                              title_suffix=' (All)',
                                                                              y_preds=y_preds,
                                                                              all_attentions=all_attentions,
                                                                              filename_suffix='_all',
                                                                              include_oligo=True,
                                                                              include_neuron=True,
                                                                              include_nucleus=True,
                                                                              dpi=out_image_dpi)
            data_renderer.render_attention_cytometry_prediction_distributions_partitioned(out_dir=out_dir_cytometry,
                                                                                          X_metadatas=X_metadata,
                                                                                          partitions=[2, 4],
                                                                                          title_suffix=' (All)',
                                                                                          y_preds=y_preds,
                                                                                          all_attentions=all_attentions,
                                                                                          filename_suffix='_all',
                                                                                          include_oligo=True,
                                                                                          include_neuron=True,
                                                                                          include_nucleus=True,
                                                                                          dpi=out_image_dpi)
        except Exception as e:
            log.write_exception(e)

    #####################################################
    # RENDERING ATTENTION CELL DISTRIBUTIONS
    #####################################################
    if render_attention_cell_distributions:
        # calculating the cell count distribution for every attention value
        distributions = mil_metrics.get_cells_per_attention(all_attentions, X_metadata)
        data_renderer.render_attention_cell_distributions(out_dir=out_dir, distributions=distributions,
                                                          X_metadata=X_metadata, alpha=0.45,
                                                          title_suffix=' (Oligodendrocytes)', filename_suffix='_oligo',
                                                          include_oligo=True,
                                                          include_neuron=False,
                                                          include_nucleus=False,
                                                          dpi=out_image_dpi)
        data_renderer.render_attention_cell_distributions(out_dir=out_dir, distributions=distributions,
                                                          X_metadata=X_metadata, alpha=0.45,
                                                          title_suffix=' (Neurons)', filename_suffix='_neuron',
                                                          include_oligo=False,
                                                          include_neuron=True,
                                                          include_nucleus=False,
                                                          dpi=out_image_dpi)
        data_renderer.render_attention_cell_distributions(out_dir=out_dir, distributions=distributions,
                                                          X_metadata=X_metadata, alpha=0.45,
                                                          title_suffix=' (Nuclei)', filename_suffix='_nucleus',
                                                          include_oligo=False,
                                                          include_neuron=False,
                                                          include_nucleus=True,
                                                          dpi=out_image_dpi)
        data_renderer.render_attention_cell_distributions(out_dir=out_dir, distributions=distributions,
                                                          X_metadata=X_metadata,
                                                          title_suffix=None, filename_suffix='_all',
                                                          include_oligo=True,
                                                          include_neuron=True,
                                                          include_nucleus=True,
                                                          dpi=out_image_dpi)

    #####################################################
    # RENDER ATTENTION HISTOGRAMS
    #####################################################
    data_renderer.render_attention_histograms(out_dir=out_dir, metadata_list=attention_metadata_list,
                                              n_list=attention_n_list,
                                              bins_list=attention_bins_list, otsu_index_list=attention_otsu_index_list,
                                              otsu_threshold_list=attention_otsu_threshold_list,
                                              entropy_attention_list=attention_entropy_attention_list,
                                              entropy_hist_list=attention_entropy_hist_list, dpi=out_image_dpi * 2)

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
    sigmoid_bmc30_map = None
    if r.has_connection(also_test_script=True) and render_dose_response_curves_enabled:
        sigmoid_score_map, sigmoid_score_detail_map, sigmoid_plot_estimation_map, sigmoid_plot_data_map, sigmoid_instructions_map, sigmoid_bmc30_map = r.prediction_sigmoid_evaluation(
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
                                             sigmoid_bmc30_map=sigmoid_bmc30_map,
                                             sigmoid_score_map=sigmoid_score_map, dpi=int(out_image_dpi * 1.337))

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

        # Saving raw data as CSV
        f = open(current_out_dir + experiment_names_current + '-' + well_names_current + '-attention.csv', 'w')
        f.write('Index;Attention;Attention (Normalized)\n')
        for j in range(len(all_attentions_current)):
            f.write(str(j) + ';')
            f.write(str(all_attentions_current[j]) + ';' + str(all_attentions_used[j]))
            f.write('\n')
        f.write('Sum;' + str(sum(all_attentions_current)) + ';' + str(sum(all_attentions_used)))
        f.close()

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

    if sys.platform == 'win32':
        debug = True

    normalize_enum = None
    model_path = default_out_dir_unix_base + os.sep + 'hnm-early_inverted-O3-adam-NoNeuron2-wells-normalize-7repack-0.65/'
    if sys.platform == 'win32':
        image_folder = paths.nucleus_predictions_image_folder_win
        current_global_log_dir = 'U:\\bioinfdata\\work\\OmniSphero\\Sciebo\\HCA\\00_Logs\\mil_log\\win\\'
        log.add_file('U:\\bioinfdata\\work\\OmniSphero\\Sciebo\\HCA\\00_Logs\\mil_log\\win\\all_logs.txt')
        model_path = paths.windows_debug_model_path
        normalize_enum = 6
    else:
        # good one:
        model_path = '/mil/oligo-diff/models/linux/ep-aug-overlap-adadelta-endpoints-wells-normalize-6no-repack-round1/'

        # testing:
        model_path = '/mil/oligo-diff/models/linux/ep-aug-overlap-adadelta-endpoints-wells-normalize-8repack-0.5-round1/'
        model_path = '/mil/oligo-diff/models/linux/ep-aug-overlap-adadelta-endpoints-wells-normalize-6repack-0.3-round1/'

        # in production
        model_path = '/mil/oligo-diff/models/production/paper_candidate_2/'

        current_global_log_dir = '/Sciebo/HCA/00_Logs/mil_log3/linux-pred/'
        image_folder = paths.nucleus_predictions_image_folder_unix
        normalize_enum = 7

    print('Log location: ' + current_global_log_dir)
    print('Model path: ' + model_path)
    assert os.path.exists(model_path)

    checkpoint_file = model_path + os.sep + 'hnm' + os.sep + 'model_best.h5'
    if not os.path.exists(checkpoint_file):
        checkpoint_file = model_path + os.sep + 'model_best.h5'
    assert os.path.exists(checkpoint_file)

    if debug and sys.platform == 'win32':
        predict_path(
            model_save_path=model_path,
            global_log_dir=current_global_log_dir,
            checkpoint_file=checkpoint_file,
            bag_paths=debug_prediction_dirs_win,
            image_folder=image_folder,
            data_loader_data_saver=True,
            render_attention_spheres_enabled=render_attention_spheres_enabled,
            channel_inclusions=loader.default_channel_inclusions_no_neurites,
            render_merged_predicted_tiles_activation_overlays=False,
            render_attention_histogram_enabled=False,
            render_dose_response_curves_enabled=True,
            predict_samples_as_bags=True,
            used_tile_quartiles=None,
            # oligo_positive_z_score_scale=2.0,
            # oligo_z_score_max_kernel_size=10,
            render_attention_instance_range_min=0.8,
            render_attention_instance_range_max=1.0,
            hist_bins_override=50,
            sigmoid_verbose=True,
            out_image_dpi=300,
            render_attention_cytometry_prediction_distributions_enabled=True,
            out_dir='U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\debug_predictions-win\\bmc\\',
            gpu_enabled=False, normalize_enum=normalize_enum, max_workers=4)
    elif sys.platform == 'win32':
        checkpoint_file = 'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\ep-aug-overlap-adadelta-n-6-rp-0.3-l-mean_square_error-BMC\\model_best.h5'
        for prediction_dir in paths.all_prediction_dirs_win:
            predict_path(model_save_path=model_path,
                         global_log_dir=current_global_log_dir,
                         render_attention_spheres_enabled=render_attention_spheres_enabled,
                         render_attention_histogram_enabled=False,
                         render_merged_predicted_tiles_activation_overlays=False,
                         checkpoint_file=checkpoint_file,
                         out_dir=checkpoint_file + 'predictions-bmc\\',
                         bag_paths=[prediction_dir],
                         channel_inclusions=loader.default_channel_inclusions_no_neurites,
                         image_folder=image_folder,
                         hist_bins_override=50,
                         sigmoid_verbose=True,
                         used_tile_quartiles=None,
                         out_image_dpi=300,
                         render_attention_instance_range_min=0.8,
                         render_attention_instance_range_max=1.0,
                         render_dose_response_curves_enabled=True,
                         predict_samples_as_bags=False,
                         render_attention_cytometry_prediction_distributions_enabled=True,
                         gpu_enabled=False, normalize_enum=normalize_enum, max_workers=6)
    else:
        print('Predicting linux batches')

        ###########################################################
        ############## SETTING THE INPUT PATH HERE ################
        ###########################################################

        prediction_dirs_used = [paths.all_prediction_dirs_unix]
        # prediction_dirs_used = [paths.curated_overlapping_source_dirs_unix_channel_transformed_rbg]
        # prediction_dirs_used = [paths.default_sigmoid_validation_dirs_unix]
        ###########################################################

        ###########################################################
        # Setting the quartiles to be used
        tile_quartiles = [[True, False, False, False],
                          [False, True, False, False],
                          [False, False, True, False],
                          [False, False, False, True]
                          ]
        # tile_quartiles = [loader.default_used_tile_quartiles]
        ###########################################################

        if debug:
            prediction_dirs_used = [prediction_dirs_used[0][0:3]]
        if data_saver:
            prediction_dirs_used = [[d] for d in prediction_dirs_used[0]]
        prediction_dirs_used.sort()

        time.sleep(2)
        log.write('Number of plates to be predicted: ' + str(len(prediction_dirs_used)))
        for i in range(len(prediction_dirs_used)):
            prediction_dir = prediction_dirs_used[i]
            log.write('#' + str(i + 1) + ': ' + str(prediction_dir))
            del prediction_dir
            time.sleep(0.69)
        time.sleep(3)

        for used_tile_quartiles in tile_quartiles:
            used_tile_quartiles_enum = utils.boolean_to_integer(used_tile_quartiles[0],
                                                                used_tile_quartiles[1],
                                                                used_tile_quartiles[2],
                                                                used_tile_quartiles[3])

            for i in range(len(prediction_dirs_used)):
                prediction_dir = prediction_dirs_used[i]
                log.write(
                    str(i + 1) + '/' + str(len(prediction_dirs_used)) + ' - Predicting: ' + str(prediction_dir))
                if not type(prediction_dir) == list:
                    prediction_dir = [prediction_dir]
                try:
                    out_dir_used = '/mil/oligo-diff/models/production/predictions/paper_candidate_2-quartile-' + str(
                        used_tile_quartiles_enum)
                    os.makedirs(out_dir_used, exist_ok=True)
                    predict_path(checkpoint_file=checkpoint_file, model_save_path=model_path,
                                 bag_paths=prediction_dir,
                                 out_dir=out_dir_used,
                                 global_log_dir=current_global_log_dir,
                                 render_attention_spheres_enabled=render_attention_spheres_enabled,
                                 render_merged_predicted_tiles_activation_overlays=False,
                                 render_attention_histogram_enabled=True,
                                 render_attention_cell_distributions=False,
                                 render_dose_response_curves_enabled=True,
                                 predict_samples_as_bags=False,
                                 used_tile_quartiles=used_tile_quartiles,
                                 render_attention_cytometry_prediction_distributions_enabled=False,
                                 hist_bins_override=50,
                                 out_image_dpi=800,
                                 sigmoid_verbose=False,
                                 render_attention_instance_range_min=0.9,
                                 render_attention_instance_range_max=1.0,
                                 image_folder=image_folder,
                                 tile_constraints=loader.default_tile_constraints_none,
                                 # tile_constraints=loader.default_tile_constraints_nuclei,
                                 channel_inclusions=loader.default_channel_inclusions_all,
                                 gpu_enabled=False, normalize_enum=normalize_enum, max_workers=20)
                except Exception as e:
                    log.write('\n\n============================================================')
                    log.write('Fatal error during HT predictions: "' + str(e) + '"!')
                    log.write(str(e.__class__.__name__) + ': "' + str(e) + '"')
                    log.write_exception(e)
                    return


# Finishing up the logging process
log.write('Finished predicting & Dose-Response for all bags.')

if __name__ == '__main__':
    main()
