import math

import mil_metrics
import nucleus_predictions
from util import log
import numpy as np
import os

from util.utils import line_print
from util.well_metadata import TileMetadata
import matplotlib.pyplot as plt


def renderAttentionSpheres(X_raw: [np.ndarray], X_metadata: [TileMetadata], y_pred: [np.ndarray],
                           y_pred_binary: [np.ndarray], image_folder: str, input_dim, y_attentions: [np.ndarray] = None,
                           out_dir: str = None, colormap_name: str = 'jet', dpi: int = 650,
                           render_merged_predicted_tiles_activation_overlays: bool = False,
                           overlay_alpha: float = 0.65):
    os.makedirs(out_dir, exist_ok=True)

    assert input_dim[0] == 3
    tile_w = int(input_dim[1])
    tile_h = int(input_dim[2])
    image_height = float('nan')
    image_width = float('nan')
    image_height_detail = float('nan')
    image_width_detail = float('nan')
    color_map = plt.get_cmap(colormap_name)

    activation_grayscale_overlay_images = {}
    activation_grayscale_overlay_images_detail = {}

    print('\n')
    for i in range(len(X_raw)):
        X_raw_current = X_raw[i]
        X_metadata_current: [TileMetadata] = X_metadata[i]
        y_attention_current = y_attentions[i]
        y_pred_current = float(y_pred[i])
        y_pred_binary_current = int(y_pred_binary[i])
        y_attention_current_normalized = y_attention_current / max(y_attention_current)
        color_bar_min = y_attention_current.min()
        color_bar_max = y_attention_current.max()

        # Collecting meta data to reconstruct the wells
        X_metadata_sample: TileMetadata = X_metadata_current[0]
        image_height = X_metadata_sample.well_image_height
        image_width = X_metadata_sample.well_image_width
        experiment_name = X_metadata_sample.experiment_name
        well_name = X_metadata_sample.get_formatted_well(long=True)

        # Loading previously predicted data
        prediction_coordinates: [int, int] = []
        _, exists_oligo_predictions = nucleus_predictions.get_prediction_file_path(image_folder=image_folder,
                                                                                   experiment_name=experiment_name,
                                                                                   well_name=well_name)
        if exists_oligo_predictions:
            prediction_coordinates = nucleus_predictions.load_oligo_predictions(image_folder=image_folder,
                                                                                experiment_name=experiment_name,
                                                                                well_name=well_name)

        out_dir_current = out_dir + os.sep + experiment_name + os.sep + well_name + os.sep
        os.makedirs(out_dir_current, exist_ok=True)
        line_print(str(i) + '/' + str(len(X_raw)) + ' Rendering: overlay sphere ' + experiment_name + ' - ' + well_name,
                   include_in_log=True)
        rendered_image = np.zeros((image_width, image_height, 3), dtype=np.uint8)

        for current_raw_tile, current_raw_metadata in zip(X_raw_current, X_metadata_current):
            # Checking if the metadata is correct
            assert current_raw_metadata.get_formatted_well(long=True) == well_name
            assert current_raw_metadata.experiment_name == experiment_name
            assert current_raw_tile.shape == (tile_w, tile_h, 3)
            current_raw_tile = current_raw_tile.astype(np.uint8)

            pos_x = int(current_raw_metadata.pos_x)
            pos_y = int(current_raw_metadata.pos_y)
            assert not math.isnan(pos_x) and not math.isnan(pos_y)
            # TODO handle this better than assertion

            # Debugging output
            # log.write('Rendered image size: ' + str(image_width) + 'x' + str(image_height))
            # log.write('Tile size: ' + str(tile_w) + 'x' + str(tile_h))
            # log.write('Tile insertion: ' + str(pos_y) + ':' + str(pos_y + tile_h) + ',' + str(pos_x) + ':' + str(
            #     pos_x + tile_w))

            rendered_image[pos_y:pos_y + tile_h, pos_x:pos_x + tile_w] = current_raw_tile
            del pos_x, pos_y, current_raw_tile

        # Saving the rendered image
        plt.imsave(out_dir_current + 'predicted_tiles.jpg', rendered_image)
        rendered_image_grayscale = rgb_to_gray(rendered_image)
        del rendered_image
        rendered_image_grayscale = rendered_image_grayscale.astype(np.uint8)
        plt.imsave(out_dir_current + 'predicted_tiles_grayscale.jpg', rendered_image_grayscale)

        activation_map = np.zeros((image_width, image_height, 3), dtype=np.float64)
        activation_grayscale_overlay = np.zeros((image_width, image_height, 3), dtype=np.uint8)
        activation_map_mask = np.zeros((image_width, image_height, 3), dtype=bool)
        for attention, current_raw_metadata in zip(y_attention_current_normalized, X_metadata_current):
            pos_x = current_raw_metadata.pos_x
            pos_y = current_raw_metadata.pos_y

            existing_attention = activation_map[pos_y:pos_y + tile_h, pos_x:pos_x + tile_w]
            activation_map_mask[pos_y:pos_y + tile_h, pos_x:pos_x + tile_w] = True
            if np.sum(existing_attention) == 0:
                # Clean set. Attention at this tile has not been set.
                activation_map[pos_y:pos_y + tile_h, pos_x:pos_x + tile_w] = attention
            else:
                # log.write('Overwriting attention!')
                attention_tile = np.ones((tile_h, tile_w, 3)) * attention
                attention_tile = np.maximum(attention_tile, activation_map[pos_y:pos_y + tile_h, pos_x:pos_x + tile_w])
                activation_map[pos_y:pos_y + tile_h, pos_x:pos_x + tile_w] = attention_tile
                del attention_tile

            attention_color = np.asarray(color_map(attention)[:-1])
            rgb = rendered_image_grayscale[pos_y:pos_y + tile_h, pos_x:pos_x + tile_w]
            r = rgb[:, :, 0] / 255 * overlay_alpha
            g = rgb[:, :, 1] / 255 * overlay_alpha
            b = rgb[:, :, 2] / 255 * overlay_alpha

            r = r + (attention_color[0] * (1 - overlay_alpha))
            g = g + (attention_color[1] * (1 - overlay_alpha))
            b = b + (attention_color[2] * (1 - overlay_alpha))

            r = r * 255
            g = g * 255
            b = b * 255
            r = r.astype('uint8')
            g = g.astype('uint8')
            b = b.astype('uint8')
            rgb = np.dstack((r, g, b))
            activation_grayscale_overlay[pos_y:pos_y + tile_h, pos_x:pos_x + tile_w] = rgb

            del pos_x, pos_y, attention, existing_attention, r, g, b, rgb, attention_color, current_raw_metadata

        prediction_coordinate_map = np.zeros((image_width, image_height, 3))
        for (x, y) in prediction_coordinates:
            prediction_coordinate_map[x, y, :] = (1, 1, 1)
            activation_grayscale_overlay[x, y, :] = (255, 255, 255)
            del x, y

        assert activation_map.min() == 0
        assert activation_map.max() == 1
        activation_map = activation_map * 255
        activation_map = activation_map.astype(np.uint8)
        activation_map_mask = activation_map_mask.astype(np.uint8)
        activation_map_mask = activation_map_mask * 255
        prediction_coordinate_map = prediction_coordinate_map.astype(np.uint8)
        prediction_coordinate_map = prediction_coordinate_map * 255

        plt.imsave(out_dir_current + 'predicted_coordinates_map.png', prediction_coordinate_map)
        plt.imsave(out_dir_current + 'predicted_tiles_activation_map.png', activation_map)
        plt.imsave(out_dir_current + 'predicted_tiles_activation_map_mask.png', activation_map_mask)
        plt.imsave(out_dir_current + 'predicted_tiles_activation_overlay.png', activation_grayscale_overlay)

        plt.clf()
        img = plt.imshow(np.array([[color_bar_min, color_bar_max]]), cmap=colormap_name)
        img.set_visible(False)
        c_bar = plt.colorbar(orientation='vertical')
        c_bar.ax.set_ylabel('Attention', rotation=270)
        plt.imshow(activation_grayscale_overlay)
        plt.xticks([], [])
        plt.yticks([], [])

        plt.xlabel('Prediction: ' + str(y_pred_current) + ' -> ' + str(y_pred_binary_current))
        plt.ylabel('Tiles: ' + str(len(y_attention_current)))
        plt.title('Localized Attention Scores: ' + experiment_name + ' - ' + well_name)

        plt.tight_layout()
        plt.autoscale()
        plt.savefig(out_dir_current + 'predicted_tiles_activation_overlay-detail.png', dpi=dpi, bbox_inches='tight')
        plt.savefig(out_dir_current + 'predicted_tiles_activation_overlay-detail.pdf', dpi=dpi, bbox_inches='tight')

        rendered_detail_fig = plt.imread(out_dir_current + 'predicted_tiles_activation_overlay-detail.png')
        rendered_detail_fig = rendered_detail_fig[:, :, :-1] * 255
        rendered_detail_fig = rendered_detail_fig.astype(np.uint8)

        if experiment_name not in activation_grayscale_overlay_images.keys():
            activation_grayscale_overlay_images[experiment_name] = []
            activation_grayscale_overlay_images_detail[experiment_name] = []
        activation_grayscale_overlay_images[experiment_name].append(activation_grayscale_overlay)
        activation_grayscale_overlay_images_detail[experiment_name].append(rendered_detail_fig)

    # Printing the imfused images
    if render_merged_predicted_tiles_activation_overlays:
        for experiment_name in activation_grayscale_overlay_images.keys():
            activation_grayscale_overlay_image = activation_grayscale_overlay_images[experiment_name]
            activation_grayscale_overlay_image_detail = activation_grayscale_overlay_images_detail[experiment_name]

            activation_grayscale_overlay_image = mil_metrics.fuse_image_tiles(images=activation_grayscale_overlay_image,
                                                                              light_mode=True)
            activation_grayscale_overlay_image_detail = mil_metrics.fuse_image_tiles(
                images=activation_grayscale_overlay_image_detail,
                light_mode=True)

            out_dir_exp = out_dir + os.sep + experiment_name
            os.makedirs(out_dir_exp, exist_ok=True)
            plt.imsave(out_dir_exp + os.sep + experiment_name + '-predicted_tiles_activation_overlays.png',
                       activation_grayscale_overlay_image)
            plt.imsave(out_dir_exp + os.sep + experiment_name + '-predicted_tiles_activation_overlays_detail.png',
                       activation_grayscale_overlay_image_detail)

    log.write('Finished rendering all attention overlays.')


def render_response_curves(X_metadata: [TileMetadata], y_pred: [np.ndarray], sigmoid_score_map: {str} = None,
                           file_name_suffix: str = None, title_suffix: str = None, out_dir: str = None,
                           sigmoid_plot_fit_map: {np.ndarray} = None, sigmoid_plot_estimation_map=None,
                           sigmoid_score_detail_map: {} = None, dpi: int = 650):
    # Remapping predictions so they can be evaluated
    experiment_prediction_map_pooled = {}
    experiment_prediction_map = {}
    experiment_well_tick_map = {}
    all_experiment_names = []
    all_well_names = []
    all_well_indices = []
    all_well_letters = []

    if file_name_suffix is None:
        file_name_suffix = ''
    if title_suffix is None:
        title_suffix = ''

    for (X_metadata_current, y_pred_current) in zip(X_metadata, y_pred):
        y_pred_current: float = float(y_pred_current)
        metadata: TileMetadata = X_metadata_current[0]

        experiment_name = metadata.experiment_name
        well_letter = metadata.well_letter
        well_number = metadata.well_number
        well = metadata.get_formatted_well()

        if experiment_name not in all_experiment_names:
            all_experiment_names.append(experiment_name)
        if well not in all_well_names:
            all_well_names.append(well)
        if well_number not in all_well_indices:
            all_well_indices.append(well_number)
        if well_letter not in all_well_letters:
            all_well_letters.append(well_letter)

        if experiment_name not in experiment_prediction_map_pooled.keys():
            experiment_prediction_map_pooled[experiment_name] = {}
            experiment_prediction_map[experiment_name] = {}
            experiment_well_tick_map[experiment_name] = {}

        experiment_well_tick_index_map = experiment_well_tick_map[experiment_name]
        well_index_map = experiment_prediction_map_pooled[experiment_name]
        if well_number not in well_index_map.keys():
            well_index_map[well_number] = []
            experiment_well_tick_index_map[well_number] = []

        well_map = experiment_prediction_map[experiment_name]
        well_map[well] = y_pred_current

        # Adding the current prediction to the list
        well_index_map[well_number].append(y_pred_current)
        experiment_well_tick_index_map[well_number].append(well)
        del X_metadata_current, y_pred_current, experiment_name, well_letter, well_number, well
    all_well_names.sort()
    all_experiment_names.sort()
    all_well_letters.sort()
    all_well_indices.sort()

    # Iterating over the experiment metadata so we can run the sigmoid evaluations
    for experiment_name in all_experiment_names:
        experiment_dir = out_dir + os.sep + experiment_name + os.sep
        os.makedirs(experiment_dir, exist_ok=True)

        # Extracting sigmoid score
        sigmoid_score: str = str(float('nan'))
        sigmoid_plot_estimations: np.ndarray = None
        sigmoid_plot_fit: np.ndarray = None
        if sigmoid_score_map is None:
            sigmoid_score = 'Not evaluated.'
        if experiment_name in sigmoid_score_map:
            sigmoid_score = str(sigmoid_score_map[experiment_name])
            sigmoid_plot_estimations = sigmoid_plot_estimation_map[experiment_name]
            sigmoid_plot_fit = sigmoid_plot_fit_map[experiment_name]
        else:
            sigmoid_score = 'Not available.'

        # writing to CSV
        out_csv = experiment_dir + os.sep + experiment_name + '-prediction_map' + file_name_suffix + '.csv'
        log.write('Saving prediction matrix to: ' + out_csv)
        f = open(out_csv, 'w')
        f.write(experiment_name + ' [' + str(sigmoid_score) + '];')
        [f.write(str(i) + ';') for i in all_well_indices]
        for w in all_well_letters:
            f.write('\n' + w)
            current_well = None
            for i in all_well_indices:
                f.write(';')
                current_well = w + str(i)
                if i < 10:
                    current_well = w + '0' + str(i)

                if current_well in experiment_prediction_map[experiment_name].keys():
                    r: float = experiment_prediction_map[experiment_name][current_well]
                    f.write(str(r))
            del w, i, current_well
        f.close()
        del f, out_csv

        ##########################################

        well_index_map: {} = experiment_prediction_map_pooled[experiment_name]
        well_indices = list(well_index_map.keys())
        prediction_entries = []
        prediction_ticks = []
        y_error = []
        well_indices.sort()
        for i in range(len(well_indices)):
            k = well_indices[i]
            predictions: [float] = well_index_map[k]
            predictions = np.array(predictions)
            prediction_entries.append(np.mean(predictions))
            ticks: [str] = experiment_well_tick_map[experiment_name][k]
            ticks.sort()
            ticks = str(ticks)
            ticks = ticks.replace("'", "").replace("]", "").replace("[", "")
            prediction_ticks.append(ticks)

            error = np.std(predictions, ddof=1) / np.sqrt(np.size(predictions))
            y_error.append(error)

        assert len(prediction_entries) == len(well_index_map.keys())
        assert len(y_error) == len(well_index_map.keys())
        assert len(prediction_ticks) == len(well_index_map.keys())

        plt.clf()
        plt.plot(well_indices, prediction_entries, color='blue')
        plt.errorbar(well_indices, prediction_entries, yerr=y_error, fmt='o', ecolor='orange', color='red')

        legend_entries = ['Mean Predictions']
        if sigmoid_plot_fit is not None:
            # Plotting the fitted coordinates, if they exist
            estimations_x = list(sigmoid_plot_fit[0])
            estimations_y = list(sigmoid_plot_fit[1])
            assert len(estimations_x) == len(estimations_y)

            plt.plot(estimations_x, estimations_y, color='lightblue')
            legend_entries.append('Sigmoid Curve Fit')
        elif sigmoid_plot_estimations is not None:
            # Plotting the estimations instead
            estimations_y = list(sigmoid_plot_estimations)
            estimations_x = [float(e + 1) / len(estimations_y) * (well_indices[-1] - well_indices[0]) + well_indices[0]
                             for e in range(len(estimations_y))]
            assert len(estimations_x) == len(estimations_y)

            estimations_x[0] = float(well_indices[0])
            plt.plot(estimations_x, estimations_y, color='lightblue')
            legend_entries.append('Sigmoid Curve Fit (Estimated)')

        title = 'Predictions: ' + experiment_name + ' ' + title_suffix
        if sigmoid_score_detail_map is not None and sigmoid_score_detail_map[experiment_name] is not None:
            sigmoid_score_detail = sigmoid_score_detail_map[experiment_name]
            title = title + '\n\nAsymptote-Score: ' + str(sigmoid_score_detail['AsympScore'])
            title = title + '\nEffect-Score: ' + str(sigmoid_score_detail['EffectScore'])
            title = title + '\nGradient-Score: ' + str(sigmoid_score_detail['GradientScore'])
            title = title + '\nResidual-Score: ' + str(sigmoid_score_detail['ResidualScore'])
            title = title + '\nRaw Score: ' + str(sigmoid_score_detail['rawScore'])
            title = title + '\n\nFinal Sigmoid Score: ' + sigmoid_score

            del sigmoid_score_detail
        else:
            title = title + '\nSigmoid Score: ' + sigmoid_score

        plt.title(title)
        plt.legend(legend_entries, loc='best')
        plt.ylabel('Prediction Score')
        plt.xlabel('Wells')
        plt.xticks(well_indices, prediction_ticks, rotation=90)

        plt.tight_layout()
        plt.autoscale()
        # Setting the ylim after the autoscale, to make sure the lim actually is applied
        plt.ylim([0.0, 1.05])
        ax = plt.gca()
        ax.set_ylim([0.0, 1.05])

        out_plot_name_base = experiment_dir + os.sep + experiment_name + '-predictions_concentration_response' + file_name_suffix
        plt.savefig(out_plot_name_base + '.png', dpi=dpi)
        plt.savefig(out_plot_name_base + '.svg', dpi=dpi, transparent=True)
        plt.savefig(out_plot_name_base + '.pdf', dpi=dpi)


def render_attention_histograms(out_dir: str, n_list: [np.ndarray], bins_list: [np.ndarray],
                                otsu_index_list: [np.ndarray], otsu_threshold_list: [np.ndarray],
                                entropy_attention_list: [np.ndarray], entropy_hist_list: [np.ndarray],
                                metadata_list: [TileMetadata], dpi: int = 300):
    os.makedirs(out_dir, exist_ok=True)
    assert len(otsu_index_list) == len(metadata_list)
    assert len(otsu_threshold_list) == len(metadata_list)
    assert len(entropy_attention_list) == len(metadata_list)
    assert len(entropy_hist_list) == len(metadata_list)
    assert len(n_list) == len(metadata_list)
    assert len(bins_list) == len(metadata_list)

    log.write('Rendering attention histograms to: ' + out_dir)
    print('')

    for n, bins, otsu_index, otsu_threshold, entropy_attention, entropy_hist, metadata, i in zip(n_list, bins_list,
                                                                                                 otsu_index_list,
                                                                                                 otsu_threshold_list,
                                                                                                 entropy_attention_list,
                                                                                                 entropy_hist_list,
                                                                                                 metadata_list,
                                                                                                 range(len(n_list))):
        exp_name = metadata.experiment_name
        well = metadata.get_formatted_well(long=True)

        line_print(str(i) + '/' + str(len(n_list)) + ': Rendering histogram for ' + exp_name + ' - ' + well)
        current_out_dir = out_dir + os.sep + exp_name + os.sep + well + os.sep
        os.makedirs(current_out_dir, exist_ok=True)

        render_attention_histogram(otsu_index=otsu_index, otsu_threshold=otsu_threshold, n=n, bins=bins,
                                   entropy_attention=entropy_attention, entropy_hist=entropy_hist, dpi=dpi,
                                   filename='attention-list-' + exp_name + '-' + well,
                                   title='Attention Histogram: ' + exp_name + ' - ' + well,
                                   file_formats=['.png', '.svg', '.pdf'], out_dir=current_out_dir)
        del otsu_index, otsu_threshold, entropy_attention, entropy_hist, metadata, exp_name, well, i, n, bins

    log.write('Finished rendering all histograms.')


def render_attention_histogram(n: np.ndarray, bins: np.ndarray, otsu_index: int, otsu_threshold: float,
                               entropy_attention: float, entropy_hist: float, title: str, filename: str,
                               out_dir: str, dpi: int = 400, file_formats: [str] = ['.png', '.svg', '.pdf']):
    os.makedirs(out_dir, exist_ok=True)
    assert len(n) == len(bins)
    width = min(bins[np.where(bins > 0)])

    plt.clf()
    plt.bar(x=bins, width=width, height=n)
    plt.axvline(x=otsu_threshold, color='orange')

    x_label = 'Attention'
    if bins.min() == 0.0 and bins.max() == 1.0:
        x_label = x_label + ' (Normalized)'

    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel(x_label)
    plt.legend(['Otsu Threshold: ' + str(otsu_threshold), 'Histogram (Entropy: ' + str(entropy_attention) + ')'],
               loc='best')

    plt.tight_layout()
    plt.autoscale()
    out_file = out_dir + os.sep + filename

    for f_format in file_formats:
        plt.savefig(out_file + f_format, dpi=dpi)


def render_bootstrapped_histograms(out_dir, bootstrap_list: [np.ndarray], bootstrap_threshold_indices_list: [int],
                                   metadata_list: [TileMetadata], n_replications: int, dpi: int = 600):
    ###
    # DEPRECATED
    ###

    os.makedirs(out_dir, exist_ok=True)
    assert len(bootstrap_list) == len(bootstrap_threshold_indices_list)
    assert len(bootstrap_list) == len(metadata_list)
    log.write('Writing bootstrapped histogram to: ' + out_dir)

    for (boostrap_result, threshold_index, metadata, i) in zip(bootstrap_list, bootstrap_threshold_indices_list,
                                                               metadata_list,
                                                               range(len(metadata_list))):
        current_out_dir = out_dir + os.sep + metadata.experiment_name + os.sep
        os.makedirs(current_out_dir, exist_ok=True)

        render_bootstrapped_histogram(boostrap_result=boostrap_result, threshold_index=threshold_index,
                                      metadata=metadata, dpi=dpi, out_dir=current_out_dir)


def render_bootstrapped_histogram(boostrap_result, threshold_index, metadata, out_dir, dpi: int = 600):
    ###
    # DEPRECATED
    ###

    x_entries = (np.asarray(list(range(len(boostrap_result))), dtype=np.float64) + 1) / float(len(boostrap_result))
    assert len(x_entries) == len(boostrap_result)
    threshold_normalized = float(threshold_index) / float(len(boostrap_result))

    plt.clf()
    plt.bar(x=x_entries, width=min(x_entries / 1.1337), height=boostrap_result)
    # plt.bar(height=x_entries, width=min(x_entries / 1.1337), x=boostrap_result)
    plt.axvline(x=threshold_normalized, color='orange')

    plt.title(metadata.experiment_name + '-' + metadata.get_formatted_well() + ' - Bootstrapped Histogram')
    plt.legend(['Otsu Threshold: ' + str(threshold_normalized), 'Histogram'], loc='best')

    plt.tight_layout()
    plt.autoscale()
    out_file = out_dir + metadata.experiment_name + '-' + metadata.get_formatted_well() + '-bootstrap-hist.png'
    plt.savefig(out_file, dpi=dpi)


def rgb_to_gray(img: np.ndarray, weights_r=0.299, weights_g=0.587, weights_b=0.114):
    gray_image = np.zeros(img.shape)
    r = np.array(img[:, :, 0])
    g = np.array(img[:, :, 1])
    b = np.array(img[:, :, 2])

    r = (r * weights_r)
    g = (g * weights_g)
    b = (b * weights_b)

    avg = (r + g + b)
    gray_image = img.copy()

    for i in range(3):
        gray_image[:, :, i] = avg

    return gray_image


if __name__ == '__main__':
    log.write('This function renders neurosphere data. Running this file does nothing.')
