import math
import os
from typing import Union

import matplotlib.pyplot as plt
import mil_metrics
import nucleus_predictions
import numpy as np
from util import log
from util import utils
from util.utils import line_print
from util.well_metadata import TileMetadata


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

            # Getting the metadata location.
            # Since it's in float format we need to check for NaN entries and convert to int for coordinates.
            # assert not math.isnan(pos_x) and not math.isnan(pos_y)
            pos_x = current_raw_metadata.pos_x
            pos_y = current_raw_metadata.pos_y
            if current_raw_metadata.has_valid_position():
                pos_x = int(pos_x)
                pos_y = int(pos_y)

                # Debugging output
                # log.write('Rendered image size: ' + str(image_width) + 'x' + str(image_height))
                # log.write('Tile size: ' + str(tile_w) + 'x' + str(tile_h))
                # log.write('Tile insertion: ' + str(pos_y) + ':' + str(pos_y + tile_h) + ',' + str(pos_x) + ':' + str(
                #     pos_x + tile_w))
                rendered_image[pos_y:pos_y + tile_h, pos_x:pos_x + tile_w] = current_raw_tile
            else:
                log.write('Warning: No X/Y tile position for a tile in ' + current_raw_metadata.experiment_name
                          + ' - ' + current_raw_metadata.get_formatted_well())
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
            if not current_raw_metadata.has_valid_position():
                # If the metadata cannot be placed, nothing shall be displayed.
                continue

            pos_x = int(current_raw_metadata.pos_x)
            pos_y = int(current_raw_metadata.pos_y)
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

        # assert activation_map.min() == 0
        # assert activation_map.max() == 1
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
                           sigmoid_bmc30_map: {float} = None, sigmoid_score_detail_map: {} = None,
                           hide_na_metrics: bool = True, dpi: int = 650) -> [str]:
    # Remapping predictions so they can be evaluated
    experiment_prediction_map_pooled = {}
    experiment_prediction_map = {}
    experiment_well_tick_map = {}
    experiment_compound_metadata_map = {}
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
            experiment_compound_metadata_map[experiment_name] = metadata.plate_metadata
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

    all_render_out_dirs = []

    # Iterating over the experiment metadata so we can run the sigmoid evaluations
    for experiment_name in all_experiment_names:
        experiment_dir = out_dir + os.sep + experiment_name + os.sep
        os.makedirs(experiment_dir, exist_ok=True)

        plate_metadata = experiment_compound_metadata_map[experiment_name]
        bmc30_plate = float('NaN')
        bmc30_compound = float('NaN')
        bmc30_plate_text = '?'
        bmc30_compound_text = '?'
        bmc_out_text_content = ''

        if plate_metadata is not None:
            bmc30_plate_text = '{:0.2f}'.format(plate_metadata.plate_bmc30) + ' ' + utils.mu + 'M'
            bmc30_compound_text = '{:0.2f}'.format(plate_metadata.compound_bmc30) + ' ' + utils.mu + 'M'

            bmc30_plate = plate_metadata.interpolate_concentration_to_well(plate_metadata.plate_bmc30)
            bmc30_compound = plate_metadata.interpolate_concentration_to_well(plate_metadata.compound_bmc30)

        # Extracting sigmoid score
        sigmoid_score: str = str(float('nan'))
        sigmoid_plot_estimations: np.ndarray = None
        sigmoid_plot_fit: np.ndarray = None
        if sigmoid_score_map is None:
            sigmoid_score = 'Not evaluated.'
        else:
            if experiment_name in sigmoid_score_map:
                sigmoid_score = str(round(sigmoid_score_map[experiment_name], 4))
                sigmoid_plot_estimations = sigmoid_plot_estimation_map[experiment_name]
                sigmoid_plot_fit = sigmoid_plot_fit_map[experiment_name]

                # Saving fit TO CSV
                sigmoid_out_csv = experiment_dir + os.sep + experiment_name + '-sigmoid_fit' + file_name_suffix + '.csv'
                f = open(sigmoid_out_csv, 'w')
                f.write('i;x;y\n')
                if sigmoid_plot_fit is None:
                    f.write('No sigmoid fit.')
                else:
                    try:
                        [f.write(str(i) + ';' + str(sigmoid_plot_fit[0][i]) + ';' + str(sigmoid_plot_fit[1][i]) + '\n')
                         for i in range(len(sigmoid_plot_fit[0]))]
                    except Exception as e:
                        f.write('FATAL ERROR!\n')
                        f.write(str(e) + '\n')
                        f.write(str(e.__class__) + '\n')
                f.close()
            else:
                sigmoid_score = 'Not available.'

        # Extracting BMC30
        bmc30_prediction: float = float('NaN')
        bmc30_prediction_text: str = None
        bmc30_prediction_um = float('NaN')
        if sigmoid_bmc30_map is None:
            bmc30_prediction_text = 'BMC30 not available.'
        else:
            if experiment_name in sigmoid_bmc30_map:
                bmc30_prediction = sigmoid_bmc30_map[experiment_name]
                bmc30_prediction_text = str(bmc30_prediction)
                if np.isnan(bmc30_prediction):
                    bmc30_prediction_text = 'BMC30 failed.'
                else:
                    bmc30_prediction_text = str(round(bmc30_prediction, 3))
                    if plate_metadata is not None:
                        bmc30_prediction_um = plate_metadata.interpolate_well_index_to_concentration(bmc30_prediction)
                        bmc30_prediction_text = str(round(bmc30_prediction_um, 3)) + ' ' + utils.mu + 'M'
            else:
                bmc30_prediction_text = 'BMC30 not calculated.'

        # writing to CSV
        out_csv = experiment_dir + os.sep + experiment_name + '-prediction_map' + file_name_suffix + '.csv'
        log.write('Saving prediction matrix to: ' + out_csv)
        f = open(out_csv, 'w')
        f.write(experiment_name + ' [Score: ' + str(sigmoid_score) + ', BMC30 (Well): ' + str(
            bmc30_prediction) + ', BMC30 (uM): ' + str(bmc30_prediction_um) + '];')
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
        del f

        ##########################################

        well_index_map: {} = experiment_prediction_map_pooled[experiment_name]
        well_indices = list(well_index_map.keys())
        prediction_entries = []
        prediction_ticks = []
        y_error = []
        well_indices.sort()
        for i in range(len(well_indices)):
            # Getting the predicted value and adding it to the 'plot'
            k = well_indices[i]
            predictions: [float] = well_index_map[k]
            predictions = np.array(predictions)
            prediction_entries.append(np.mean(predictions))

            # Writing x axis ticks
            # Well names as ticks
            ticks: [str] = experiment_well_tick_map[experiment_name][k]
            ticks.sort()
            ticks = str(ticks)
            ticks = ticks.replace("'", "").replace("]", "").replace("[", "").replace("0", "")

            # Adding compound concentrations to ticks
            if plate_metadata is not None:
                concentration = plate_metadata.get_concentration_at(well_indices[i])
                concentration = '{:0.2f}'.format(concentration)
                ticks = ticks + '\n(' + concentration + ' ' + utils.mu + 'M)'

            # Storing the ticks for later
            prediction_ticks.append(ticks)

            # Error Bar using 'Standard error of the mean'
            # https://en.wikipedia.org/wiki/Standard_error
            sdm_error = np.std(predictions, ddof=1) / np.sqrt(np.size(predictions))
            y_error.append(sdm_error)
            del sdm_error

        # Writing the errors to CSV
        f = open(out_csv, 'a')
        f.write('\nSDM Error;;;')
        [f.write(str(i) + ';') for i in y_error]
        f.close()
        del f, out_csv

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

        # Plotting the BMC30
        plt.plot([bmc30_prediction, bmc30_prediction], [0, 1], color='lightgreen')
        legend_entries.append('BMC30 (pred.): ' + bmc30_prediction_text)
        bmc_out_text_content = bmc_out_text_content + 'BMC30 (pred.): ' + bmc30_prediction_text + ' at ' + str(
            bmc30_prediction) + '\n'

        if not math.isnan(bmc30_plate):
            plt.plot([bmc30_plate, bmc30_plate], [0, 1], color='green')
            legend_entries.append('BMC30 (plate): ' + bmc30_plate_text)
            bmc_out_text_content = bmc_out_text_content + 'BMC30 (plate.): ' + bmc30_plate_text + ' at ' + str(
                bmc30_plate) + '\n'

        if not math.isnan(bmc30_compound):
            plt.plot([bmc30_compound, bmc30_compound], [0, 1], color='darkgreen')
            legend_entries.append('BMC30 (compound): ' + bmc30_compound_text)
            bmc_out_text_content = bmc_out_text_content + 'BMC30 (compound.): ' + bmc30_compound_text + ' at ' + str(
                bmc30_compound) + '\n'

        title = 'Predictions: ' + experiment_name + ' ' + title_suffix
        title_tex = title
        if plate_metadata is not None:
            title = title + '\n' + plate_metadata.compound_name + ' [' + plate_metadata.compound_cas + ']'

        if sigmoid_score_detail_map is not None and sigmoid_score_detail_map[experiment_name] is not None:
            sigmoid_score_detail = sigmoid_score_detail_map[experiment_name]
            title = title + '\n\nAsymptote-Score: ' + str(round(sigmoid_score_detail['AsympScore'], 4))
            title = title + '\nEffect-Score: ' + str(round(sigmoid_score_detail['EffectScore'], 4))
            title = title + '\nGradient-Score: ' + str(round(sigmoid_score_detail['GradientScore'], 4))
            title = title + '\nResidual-Score: ' + str(round(sigmoid_score_detail['ResidualScore'], 4))
            title = title + '\nRaw Score: ' + str(round(sigmoid_score_detail['rawScore'], 4))
            title = title + '\n\nFinal Sigmoid Score: ' + sigmoid_score

            del sigmoid_score_detail
        else:
            if hide_na_metrics:
                title = title + '\nSigmoid Score: ' + sigmoid_score
            else:
                title = title + '\n\nAsymptote-Score: N/A'
                title = title + '\nEffect-Score: N/A'
                title = title + '\nGradient-Score: N/A'
                title = title + '\nResidual-Score: N/A'
                title = title + '\nRaw Score: N/A'
                title = title + '\n\nFinal Sigmoid Score: ' + sigmoid_score

        # Setting the legend (should be done first)
        plt.legend(legend_entries, loc='best')

        # Setting the x label
        x_label_text = 'Wells'
        if plate_metadata is not None:
            x_label_text = x_label_text + ' / Concentration (' + utils.mu + 'M)'

        # Setting plot texts and labels
        plt.title(title)
        plt.ylabel('Prediction Score')
        plt.xlabel(x_label_text)
        plt.xticks(well_indices, prediction_ticks, rotation=90)

        plt.tight_layout()
        plt.autoscale()
        # Setting the ylim after the autoscale, to make sure the lim actually is applied
        plt.ylim([0.0, 1.05])
        ax = plt.gca()
        ax.set_ylim([0.0, 1.05])

        # making note of this out dir
        if experiment_dir not in all_render_out_dirs:
            all_render_out_dirs.append(experiment_dir)

        # Rendering the images
        out_plot_name_base = experiment_dir + os.sep + experiment_name + '-predictions_concentration_response' + file_name_suffix
        plt.savefig(out_plot_name_base + '.png', dpi=dpi)
        plt.savefig(out_plot_name_base + '.svg', dpi=dpi, transparent=True)
        plt.savefig(out_plot_name_base + '.pdf', dpi=dpi)

        # Writing the BMCs:
        f = open(out_plot_name_base + '-bmc.txt', 'w')
        f.write(bmc_out_text_content.replace(utils.mu, 'u'))
        f.close()

        # Writing as .tex
        data_list_y = [prediction_entries]
        data_list_x = [well_indices]
        tikz_colors = ['blue']
        legend_entries_tex = ['Mean Predictions']
        if sigmoid_plot_fit is not None:
            data_list_y.append(sigmoid_plot_fit[1])
            data_list_x.append(sigmoid_plot_fit[0])

            tikz_colors.append('cyan')
            legend_entries_tex.append('Curve fit')

        all_well_indices = list(range(min(well_indices), max(well_indices), 1))
        tikz = utils.get_plt_as_tex(data_list_y=data_list_y, data_list_x=data_list_x, plot_colors=tikz_colors,
                                    title=title_tex.replace('\n', ' - ').replace('  ', ' ').replace('_', '-').replace(
                                        utils.mu, 'm'),
                                    plot_titles=legend_entries_tex,
                                    max_y=1.15, min_y=0.0,
                                    max_x=max(well_indices), min_x=min(well_indices),
                                    tick_count_x=len(all_well_indices),
                                    legend_pos='south west',
                                    label_y='Predictions', label_x='Well Index')
        f = open(out_plot_name_base + '.tex', 'w')
        f.write(tikz)
        f.close()

    # Rending done. Returning the newly created directories
    return all_render_out_dirs


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
    os.makedirs(out_dir, exist_ok=True)
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
    assert weights_r + weights_g + weights_b < 1.1
    # so, there would be a check of the sum to be equal to 1.0. But we all know floating points....

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


def render_attention_cytometry_prediction_distributions_partitioned(out_dir: str,
                                                                    X_metadatas: [[TileMetadata]],
                                                                    partitions: Union[int, list],
                                                                    y_preds: [float],
                                                                    all_attentions: [np.ndarray],
                                                                    title_suffix: str = None,
                                                                    filename_suffix: str = None,
                                                                    include_oligo: bool = False,
                                                                    include_neuron: bool = False,
                                                                    include_nucleus: bool = False,
                                                                    scatter_plot_item_scale: float = 20.0,
                                                                    dpi: int = 400) -> [str]:
    if filename_suffix is None:
        filename_suffix = ''

    if isinstance(partitions, list):
        for partition in partitions:
            # Calling itself so it can process the list
            render_attention_cytometry_prediction_distributions_partitioned(out_dir=out_dir,
                                                                            X_metadatas=X_metadatas,
                                                                            partitions=partition,
                                                                            y_preds=y_preds,
                                                                            all_attentions=all_attentions,
                                                                            title_suffix=title_suffix,
                                                                            filename_suffix=filename_suffix,
                                                                            include_oligo=include_oligo,
                                                                            include_neuron=include_neuron,
                                                                            include_nucleus=include_nucleus,
                                                                            scatter_plot_item_scale=scatter_plot_item_scale,
                                                                            dpi=dpi,
                                                                            )
        return
    partitions = int(partitions)

    assert partitions % 2 == 0  # only allowing partitions to be divisible by 2
    assert include_oligo or include_neuron or include_nucleus
    assert len(y_preds) == len(X_metadatas)
    assert len(all_attentions) == len(X_metadatas)
    assert partitions > 0
    scatter_plot_item_scale = float(scatter_plot_item_scale)
    assert scatter_plot_item_scale > 0

    os.makedirs(out_dir, exist_ok=True)
    log.write('Writing partitioned Cytometry predictions to: ' + out_dir)
    log.write('Number of partitions: ' + str(partitions))

    # Sorting out negative bags
    X_metadatas_positive = []
    y_preds_positive = []
    all_attentions_positive = []
    for (X_metadata, y_pred, all_attention) in zip(X_metadatas, y_preds, all_attentions):
        if y_pred == 1.0:
            X_metadatas_positive.append(X_metadata)
            y_preds_positive.append(y_pred)
            all_attentions_positive.append(all_attention)

        del X_metadata, y_pred, all_attention

    # Running render for all positive bags
    partitions_step = 1.0 / float(partitions)
    for (X_metadata, y_pred, all_attention) in zip(X_metadatas_positive, y_preds_positive, all_attentions_positive):
        assert len(X_metadata) == len(all_attention)
        experiment_name = X_metadata[0].experiment_name
        current_well = X_metadata[0].get_formatted_well()

        out_dir_current = out_dir + os.sep + 'cytometry_distribution-partitions-' + str(partitions) + os.sep
        out_dir_current_detailed = out_dir_current + 'detailed' + os.sep
        os.makedirs(out_dir_current_detailed, exist_ok=True)
        out_dirs_image_paths_detailed = []
        whole_plate_image_path_detailed = []
        for i in range(partitions):
            partition_range_lower = 1.0 * float(i) * partitions_step
            partition_range_upper = partition_range_lower + partitions_step
            print('Partition range lower: ' + str(partition_range_lower))
            print('Partition range upper: ' + str(partition_range_upper))

            # attentions: np.ndarray = all_attention[i]
            attentions_min = float(min(all_attention))
            attentions_max = float(max(all_attention))
            normalized_attention = (all_attention - attentions_min) / (attentions_max - attentions_min)

            all_attention_partitioned = []
            X_metadata_partitioned = []
            for j in range(len(normalized_attention)):
                attention_normalized = float(normalized_attention[j])
                if partition_range_lower < attention_normalized < partition_range_upper:
                    all_attention_partitioned.append(attention_normalized)
                    X_metadata_partitioned.append(X_metadata[j])

                del attention_normalized

            # attention and metadata are now partitioned
            assert len(all_attention_partitioned) > 0
            assert len(X_metadata_partitioned) > 0

            # Rendering them now
            out_dirs_image_paths, whole_plate_image_path = render_attention_cytometry_prediction_distributions(
                out_dir=out_dir_current_detailed,
                X_metadata=[X_metadata_partitioned],
                y_preds=[y_pred],
                all_attentions=[all_attention_partitioned],
                filename_suffix=filename_suffix + '_partition-' + str(i),
                title_suffix=title_suffix + '\n[' + str(partition_range_lower) + '% - ' + str(
                    partition_range_upper) + '%]',
                include_oligo=include_oligo,
                include_neuron=include_neuron,
                include_nucleus=include_nucleus,
                include_plate=False,
                scatter_plot_item_scale=scatter_plot_item_scale,
                dpi=dpi)

            assert len(out_dirs_image_paths) == 1
            out_dirs_image_paths_detailed.append(out_dirs_image_paths[0])
            whole_plate_image_path_detailed.append(whole_plate_image_path)
            del out_dirs_image_paths
            del whole_plate_image_path

        # Rendering composite images
        out_dirs_images_detailed = read_and_composite_images(out_dirs_image_paths_detailed, light_mode=True)
        out_dir_base = out_dir_current + os.sep + experiment_name + os.sep
        os.makedirs(out_dir_base, exist_ok=True)
        out_name_base = out_dir_base + os.sep + 'cytometry_' + experiment_name + '-' + current_well + filename_suffix + '-partitioned'

        # Saving the images
        plt.imsave(out_name_base + '.png', out_dirs_images_detailed)


def read_and_composite_images(image_paths: [str], light_mode: bool = True):
    # Checking if exists
    for path in image_paths:
        assert os.path.exists(path)
        del path

    # Reading paths
    images = []
    for path in image_paths:
        img = plt.imread(path)
        img = img[:, :, :3] * 255
        img = img.astype(np.uint8)
        images.append(img)
        del img

    fused_images = mil_metrics.fuse_image_tiles(images=images, light_mode=light_mode)
    return fused_images


def render_attention_instance_range(out_dir: str, X_metadata: [[TileMetadata]],
                                    y_preds: [float],
                                    all_attentions: [np.ndarray],
                                    X_raw,
                                    render_attention_instance_range_min: float,
                                    render_attention_instance_range_max: float,
                                    dpi: int = 400):
    render_attention_instance_range_min = float(render_attention_instance_range_min)
    render_attention_instance_range_max = float(render_attention_instance_range_max)
    assert render_attention_instance_range_min <= render_attention_instance_range_max
    os.makedirs(out_dir, exist_ok=True)

    assert len(y_preds) == len(all_attentions)
    assert len(y_preds) == len(X_raw)
    assert len(y_preds) == len(X_metadata)

    print('')
    for i in range(len(y_preds)):
        all_attentions_current = all_attentions[i]
        X_raw_current = X_raw[i]
        X_metadata_current = X_metadata[i]
        y_preds_current = y_preds[i]
        assert len(X_raw_current) == len(all_attentions_current)
        assert len(X_raw_current) == len(X_metadata_current)

        metadata = X_metadata_current[0]
        current_experiment_name = metadata.experiment_name
        current_well = metadata.get_formatted_well()
        out_dir_current = out_dir + current_experiment_name + os.sep
        out_dir_current_detail = out_dir_current + 'detail' + os.sep
        os.makedirs(out_dir_current, exist_ok=True)
        os.makedirs(out_dir_current_detail, exist_ok=True)
        line_print(str(i + 1) + '/' + str(
            len(y_preds)) + ': Rendering attentions for ' + current_experiment_name + ' - ' + current_well)

        # extracting the samples that are within normalized attention range
        matching_samples = []
        out_image_localized_raw = np.zeros((metadata.well_image_height, metadata.well_image_width, 3), dtype=np.uint8)
        for j in range(len(X_raw_current)):
            attention_sample = all_attentions_current[j]
            X_metadata_sample = X_metadata_current[j]
            X_raw_sample = X_raw_current[j]

            attention_sample_normalized = (attention_sample - all_attentions_current.min()) / (
                    all_attentions_current.max() - all_attentions_current.min())
            attention_sample_normalized = float(attention_sample_normalized)

            if render_attention_instance_range_min <= attention_sample_normalized <= render_attention_instance_range_max:
                X_raw_sample = np.copy(X_raw_sample)
                X_raw_sample = X_raw_sample.copy()
                matching_samples.append(X_raw_sample)

                width, height, _ = X_raw_sample.shape
                pos_x: int = int(X_metadata_sample.pos_x)
                pos_y: int = int(X_metadata_sample.pos_y)
                out_image_localized_raw[pos_y:pos_y + height, pos_x:pos_x + width] = X_raw_sample

            del attention_sample, X_metadata_sample, X_raw_sample, attention_sample_normalized
            del j

        # Rendering the matching samples, if they exist
        if len(matching_samples) > 0:
            for j in range(len(matching_samples)):
                sample = matching_samples[j]
                detail_image_name = out_dir_current_detail + os.sep + current_experiment_name + '-' + current_well + '-' + str(
                    j) + '.png'
                plt.imsave(detail_image_name, sample)
                matching_samples[j] = mil_metrics.outline_rgb_array(sample, None, None, bright_mode=True,
                                                                    override_colormap=[255, 255, 255])

            matching_samples_fused = mil_metrics.fuse_image_tiles(images=matching_samples, light_mode=False)
            fused_image_name = out_dir_current + os.sep + current_experiment_name + '-' + current_well + '_fused.png'
            fused_image_name_detail = out_dir_current + os.sep + current_experiment_name + '-' + current_well + '_fused-detail'
            plt.imsave(fused_image_name, matching_samples_fused)

            localized_image_name = out_dir_current + os.sep + current_experiment_name + '-' + current_well + '_localized.png'
            plt.imsave(localized_image_name, out_image_localized_raw)

            plt.clf()
            plt.imshow(out_image_localized_raw)
            plt.title(current_experiment_name + ' - ' + current_well + '\nNormalized Attention Samples: ' + str(
                render_attention_instance_range_min) + ' - ' + str(render_attention_instance_range_max))
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.autoscale()
            plt.savefig(fused_image_name_detail + '.png', dpi=dpi)
            plt.savefig(fused_image_name_detail + '.svg', dpi=dpi, transparent=True)
            plt.savefig(fused_image_name_detail + '.pdf', dpi=dpi)

        # cleanup
        del matching_samples
        del i
        del all_attentions_current, X_raw_current, X_metadata_current, y_preds_current


def render_attention_cytometry_prediction_distributions(out_dir: str, X_metadata: [[TileMetadata]],
                                                        y_preds: [float],
                                                        all_attentions: [np.ndarray],
                                                        title_suffix: str = None, filename_suffix: str = None,
                                                        include_oligo: bool = False,
                                                        include_neuron: bool = False,
                                                        include_nucleus: bool = False,
                                                        include_plate: bool = True,
                                                        scatter_plot_item_scale: float = 20.0,
                                                        dpi: int = 400) -> ([str], str):
    assert include_oligo or include_neuron or include_nucleus
    assert len(y_preds) == len(X_metadata)
    assert len(all_attentions) == len(X_metadata)
    scatter_plot_item_scale = float(scatter_plot_item_scale)
    assert scatter_plot_item_scale > 0

    os.makedirs(out_dir, exist_ok=True)
    log.write('Writing Cytometry predictions to: ' + out_dir)
    print('')

    if title_suffix is None:
        title_suffix = ''
    if filename_suffix is None:
        filename_suffix = ''

    num_entries: int = 0
    if include_oligo:
        num_entries = num_entries + 1
    if include_neuron:
        num_entries = num_entries + 1
    if include_nucleus:
        num_entries = num_entries + 1

    alpha = 1.0
    alpha = alpha / float(num_entries)
    assert num_entries > 0

    experiment_names = []
    plates_oligo_counts = {}
    plates_neuron_counts = {}
    plates_nuclei_counts = {}
    plates_plot_x = {}
    plates_plot_y_oligos = {}
    plates_plot_y_neurons = {}
    plates_plot_y_nuclei = {}
    plates_plot_s_neurons = {}
    plates_plot_s_oligos = {}
    plates_plot_s_nuclei = {}
    plates_oligo_count = {}
    plates_neuron_count = {}
    plates_nuclei_count = {}
    plates_out_dir = {}

    out_dirs_image_paths: [str] = []
    whole_plate_image_path: str = None
    for i in range(len(y_preds)):
        # pre-processing attention metric
        attentions: np.ndarray = all_attentions[i]
        attentions_min = float(min(attentions))
        attentions_max = float(max(attentions))

        y_pred: float = y_preds[i]
        meta_datas: [TileMetadata] = X_metadata[i]
        assert len(attentions) == len(meta_datas)

        formatted_well = meta_datas[0].get_formatted_well()
        experiment_name = meta_datas[0].experiment_name
        utils.line_print(str(i + 1) + '/' + str(len(y_preds)) + ' - ' + formatted_well)
        out_dir_current = out_dir + os.sep + experiment_name + os.sep + 'cytometry-distributions' + os.sep
        os.makedirs(out_dir_current, exist_ok=True)

        # Setting up for later when the whole plate will be used
        if experiment_name in experiment_names:
            plate_oligo_counts = plates_oligo_counts[experiment_name]
            plate_neuron_counts = plates_neuron_counts[experiment_name]
            plate_nuclei_counts = plates_nuclei_counts[experiment_name]
            plate_plot_x = plates_plot_x[experiment_name]
            plate_plot_y_oligos = plates_plot_y_oligos[experiment_name]
            plate_plot_y_neurons = plates_plot_y_neurons[experiment_name]
            plate_plot_y_nuclei = plates_plot_y_nuclei[experiment_name]
            plate_plot_s_neurons = plates_plot_s_neurons[experiment_name]
            plate_plot_s_oligos = plates_plot_s_oligos[experiment_name]
            plate_plot_s_nuclei = plates_plot_s_nuclei[experiment_name]
            plate_oligo_count = plates_oligo_count[experiment_name]
            plate_neuron_count = plates_neuron_count[experiment_name]
            plate_nuclei_count = plates_nuclei_count[experiment_name]
        else:
            experiment_names.append(experiment_name)
            plate_oligo_counts = []
            plate_neuron_counts = []
            plate_nuclei_counts = []
            plate_plot_x = []
            plate_plot_y_oligos = []
            plate_plot_y_neurons = []
            plate_plot_y_nuclei = []
            plate_plot_s_neurons = []
            plate_plot_s_oligos = []
            plate_plot_s_nuclei = []
            plate_oligo_count = 0
            plate_neuron_count = 0
            plate_nuclei_count = 0

        # checking the overall cell counts
        oligo_counts = []
        neuron_counts = []
        nuclei_counts = []
        for metadata in meta_datas:
            oligo_count = [m.count_oligos for m in meta_datas].count(metadata.count_oligos)
            oligo_counts.append(oligo_count)
            plate_oligo_counts.append(oligo_count)
            neuron_count = [m.count_neurons for m in meta_datas].count(metadata.count_neurons)
            neuron_counts.append(neuron_count)
            plate_neuron_counts.append(neuron_count)
            nuclei_count = [m.count_nuclei for m in meta_datas].count(metadata.count_nuclei)
            nuclei_counts.append(nuclei_count)
            plate_nuclei_counts.append(nuclei_count)
            del oligo_count, neuron_count, nuclei_count
        oligo_counts.sort()
        neuron_counts.sort()
        nuclei_counts.sort()

        oligo_counts = list(dict.fromkeys(oligo_counts))
        neuron_counts = list(dict.fromkeys(neuron_counts))
        nuclei_counts = list(dict.fromkeys(nuclei_counts))

        oligo_counts_min = float(min(oligo_counts))
        neuron_counts_min = float(min(neuron_counts))
        nuclei_counts_min = float(min(nuclei_counts))
        oligo_counts_max = float(max(oligo_counts))
        neuron_counts_max = float(max(neuron_counts))
        nuclei_counts_max = float(max(nuclei_counts))

        # Counting the cells
        oligo_count = sum([m.count_oligos for m in meta_datas])
        neuron_count = sum([m.count_neurons for m in meta_datas])
        nuclei_count = sum([m.count_nuclei for m in meta_datas])
        plate_oligo_count = plate_oligo_count + oligo_count
        plate_neuron_count = plate_neuron_count + neuron_count
        plate_nuclei_count = plate_nuclei_count + nuclei_count

        title = 'Cytometry Distributions: ' + experiment_name + ' - ' + formatted_well + ' ' + title_suffix.strip()
        title = title.strip()
        title = title + '\nPredicted Label: ' + str(y_pred)

        plot_x = []
        plot_y_oligos = []
        plot_y_neurons = []
        plot_y_nuclei = []
        plot_s_oligos = []
        plot_s_neurons = []
        plot_s_nuclei = []
        for (metadata, attention) in zip(meta_datas, attentions):
            attention = float(attention)
            normalized_attention = (attention - attentions_min) / (attentions_max - attentions_min)

            # Counting how many times a cell has been counted
            oligo_count_count = [m.count_oligos for m in meta_datas].count(metadata.count_oligos)
            neuron_count_count = [m.count_neurons for m in meta_datas].count(metadata.count_neurons)
            nuclei_count_count = [m.count_nuclei for m in meta_datas].count(metadata.count_nuclei)
            normalized_oligo_count = (float(oligo_count_count) - oligo_counts_min) / (
                    oligo_counts_max - oligo_counts_min)
            normalized_neuron_count = (float(neuron_count_count) - neuron_counts_min) / (
                    neuron_counts_max - neuron_counts_min)
            normalized_nuclei_count = (float(nuclei_count_count) - nuclei_counts_min) / (
                    nuclei_counts_max - nuclei_counts_min)

            plot_x.append(normalized_attention)
            plot_y_oligos.append(metadata.count_oligos)
            plot_y_neurons.append(metadata.count_neurons)
            plot_y_nuclei.append(metadata.count_nuclei)

            plate_plot_x.append(normalized_attention)
            plate_plot_y_oligos.append(metadata.count_oligos)
            plate_plot_y_neurons.append(metadata.count_neurons)
            plate_plot_y_nuclei.append(metadata.count_nuclei)

            plot_s_neurons.append(normalized_neuron_count * scatter_plot_item_scale)
            plot_s_oligos.append(normalized_oligo_count * scatter_plot_item_scale)
            plot_s_nuclei.append(normalized_nuclei_count * scatter_plot_item_scale)
            plate_plot_s_neurons.append(normalized_neuron_count * scatter_plot_item_scale)
            plate_plot_s_oligos.append(normalized_oligo_count * scatter_plot_item_scale)
            plate_plot_s_nuclei.append(normalized_nuclei_count * scatter_plot_item_scale)
            del metadata, attention

        plt.clf()
        plt.title(title)
        legend_entries = []
        if include_oligo:
            plt.scatter(x=plot_x, y=plot_y_oligos, s=plot_s_oligos, color='green', alpha=alpha)
            legend_entries.append('Oligodendrocytes [' + str(oligo_count) + ']')
        if include_neuron:
            plt.scatter(x=plot_x, y=plot_y_neurons, s=plot_s_neurons, color='blue', alpha=alpha)
            legend_entries.append('Neurons [' + str(neuron_count) + ']')
        if include_nucleus:
            plt.scatter(x=plot_x, y=plot_y_nuclei, s=plot_s_nuclei, color='red', alpha=alpha)
            legend_entries.append('Nuclei [' + str(nuclei_count) + ']')

        plt.legend(legend_entries, loc='best')
        out_name = out_dir_current + 'cytometry_' + experiment_name + '-' + formatted_well + filename_suffix
        out_name = out_name.strip()

        plt.ylabel('Cell Count\nSamples: ' + str(len(attentions)))
        plt.xlabel('Attention (Normalized)')

        plt.tight_layout()
        plt.autoscale()
        plt.savefig(out_name + '.png', dpi=dpi)
        plt.savefig(out_name + '.pdf', dpi=dpi)
        plt.savefig(out_name + '.svg', dpi=dpi)
        out_dirs_image_paths.append(out_name + '.png')
        # TODO save as .csv

        # Adding to the plate dicts
        plates_oligo_counts[experiment_name] = plate_oligo_counts
        plates_neuron_counts[experiment_name] = plate_neuron_counts
        plates_nuclei_counts[experiment_name] = plate_nuclei_counts
        plates_plot_x[experiment_name] = plate_plot_x
        plates_plot_y_oligos[experiment_name] = plate_plot_y_oligos
        plates_plot_y_neurons[experiment_name] = plate_plot_y_neurons
        plates_plot_y_nuclei[experiment_name] = plate_plot_y_nuclei
        plates_plot_s_neurons[experiment_name] = plate_plot_s_neurons
        plates_plot_s_oligos[experiment_name] = plate_plot_s_oligos
        plates_plot_s_nuclei[experiment_name] = plate_plot_s_nuclei
        plates_oligo_count[experiment_name] = plate_oligo_count
        plates_neuron_count[experiment_name] = plate_neuron_count
        plates_nuclei_count[experiment_name] = plate_nuclei_count
        plates_out_dir[experiment_name] = out_dir_current
        del experiment_name, title, legend_entries, out_dir_current
        del oligo_counts
        del neuron_counts
        del nuclei_counts

        del oligo_counts_min
        del neuron_counts_min
        del nuclei_counts_min
        del oligo_counts_max
        del neuron_counts_max
        del nuclei_counts_max

    print('')
    # Also writing a combined 'plate' file
    if include_plate:
        log.write('Writing whole plate plots:')

        for i in range(len(experiment_names)):
            experiment_name = experiment_names[i]
            utils.line_print(str(i + 1) + '/' + str(len(experiment_names)) + ' - ' + experiment_name)
            title = 'Cytometry Distributions: ' + experiment_name + ' - Whole Plate ' + title_suffix
            title = title.strip()

            # extracting the well infos
            plate_oligo_counts = plates_oligo_counts[experiment_name]
            plate_neuron_counts = plates_neuron_counts[experiment_name]
            plate_nuclei_counts = plates_nuclei_counts[experiment_name]
            plate_plot_x = plates_plot_x[experiment_name]
            plate_plot_y_oligos = plates_plot_y_oligos[experiment_name]
            plate_plot_y_neurons = plates_plot_y_neurons[experiment_name]
            plate_plot_y_nuclei = plates_plot_y_nuclei[experiment_name]
            plate_plot_s_neurons = plates_plot_s_neurons[experiment_name]
            plate_plot_s_oligos = plates_plot_s_oligos[experiment_name]
            plate_plot_s_nuclei = plates_plot_s_nuclei[experiment_name]
            plate_oligo_count = plates_oligo_count[experiment_name]
            plate_neuron_count = plates_neuron_count[experiment_name]
            plate_nuclei_count = plates_nuclei_count[experiment_name]
            out_dir_current = plates_out_dir[experiment_name]

            plate_oligo_counts.sort()
            plate_neuron_counts.sort()
            plate_nuclei_counts.sort()

            plate_oligo_counts = list(dict.fromkeys(plate_oligo_counts))
            plate_neuron_counts = list(dict.fromkeys(plate_neuron_counts))
            plate_nuclei_counts = list(dict.fromkeys(plate_nuclei_counts))

            plate_oligo_counts_min = float(min(plate_oligo_counts))
            plate_neuron_counts_min = float(min(plate_neuron_counts))
            plate_nuclei_counts_min = float(min(plate_nuclei_counts))
            plate_oligo_counts_max = float(max(plate_oligo_counts))
            plate_neuron_counts_max = float(max(plate_neuron_counts))
            plate_nuclei_counts_max = float(max(plate_nuclei_counts))

            # Finally plotting all this stuff
            os.makedirs(out_dir_current, exist_ok=True)
            plt.clf()
            plt.title(title)
            legend_entries = []
            if include_oligo:
                plt.scatter(x=plate_plot_x, y=plate_plot_y_oligos, s=plate_plot_s_oligos, color='green', alpha=alpha)
                legend_entries.append('Oligodendrocytes [' + str(plate_oligo_count) + ']')
            if include_neuron:
                plt.scatter(x=plate_plot_x, y=plate_plot_y_neurons, s=plate_plot_s_neurons, color='blue', alpha=alpha)
                legend_entries.append('Neurons [' + str(plate_neuron_count) + ']')
            if include_nucleus:
                plt.scatter(x=plate_plot_x, y=plate_plot_y_nuclei, s=plate_plot_s_nuclei, color='red', alpha=alpha)
                legend_entries.append('Nuclei [' + str(plate_nuclei_count) + ']')

            plt.legend(legend_entries, loc='best')
            out_name = out_dir_current + 'cytometry_' + experiment_name + '-plate' + filename_suffix

            plt.ylabel('Cell Count\nSamples: ' + str(len(plate_plot_y_nuclei)))
            plt.xlabel('Attention (Normalized)')

            plt.tight_layout()
            plt.autoscale()
            plt.savefig(out_name + '.png', dpi=float(dpi) * 1.337)
            plt.savefig(out_name + '.pdf', dpi=float(dpi) * 1.337)
            plt.savefig(out_name + '.svg', dpi=float(dpi) * 1.337)
            whole_plate_image_path = out_name + '.png'
            # TODO save as .csv

    log.write('Images saved (Count: ' + str(len(out_dirs_image_paths)) + ')')
    print('')
    for image in out_dirs_image_paths:
        utils.line_print(str(image))
    log.write('\nDone.')

    return out_dirs_image_paths, whole_plate_image_path


def render_attention_cell_distributions(out_dir: str, distributions, X_metadata: [[TileMetadata]], alpha: float = 0.3,
                                        dpi: int = 600,
                                        include_neuron: bool = True,
                                        include_oligo: bool = True,
                                        include_nucleus: bool = True,
                                        filename_suffix: str = None, title_suffix: str = None
                                        ):
    assert len(distributions) == len(X_metadata)
    os.makedirs(out_dir, exist_ok=True)

    if filename_suffix is None:
        filename_suffix = ''
    if title_suffix is None:
        title_suffix = ''

    for (current_distributions, current_metadata) in zip(distributions, X_metadata):
        attention_min = current_distributions['attention_min']
        attention_max = current_distributions['attention_max']
        neuron_count = current_distributions['neuron']
        oligo_count = current_distributions['oligo']
        nucleus_count = current_distributions['nucleus']

        assert len(neuron_count) == len(current_metadata)
        assert len(oligo_count) == len(current_metadata)
        assert len(nucleus_count) == len(current_metadata)

        experiment_name = current_metadata[0].experiment_name
        well_name = current_metadata[0].get_formatted_well()
        log.write('Saving cell / attention density for: ' + experiment_name + '-' + well_name)

        current_out_dir = out_dir + os.sep + experiment_name + os.sep + well_name
        os.makedirs(current_out_dir, exist_ok=True)

        y = []
        x_neuron = []
        x_oligo = []
        x_nucleus = []
        for i in range(len(current_metadata)):
            if not math.isnan(neuron_count[i][0]):
                assert neuron_count[i][0] == oligo_count[i][0]
                assert neuron_count[i][0] == nucleus_count[i][0]
                y.append(neuron_count[i][0])
                x_neuron.append(neuron_count[i][1])
                x_oligo.append(oligo_count[i][1])
                x_nucleus.append(nucleus_count[i][1])

        plt.clf()
        if include_neuron:
            plt.scatter(x=y, y=x_neuron, c='red', alpha=alpha, label='Neurons')
        if include_oligo:
            plt.scatter(x=y, y=x_oligo, c='green', alpha=alpha, label='Oligodendrocytes')
        if include_nucleus:
            plt.scatter(x=y, y=x_nucleus, c='blue', alpha=alpha, label='Nuclei')

        plt.xlabel('Attention (' + str(len(current_metadata)) + ' tiles)')
        plt.ylabel('Cell Count')
        plt.title('Attention Nuclei Density: ' + experiment_name + '-' + well_name + title_suffix)
        plt.legend(loc='best')

        out_filename = current_out_dir + os.sep + experiment_name + '-' + well_name + '-attention_nuclei_density' + filename_suffix
        plt.savefig(out_filename + '.png', dpi=dpi)
        plt.savefig(out_filename + '.svg', dpi=dpi)
        plt.savefig(out_filename + '.pdf', dpi=dpi)


if __name__ == '__main__':
    log.write('This function renders neurosphere data. Running this file does nothing.')
