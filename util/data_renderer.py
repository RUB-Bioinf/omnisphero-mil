from util import log
import numpy as np
import os

from util.utils import line_print
from util.well_metadata import TileMetadata
import matplotlib.pyplot as plt


def renderSpheres(X_raw: [np.ndarray], X_metadata: [TileMetadata], y_pred: [np.ndarray], y_pred_binary: [np.ndarray],
                  input_dim, y_attentions: [np.ndarray] = None, out_dir: str = None, colormap_name: str = 'jet',
                  dpi: int = 650, overlay_alpha: float = 0.65):
    os.makedirs(out_dir, exist_ok=True)

    assert input_dim[0] == 3
    tile_w = input_dim[1]
    tile_h = input_dim[2]
    color_map = plt.get_cmap(colormap_name)

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

        out_dir_current = out_dir + os.sep + experiment_name + os.sep + well_name + os.sep
        os.makedirs(out_dir_current, exist_ok=True)
        line_print('Rendering: ' + experiment_name + ' - ' + well_name, include_in_log=True)
        rendered_image = np.zeros((image_width, image_height, 3), dtype=np.uint8)

        for current_raw_tile, current_raw_metadata in zip(X_raw_current, X_metadata_current):
            # Checking if the metadata is correct
            assert current_raw_metadata.get_formatted_well(long=True) == well_name
            assert current_raw_metadata.experiment_name == experiment_name
            assert current_raw_tile.shape == (tile_w, tile_h, 3)
            current_raw_tile = current_raw_tile.astype(np.uint8)

            pos_x = current_raw_metadata.pos_x
            pos_y = current_raw_metadata.pos_y
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

            del pos_x, pos_y, attention, existing_attention, r, g, b, rgb, attention_color

        assert activation_map.min() == 0
        assert activation_map.max() == 1
        activation_map = activation_map * 255
        activation_map = activation_map.astype(np.uint8)
        activation_map_mask = activation_map_mask.astype(np.uint8)
        activation_map_mask = activation_map_mask * 255
        plt.imsave(out_dir_current + 'predicted_tiles_activation_map.jpg', activation_map)
        plt.imsave(out_dir_current + 'predicted_tiles_activation_map_mask.jpg', activation_map_mask)
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

    log.write('Finished rendering all attention overlays.')


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
