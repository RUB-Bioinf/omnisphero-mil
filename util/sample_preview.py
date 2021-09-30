import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import mil_metrics

import loader
from util import log
from util.utils import line_print


def save_normalized_rgb(img: np.ndarray, filename: str):
    img = img * 255
    img = img.astype(np.uint8)
    save_rgb(img=img, filename=filename)


def save_rgb(img: np.ndarray, filename: str):
    img = img.astype(np.uint8)

    try:
        plt.imsave(filename, img)
    except Exception as e:
        # TODO display stacktrace
        error_text_file = filename + '-error.txt'
        preview_error_text = str(e.__class__.__name__) + ': "' + str(e) + '"'
        log.write(preview_error_text)

        try:
            f = open(error_text_file, 'w')
            f.write(preview_error_text)
            f.close()
        except Exception as e2:
            log.write("Failed to save error-text to file: '" + error_text_file + "'! -> " + e2.__class__.__name__)
            traceback.print_exc()


def save_z_scored_image(img: np.ndarray, filename: str, normalize_enum: int, min: float = -3.0, dim_x: int = 150,
                        dim_y: int = 150, max: float = 3.0, dpi: int = 250, fig_titles: [str] = ['r', 'g', 'b']):
    x = np.arange(0, dim_x, 1)
    y = np.arange(0, dim_y, 1)
    j = cm.get_cmap('jet')
    X, Y = np.meshgrid(x, y)

    plt.clf()
    for i in range(3):
        current_channel = img[:, :, i]

        plt.subplot(1, 3, i + 1, adjustable='box', aspect=1)
        plt.pcolor(X, Y, current_channel, cmap=j, vmin=min, vmax=max)
        plt.xlabel('px')
        plt.ylabel('px')
        plt.title(fig_titles[i])

        c_bar = plt.colorbar()
        if i == 2:
            c_bar.ax.set_ylabel('z score', rotation=270)

    plt.suptitle(loader.normalize_enum_descriptions[normalize_enum])
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)


def save_hnm_bags(out_dir: str, new_bags: [np.ndarray], new_bags_raw: [np.ndarray], new_bag_names: [str]):
    os.makedirs(out_dir, exist_ok=True)
    log.write('Saving hnm bag previews to: '+out_dir)

    print('')
    for i in range(len(new_bags)):
        line_print('Saving hnm bag #' + str(i))

        current_bag = new_bags[i]
        current_bag_raw = new_bags_raw[i]
        current_name = new_bag_names[i]
        filename = out_dir + 'new_bag_' + str(i) + '.png'

        colored_tiles = []
        image_width = None
        image_height = None

        for rgb in current_bag_raw:
            rgb = rgb.astype('uint8')
            image_width, image_height = rgb[0].shape

            rgb = np.einsum('abc->bca', rgb)
            rgb = mil_metrics.outline_rgb_array(rgb, None, None, outline=2, override_colormap=[255, 255, 255])
            colored_tiles.append(rgb)

        out_image = mil_metrics.fuse_image_tiles(images=colored_tiles, image_width=image_width,
                                                 image_height=image_height)
        plt.imsave(filename, out_image)

    log.write('Finished saving previews.')


def main():
    print("This function provides helper functions to save rgb image samples.")


if __name__ == "__main__":
    main()
