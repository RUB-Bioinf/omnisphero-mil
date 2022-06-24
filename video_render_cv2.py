import os
import cv2
import numpy as np
import video_render_ffmpeg
from functools import cmp_to_key
from PIL import Image

from util import log

default_fps: int = video_render_ffmpeg.default_fps


def img_path_to_array(image_path: str, file_formats: [str] = ['.png', '.jpg', '.jpeg']) -> np.ndarray:
    """ Function to merge images to one array """
    assert os.path.exists(image_path)

    image_files = os.listdir(image_path)
    n_files = len(image_files)
    x = 0
    y = 0
    z = 0

    read_images = []
    for current_file in image_files:
        current_file_path = image_path + os.sep + current_file
        if current_file_path.lower().endswith(tuple(file_formats)):
            read_image = Image.open(current_file_path)
            current_image = np.asarray(read_image, dtype=np.uint8)

            read_x, read_y, read_z = current_image.shape
            assert read_z >= 3

            if read_z == 4:
                # Case if the read image was RGBA
                temp = Image.new("RGB", (read_y, read_x), (255, 255, 255))
                temp.paste(read_image, mask=read_image.getchannel('A'))
                current_image = np.asarray(temp, dtype=np.uint8)
                read_x, read_y, _ = current_image.shape

                alpha = np.ones((read_x, read_y), dtype=np.uint8) * 150
                current_image = np.dstack((current_image, alpha))
                read_z = 4

                del temp

            x = max(x, read_x)
            y = max(y, read_y)
            z = max(z, read_z)
            read_images.append(current_image)
    read_images = sorted(read_images, key=cmp_to_key(_compare_sigmoid_frame_file_name))

    assert len(read_images) > 0
    arr = np.ones((len(read_images), x, y, z), dtype=np.uint8) * 255
    for i in range(len(read_images)):
        current_image = read_images[i]
        current_x, current_y, current_z = current_image.shape
        assert current_x == x
        assert current_y == y
        assert current_z == z
        arr[i, :, :, :] = current_image
        # TODO apply padding so smaller images match the bigger image!!

    return arr


def array_to_video_save(frames: np.ndarray, out_name: str, fps: int = default_fps, verbose: bool = True) -> Exception:
    err = None
    try:
        array_to_video_save(frames=frames, out_name=out_name, fps=fps, verbose=verbose)
    except Exception as e:
        err = e
        log.write('Error saving video: ' + out_name + ' - ' + str(e))

    return err


def array_to_video(frames: np.ndarray, out_path: str, fps: int = default_fps, verbose: bool = True) -> None:
    """ Function to convert image-array to a video """
    l, x, y, z = frames.shape
    frame_size = (y, x)

    if not out_path.lower().endswith('.avi'):
        out_path = out_path + '.avi'

    log.write('Saving a render [' + str(l) + ' frames, ' + str(fps) + ' FPS] to: ' + out_path, print_to_console=verbose)
    out_dir_name = os.path.dirname(out_path)
    os.makedirs(out_dir_name, exist_ok=True)

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, frame_size)
    for i in range(l):
        img = frames[i, :, :, :]
        img = np.uint8(img)
        out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release()

    log.write('Saved.', print_to_console=verbose)


def render_images_to_video_multiple(image_paths: [str], out_paths: [str] = None, override_out_name: str = None,
                                    fps: int = default_fps, verbose: bool = True) -> [Exception]:
    if not type(image_paths) == list:
        image_paths = [image_paths]
    if out_paths is None:
        out_paths = [None for _ in range(len(image_paths))]

    if not type(out_paths) == list:
        out_paths = [out_paths]
    assert len(out_paths) == len(image_paths)

    error_list = []
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        out_path = out_paths[i]
        log.write('[' + str(i + 1) + '/' + str(len(image_paths)) + '] - Rendering video for: ' + image_path,
                  print_to_console=verbose)

        try:
            render_images_to_video(image_path=image_path, out_path=out_path, override_out_name=override_out_name,
                                   fps=fps, verbose=verbose)
        except Exception as e:
            log.write('Error in render #' + str(i) + ': ' + str(e))
            error_list.append(e)

    log.write('All renders done. Error count: ' + str(len(error_list)), print_to_console=verbose)
    return error_list


def render_images_to_video(image_path: str, out_path: str = None, override_out_name: str = None,
                           fps: int = default_fps, verbose: bool = True):
    assert os.path.exists(image_path)
    if out_path is None:
        if override_out_name is None:
            override_out_name = 'render.avi'

        out_path = image_path + os.sep + override_out_name

    arr_images = img_path_to_array(image_path=image_path)
    array_to_video(frames=arr_images, out_path=out_path, fps=fps, verbose=verbose)


def _compare_sigmoid_frame_file_name(file_name1: str, file_name2: str):
    return video_render_ffmpeg.compare_sigmoid_frame_file_name(file_name1=file_name1, file_name2=file_name2)


if __name__ == "__main__":
    log.write('This function saves PNG/JPG images in a directory to .avi video.')
    log.write('Use "render_images_to_video" to access functionality.')

    render_images_to_video(
        'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\ep-aug-overlap-adadelta-n-6-rp-0.3-l-mean_square_error-BMC\\metrics_live\\sigmoid_live\\naive\\ELS681\\',
        verbose=True)
