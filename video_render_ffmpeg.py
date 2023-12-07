import os
import re
import subprocess
import sys
from typing import Union

from util import log
from util import paths
from functools import cmp_to_key

default_fps: int = 3

regex_filename_suffix_pattern = '(\\d+)$'
file_formats_default = ['.png', '.jpg', '.jpeg']


def render_image_dir_to_video_multiple(image_paths: [str], file_formats: [str] = file_formats_default,
                                       override_out_name: str = None, fps: int = default_fps, verbose: bool = True) \
        -> [Exception]:
    if not type(image_paths) == list:
        image_paths = [str(image_paths)]

    error_list = []
    for i in range(len(image_paths)):
        image_path = image_paths[i]
        log.write('[' + str(i + 1) + '/' + str(len(image_paths)) + '] - Rendering video for: ' + image_path,
                  print_to_console=verbose)

        try:
            render_image_dir_to_video(image_path=image_path, out_dir=image_path, file_formats=file_formats,
                                      override_out_name=override_out_name, fps=fps, verbose=verbose)
        except Exception as e:
            log.write('Error in render #' + str(i) + ': ' + str(e))
            error_list.append(e)

    log.write('All renders done. Error count: ' + str(len(error_list)), print_to_console=verbose)
    return error_list


def render_image_dir_to_video(image_path: str, out_dir: str, file_formats: [str] = file_formats_default,
                              override_out_name: str = None, fps: int = default_fps, verbose: bool = True):
    assert os.path.exists(image_path)

    image_files = os.listdir(image_path)

    image_paths = []
    for current_file in image_files:
        current_file_path = image_path + os.sep + current_file
        if current_file_path.lower().endswith(tuple(file_formats)):
            image_paths.append(current_file_path)

    image_paths = sorted(image_paths, key=cmp_to_key(compare_sigmoid_frame_file_name))
    render_images_to_video_multiple(image_paths=image_paths, out_dir=out_dir, override_out_name=override_out_name,
                                    fps=fps, verbose=verbose)


def render_images_to_video_multiple(image_paths: [str], out_dir: str, override_out_name: str = None,
                                    fps: Union[int, float] = default_fps, verbose: bool = True) -> (str, str, str):
    assert all([os.path.exists(p) for p in image_paths])
    os.makedirs(out_dir, exist_ok=True)

    if override_out_name is None:
        override_out_name = 'render.avi'

    frames_file_name = out_dir + 'frames.txt'
    render_file_name = out_dir + override_out_name
    frames_file = open(frames_file_name, 'w')
    for p in image_paths:
        frames_file.write('file \'' + p + '\'\n')
        frames_file.write('duration ' + str(fps) + '\n')
    frames_file.close()

    shell_command = get_ffmpeg_path()
    shell_command = shell_command + ' -f concat -safe 0 -i "' + frames_file_name + '" -vsync vfr -pix_fmt yuv420p -y -c copy "' + render_file_name + '"'
    log.write(shell_command)

    log.write('Rendering ' + str(len(image_paths)) + ' frames to: ' + render_file_name)
    pipe = subprocess.Popen(shell_command, shell=True, stdout=subprocess.PIPE).stdout
    output = pipe.read().decode()
    pipe.close()
    log.write('ffmpeg console log:\n\n' + str(output), print_to_console=False)

    log.write('Render done.')
    return render_file_name, pipe, str(output)


def get_ffmpeg_path() -> str:
    if sys.platform == 'win32':
        return paths.ffmpeg_win
    return paths.ffmpeg_unix


def compare_sigmoid_frame_file_name(file_name1: str, file_name2: str) -> int:
    for f_format in file_formats_default:
        if file_name1.lower().endswith(f_format):
            file_name1 = file_name1[:-len(f_format)]
        if file_name2.lower().endswith(f_format):
            file_name2 = file_name2[:-len(f_format)]

    file_suffix1 = re.split(regex_filename_suffix_pattern, file_name1)[1]
    file_suffix2 = re.split(regex_filename_suffix_pattern, file_name2)[1]

    if file_suffix1.isnumeric() and file_suffix2.isnumeric():
        file_suffix1 = int(file_suffix1)
        file_suffix2 = int(file_suffix2)

        return file_suffix1 - file_suffix2

    return 0


test_path = 'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\linux\\ep-aug-overlap-adadelta-n-6-rp-0.2-l-binary_cross_entropy-BMC\\metrics_live\\sigmoid_live\\naive\\EFB18'
test_out = 'Z:\\nilfoe\\Python\\omnisphero-mil\\ffmpeg\\win\\'

test_path_unix = '/bph/puredata4/bioinfdata/work/OmniSphero/mil/oligo-diff/models/linux/ep-aug-overlap-adadelta-n-6-rp-0.2-l-binary_cross_entropy-BMC/metrics_live/sigmoid_live/naive/EFB18'
test_out_unix = '/bph/home//nilfoe/Python/omnisphero-mil/ffmpeg/'

if __name__ == "__main__":
    log.write('This function saves PNG/JPG images in a directory to .mp4 video.')
    log.write('Use "render_images_to_video" to access functionality.')

    dir_path = os.path.dirname(os.path.realpath(get_ffmpeg_path()))
    render_image_dir_to_video(image_path=test_path,
                              out_dir=test_out)
