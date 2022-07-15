import os

from util import log

default_out_dir_unix_base = '/mil/oligo-diff/models/linux'

training_metrics_live_dir_name = 'metrics_live'

debug_prediction_dirs_win = [
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\test_data\\debug_win\\'
]

debug_training_dirs_win = [
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_win',
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_win\\prediction-debug2\\',
]

default_out_dir_win_base = 'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\models\\win'

all_prediction_dirs_win = [
    # New CNN
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux_overlap\\EFB18',
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux_overlap\\ELS517',
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux_overlap\\ELS637',
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux_overlap\\ELS719',
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux_overlap\\ELS744',
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux_overlap\\ESM36',
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux_overlap\\ELS681',
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux_overlap\\ELS682'
]

all_prediction_dirs_unix = [
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK129_PG',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK153_Calcitriol',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK176_MP',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK177_SR92',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS64_GW39',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS66_SR92',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS127_GW4671',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH56_GW6471',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH55_GW7647',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH44_GW7647',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH26_FU',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS94_PGE2',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK165_PGE2',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS102_SR92',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS104_Calcitriol',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS137_NH-3',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH41_Fu',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS96_GW0742',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS95_GW0742',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS77_GW39',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EFB18',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS517',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS637',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS719',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS744',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ESM36',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS681',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS682',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EJK228',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS510',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EMP124',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EMP146',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ESM26'
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK159_BaP',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK180_PG',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK184_NH3',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK198_UA',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK199_UA',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK201_AL08',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK206_UA',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK207_AL08',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS63_GW39',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS78_GW39',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS109_GW0742',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH45_GW7647',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH66_GW7647',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH74_AL08',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH75_AL08',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH76_UA',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH77_UA',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH93_AL08'
]

default_sigmoid_validation_dirs_unix = [
    '/mil/oligo-diff/training_data/curated_linux/EFB18',
    '/mil/oligo-diff/training_data/curated_linux/ELS681',
    '/mil/oligo-diff/training_data/curated_linux/ESM36'
]

default_sigmoid_validation_dirs_win = [
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux\\EFB18',
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_linux\\ELS681'
]

curated_overlapping_source_dirs_win = [
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\training_data\\curated_win'
]

curated_overlapping_source_dirs_unix = [
    # Overlapping Experiments from the ENDpoiNTs dataset #1
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK129_PG',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK153_Calcitriol',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK176_MP',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK177_SR92',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS64_GW39',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS66_SR92',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS127_GW4671',

    # Overlapping Experiments from the ENDpoiNTs dataset #2
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH56_GW6471',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH55_GW7647',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH44_GW7647',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH26_FU',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS94_PGE2',

    # Potentially difficult plates.
    # The oligo channel is quite overexposed in those.
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK165_PGE2',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS102_SR92',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS104_Calcitriol',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS137_NH-3',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH41_Fu',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS96_GW0742',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS95_GW0742',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS77_GW39',

    # Overlapping Experiments from the original EFSA dataset
    '/mil/oligo-diff/training_data/curated_linux_overlap/EFB18',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS517',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS637',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS719',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS744',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ESM36',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS681',
    # '/mil/oligo-diff/training_data/curated_linux_overlap/ELS682'

    # Overlapping Experiments from EFSA, Jun22
    '/mil/oligo-diff/training_data/curated_linux_overlap/EJK228',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS510',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EMP124',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EMP146',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ESM26',

    # Overlapping Experiments from ENDPoiNTs, Jun22
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK159_BaP',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK180_PG',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK184_NH3',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK198_UA',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK199_UA',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK201_AL08',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK206_UA',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK207_AL08',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS63_GW39',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS78_GW39',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS109_GW0742',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH45_GW7647',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH66_GW7647',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH74_AL08',
    # '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH75_AL08',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH76_UA',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH77_UA',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH93_AL08'
]

curated_overlapping_source_dirs_ep_unix = [
    # Overlapping Experiments from the ENDpoiNTs dataset #1
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK129_PG',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK153_Calcitriol',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK176_MP',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK177_SR92',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS64_GW39',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS66_SR92',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS127_GW4671',

    # Overlapping Experiments from the ENDpoiNTs dataset #2
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH56_GW6471',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH55_GW7647',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH44_GW7647',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH26_FU',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS94_PGE2',

    # Potentially difficult plates.
    # The oligo channel is quite overexposed in those.
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKK165_PGE2',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS102_SR92',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS104_Calcitriol',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS137_NH-3',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPSH41_Fu',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS96_GW0742',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS95_GW0742',
    '/mil/oligo-diff/training_data/curated_linux_overlap/EPKS77_GW39',
]

curated_overlapping_debug_dirs_unix = [
    '/mil/oligo-diff/training_data/curated_linux_overlap/EFB18',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS517',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS637',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS719'
]

# ffmpeg paths:
ffmpeg_win = 'Z:\\nilfoe\\Python\\omnisphero-mil\\ffmpeg\\win\\ffmpeg.exe'
ffmpeg_unix = '/bph/home/nilfoe/Python/omnisphero-mil/ffmpeg/unix/ffmpeg-5.0.1-amd64-static/ffmpeg'

nucleus_predictions_image_folder_win = 'U:\\bioinfdata\\work\\OmniSphero\\Bilderordner\\'
nucleus_predictions_image_folder_unix = '/bilderordner'

mil_metadata_file_win = nucleus_predictions_image_folder_win + 'mil_experiment_metadata.csv'
mil_metadata_file_linux = nucleus_predictions_image_folder_unix + os.sep + 'mil_experiment_metadata.csv'

all_prediction_dirs_win.sort()
curated_overlapping_source_dirs_unix.sort()
all_prediction_dirs_unix.sort()
debug_prediction_dirs_win.sort()
default_sigmoid_validation_dirs_unix.sort()
default_sigmoid_validation_dirs_win.sort()

if __name__ == '__main__':
    log.write('This class contains all important paths.')
