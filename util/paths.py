from util import log

default_out_dir_unix_base = '/mil/oligo-diff/models/linux'

training_metrics_live_dir_name = 'metrics_live'

debug_prediction_dirs_win = [
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\test_data\\debug_win\\'
]

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

debug_prediction_dirs_unix = [
    # New CNN
    '/mil/oligo-diff/training_data/curated_linux_overlap/EFB18',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS517',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS637',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS719',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS744',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ESM36',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS681',
    '/mil/oligo-diff/training_data/curated_linux_overlap/ELS682'
]

nucleus_predictions_image_folder_win = 'U:\\bioinfdata\\work\\OmniSphero\\Bilderordner\\'
nucleus_predictions_image_folder_unix = '/bilderordner'

if __name__ == '__main__':
    log.write('This class contains all important paths.')
