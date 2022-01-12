from util import log

default_out_dir_unix_base = '/mil/oligo-diff/models/linux'

training_metrics_live_dir_name = 'metrics_live'

debug_prediction_dirs_win = [
    'U:\\bioinfdata\\work\\OmniSphero\\mil\\oligo-diff\\test_data\\debug_win\\'
]

debug_prediction_dirs_unix = [
    # New CNN
    '/mil/oligo-diff/training_data/curated_linux/EFB18',
    '/mil/oligo-diff/training_data/curated_linux/ESM36',
    '/mil/oligo-diff/training_data/curated_linux/ELS411',
    '/mil/oligo-diff/training_data/curated_linux/ELS517',
    '/mil/oligo-diff/training_data/curated_linux/ELS637',
    '/mil/oligo-diff/training_data/curated_linux/ELS681',
    '/mil/oligo-diff/training_data/curated_linux/ELS682',
    '/mil/oligo-diff/training_data/curated_linux/ELS719',
    '/mil/oligo-diff/training_data/curated_linux/ELS744'
]

nucleus_predictions_image_folder_win = 'U:\\bioinfdata\\work\\OmniSphero\\Bilderordner\\'
nucleus_predictions_image_folder_unix = ''

if __name__ == '__main__':
    log.write('This class contains all important paths.')
