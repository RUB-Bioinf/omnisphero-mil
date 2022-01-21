import os

from util import log


def load_oligo_predictions(image_folder: str, experiment_name: str, well_name: str, verbose: bool = False) -> [int,
                                                                                                               int]:
    if verbose:
        log.write('Loading oligo predictions for ' + experiment_name + ' - ' + well_name + ' from ' + image_folder)

    predictions_file, exists = get_prediction_file_path(image_folder, experiment_name, well_name)
    assert exists

    f = open(predictions_file, 'r')
    lines = f.readlines()
    f.close()

    coordinates = []
    for line in lines[1:-1]:
        entries = line.strip().split(';')

        # Having to switch x and y coordinates here, because matlab is matlab and matlab is the worst
        x = int(entries[2])
        y = int(entries[1])
        label = int(entries[3])

        if label == 1:
            coordinates.append([x, y])

    return coordinates


def get_prediction_file_path(image_folder, experiment_name, well_name):
    predictions_file = image_folder + experiment_name + os.sep + 'cnn' + os.sep + 'oligo' + os.sep + experiment_name \
                       + '_' + well_name + '_Manual-Neurons-2_overview_prediction.csv'
    exists = os.path.exists(predictions_file)

    return predictions_file, exists


def main():
    pass


if __name__ == '__main__':
    main()
