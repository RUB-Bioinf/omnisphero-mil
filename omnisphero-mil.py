import loader
import numpy as np
import os

from sys import getsizeof

from lib.utils import convert_size

default_source_dir = "Y:\\bioinfdata\\work\\Omnisphero\\Sciebo\\HCA\\04_HTE_Batches\\BatchParamTest\\false_data\\mil-tiles\\selectedExperiment"
default_source_dir = "Y:\\bioinfdata\\work\\Omnisphero\\Sciebo\\HCA\\04_HTE_Batches\\BatchParamTest\\mil_labels\\exerpt"


def main(source_dir: str = default_source_dir):
    X, y, errors = loader.load_bags_json_batch(batch_dirs=[source_dir, source_dir, source_dir], max_workers=4)

    print("Finished loading training data. Loaded data has shape: ")
    print("X-shape: " + str(X.shape))
    print("y-shape: " + str(y.shape))

    # print("Correcting axes...")
    # X = np.moveaxis(X, 1, 3)
    # y = y.astype(np.int)
    # print("X-shape (corrected): " + str(X.shape))

    X_s = convert_size(getsizeof(X))
    y_s = convert_size(getsizeof(y))

    print("X-size: " + str(X_s))
    print("y-size: " + str(y_s))


if __name__ == '__main__':
    print("OmniSphero MIL")
    main()
