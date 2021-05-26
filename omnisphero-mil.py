import loader
import numpy as np
import os

from sys import getsizeof

from util.utils import convert_size

default_source_dir = "Y:\\bioinfdata\\work\\Omnisphero\\Sciebo\\HCA\\04_HTE_Batches\\BatchParamTest\\false_data\\mil-tiles\\selectedExperiment"
default_source_dir = "Y:\\bioinfdata\\work\\Omnisphero\\Sciebo\\HCA\\04_HTE_Batches\\BatchParamTest\\mil_labels\\exerpt"

# normalize_enum is an enum to determine normalisation as follows:
# 0 = no normalisation
# 1 = normalize every cell between 0 and 255 (8 bit)
# 2 = normalize every cell individually with every color channel independent
# 3 = normalize every cell individually with every color channel using the min / max of all three
# 4 = normalize every cell but with bounds determined by the brightest cell in the same well
normalize_enum_default = 4


def main(source_dir: str = default_source_dir, normalize_enum: int = normalize_enum_default):
    X, y, errors = loader.load_bags_json_batch(batch_dirs=[source_dir], max_workers=4,
                                               normalize_enum=normalize_enum)

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
