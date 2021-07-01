import numpy as np

_label_map = {}


def put_label(sample: np.ndarray, label: int):
    global _label_map

    if contains_sample(sample):
        raise Exception('The label for sample is already known! -> ' + str(sample))

    _label_map[to_hash(sample)] = label


def contains_sample(sample: np.ndarray) -> bool:
    global _label_map
    return to_hash(sample) in _label_map


def clear():
    global _label_map
    _label_map.clear()


def count() -> int:
    global _label_map
    return len(_label_map)


def get_label(sample: np.ndarray):
    global _label_map
    return _label_map[to_hash(sample)]


def get_hashes() -> [int]:
    global _label_map
    return _label_map.keys()


def to_file(filename: str):
    f = open(filename, 'w')
    f.write('Index;Hash;Label')

    i = 0
    for key in _label_map.keys():
        i = i + 1
        label = _label_map[key]

        f.write('\n'+str(i) + ';' + str(key) + ';' + str(label))

    f.close()


def invert_labels():
    for key in _label_map.keys():
        label = _label_map[key]
        _label_map[key] = not label


def to_hash(n: np.ndarray) -> int:
    s = n.shape
    if not (s[0] == 3 and len(s) == 3):
        raise Exception('Numpy Shape does not fit hashing criterium: ' + str(s))

    n = n.astype('float32')
    return hash(str(n))


def main():
    print('This function houses a map of sample-hashes to label pairs.')


if __name__ == '__main__':
    main()
