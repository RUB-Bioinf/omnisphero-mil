import math
import re

# A regex to extract well info
from util import log

well_regex = '([A-Z]+)(\\d+)'


class TileMetadata:

    def __init__(self, experiment_name: str, well_letter: str = '', well_number: int = math.nan, pos_x: int = 0,
                 pos_y: int = 0, well_image_width: int = 0, well_image_height: int = 0,
                 count_nuclei: int = 0, count_oligos: int = 0, count_neurons: int = 0,
                 read_from_source: bool = True) -> None:
        super().__init__()

        # Setting fields
        self.experiment_name = experiment_name
        self.read_from_source = read_from_source

        self.count_nuclei = count_nuclei
        self.count_oligos = count_oligos
        self.count_neurons = count_neurons

        if read_from_source:
            self.well_letter = well_letter
            self.well_number = well_number
            self.pos_x = pos_x
            self.pos_y = pos_y
            self.well_image_width = well_image_width
            self.well_image_height = well_image_height
        else:
            # But not for generated images
            self.well_letter = '<runtime metadata>'

    def get_formatted_well(self, long: bool = True):
        if not self.read_from_source:
            return '<runtime generated tile>'

        if len(self.well_letter) == 1 and long:
            return self.well_letter + '0' + str(self.well_number)

        return self.well_letter + str(self.well_number)

    def get_bag_name(self):
        return self.experiment_name + ' - ' + self.get_formatted_well()

    def has_valid_position(self):
        return not math.isnan(self.pos_x) and not math.isnan(self.pos_y)

    def __str__(self) -> str:
        if self.read_from_source:
            return "Tile from bag: " + self.get_bag_name() + " at well " + str(self.well_letter) + str(
                self.well_number) + ". Pos: x: " + str(self.pos_x) + ' y: ' + str(
                self.pos_y) + '. Original image size: ' + str(self.well_image_width) + 'x' + str(self.well_image_height)
        else:
            return 'Unknown tile, created at runtime.'


def extract_well_info(well: str, verbose: bool = False) -> (str, int):
    m = re.findall(well_regex, well)[0]
    well_letter = m[0]
    well_number = int(m[1])
    if verbose:
        log.write('Reconstructing well: "' + well + '" -> "' + well_letter + str(well_number) + '".')

    return well_letter, well_number


def main():
    print('This file contains no functioning code. It is a holder for the WellMetadata class containing well metadata.')


if __name__ == '__main__':
    main()
