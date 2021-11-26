import math


class TileMetadata:

    def __init__(self, experiment_name: str, well_letter: str = '', well_number: int = math.nan, pos_x: int = 0,
                 pos_y: int = 0, well_image_width: int = 0, well_image_height: int = 0,
                 read_from_source: bool = True) -> None:
        super().__init__()

        # Setting fields
        self.experiment_name = experiment_name
        self.read_from_source = read_from_source
        if read_from_source:
            self.well_letter = well_letter
            self.well_number = well_number
            self.pos_x = pos_x
            self.pos_y = pos_y
            self.well_image_width = well_image_width
            self.well_image_height = well_image_height
        else:
            # But not for generated images
            self.well_letter = 'Auto'

    def get_formatted_well(self, long: bool = True):
        if self.read_from_source:
            return '<unknown>'

        if len(self.well_letter) == 1 and long:
            return self.well_letter + '0' + str(self.well_number)

        return self.well_letter + str(self.well_number)

    def get_bag_name(self):
        return self.experiment_name + ' - ' + self.get_formatted_well()

    def __str__(self) -> str:
        return "Well Metadata: " + super().__str__()


def main():
    print('This file contains no functioning code. It is a holder for the WellMetadata class containing well metadata.')


if __name__ == '__main__':
    main()
