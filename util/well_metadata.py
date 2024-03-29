import math
import os
import re

import matplotlib.pyplot as plt
import r
# A regex to extract well info
from util import log
from util import utils

well_regex = '([A-Z]+)(\\d+)'


class PlateMetadata:

    def __init__(self, compound_name: str,
                 experiment_id: str,
                 compound_cas: str,
                 compound_concentration_max: float,
                 compound_concentration_dilution: float,
                 well_max_compound_concentration: int,
                 well_control: int,
                 compound_bmc30: float,
                 compound_oligo_diff: [float],
                 plate_bmc30: float) -> None:
        super().__init__()

        assert well_control < well_max_compound_concentration

        self.experiment_id = experiment_id
        self.compound_name = compound_name
        self.compound_cas = compound_cas
        self.compound_concentration_max = compound_concentration_max
        self.compound_concentration_dilution = compound_concentration_dilution
        self.well_max_compound_concentration = well_max_compound_concentration
        self.well_control = well_control
        self.plate_bmc30 = plate_bmc30
        self.compound_bmc30 = compound_bmc30
        self.compound_oligo_diff = compound_oligo_diff

    def __str__(self) -> str:
        return 'Compound: "' + self.compound_name + '" with ' + str(self.compound_concentration_max) + '@' + str(
            self.well_max_compound_concentration) + ' (1:' + str(
            self.compound_concentration_dilution) + ' until ' + str(self.well_control) + '.)'

    def has_valid_bmcs(self):
        return not math.isnan(self.plate_bmc30) and not math.isnan(self.compound_bmc30)

    def bake_plate_bmc30(self, out_dir: str, replicates: int = 5, dpi: int = 600,
                         bmc_30_compound_concentration: float = None, verbose: bool = False):
        os.makedirs(out_dir, exist_ok=True)
        out_name = out_dir + os.sep + self.experiment_id
        if not r.has_connection():
            error_text = 'Cannot bake BMC30 for ' + str(self) + '! No connection to R!'
            log.write(error_text)
            f = open(out_name + '.txt', 'w')
            f.write(error_text)
            f.close()
            return
        os.makedirs(out_dir, exist_ok=True)

        if bmc_30_compound_concentration is None:
            bmc_30_compound_concentration = float('NaN')

        wells = self.well_max_compound_concentration - self.well_control + 1
        assert self.compound_oligo_diff is not None
        assert wells == len(self.compound_oligo_diff)

        doses = []
        responses = []
        y_errors = []
        for i in range(wells):
            y_errors.append(0.0)
            for j in range(replicates):
                current_response = self.compound_oligo_diff[i]

                if not math.isnan(current_response):
                    responses.append(float(current_response))
                    doses.append(float(i + self.well_control))

        assert len(doses) == len(responses)
        if len(doses) == 0:
            error_text = 'Cannot bake BMC30 for ' + str(self) + '! No oligo diff responses or they are all "NaN"!'
            log.write(error_text)
            f = open(out_name + '.txt', 'w')
            f.write(error_text)
            f.close()
            return

        sigmoid_score, score_detail, estimate_plot, fitted_plot, instructions, bmc_30_plate_well = r.pooled_sigmoid_evaluation(
            save_sigmoid_plot=False,
            doses=doses, responses=responses, out_image_filename='Test.png')
        bmc_out_text_content = ''

        #############
        # Writing all them BMCs
        well_curve = fitted_plot[0]
        dose_curve = fitted_plot[1]
        f = open(out_name + '-all-bmcs.csv', 'w')
        f.write('BMC;Well Number;Concentration\n')
        log.write('Writing all BMCs to: '+str(f.name))
        for i in range(101):
            # Calculating BMCs in single steps
            try:
                current_bmc_well = r.calculate_bmc(well_curve=well_curve, dose_curve=dose_curve, bmc=i)
                current_bmc_concentration = self.interpolate_well_index_to_concentration(current_bmc_well)
            except Exception as e:
                current_bmc_well = float('NaN')
                current_bmc_concentration = float('NaN')
                log.write_exception(e)
                del e

            # Writing to file
            f.write(str(i) + ';' + str(current_bmc_well) + ';' + str(current_bmc_concentration) + '\n')
            del current_bmc_well, current_bmc_concentration, i
        f.close()
        del well_curve, dose_curve, f

        ####################
        # Plotting the dose response curve
        plt.clf()
        plt.plot(doses, responses, linestyle='-', marker='o', color='blue')
        plt.plot(fitted_plot[0], fitted_plot[1], color='lightblue')
        legend_entries = ['Raw Measurements', 'Curve Fit']
        legend_entries_tex = legend_entries.copy()
        y_axis_max: float = float(max(1.0, max(fitted_plot[1]), max(responses)))
        x_axis_max: int = math.floor(max(1.0, max(fitted_plot[0]), max(doses)))
        x_axis_min: int = math.floor(min(x_axis_max, min(fitted_plot[0]), min(doses)))

        bmc_30_plate_concentration = self.interpolate_well_index_to_concentration(bmc_30_plate_well)
        self.plate_bmc30 = bmc_30_plate_concentration
        bmc_30_plate_text = 'BMC30 (Plate): ' + str(round(bmc_30_plate_concentration, 3)) + ' ' + utils.mu + 'M'
        if not math.isnan(bmc_30_plate_concentration):
            plt.plot([bmc_30_plate_well, bmc_30_plate_well], [0, y_axis_max], color='lightgreen')
            legend_entries.append(bmc_30_plate_text)
            bmc_out_text_content = bmc_out_text_content + bmc_30_plate_text + ' at ' + str(bmc_30_plate_well) + '\n'

        if not math.isnan(bmc_30_compound_concentration):
            bmc_30_compound_well = self.interpolate_concentration_to_well(bmc_30_compound_concentration)
            bmc_30_compound_text = 'BMC30 (Compound): ' + str(
                round(bmc_30_compound_concentration, 3)) + ' ' + utils.mu + 'M'
            plt.plot([bmc_30_compound_well, bmc_30_compound_well], [0, y_axis_max], color='darkgreen')
            legend_entries.append(bmc_30_compound_text)
            bmc_out_text_content = bmc_out_text_content + bmc_30_compound_text + ' at ' + str(
                bmc_30_compound_well) + '\n'

        plt_title = 'AXIS Oligo Diff: ' + self.experiment_id + '\n' + self.compound_name + '\n Sigmoid Score: ' + str(
            round(sigmoid_score, 3))
        plt.title(plt_title)
        plt.xlabel('Well Index / Compound Concentration')
        plt.ylabel('% Control')

        plt.legend(legend_entries, loc='best')

        well_ticks = [i + 2 for i in range(wells)]
        all_well_indices = well_ticks.copy()
        well_ticks = [str(i + 2) + ' (' + str(round(self.get_concentration_at(i + 2), 3)) + ' ' + utils.mu + 'M)' for
                      i in range(wells)]

        plt.tight_layout()
        plt.autoscale()
        plt.xticks(all_well_indices, well_ticks, rotation=45)
        plt.ylim([0.0, y_axis_max + 0.05])
        plt.xlim([float(self.well_control) - 0.15, float(self.well_max_compound_concentration) + 0.15])
        plt.tight_layout()

        # Saving as image
        plt.savefig(out_name + '.png', dpi=dpi)
        plt.savefig(out_name + '.svg', dpi=dpi, transparent=True)
        plt.savefig(out_name + '.pdf', dpi=dpi)

        # Saving as .txt
        data_list_y = [responses, fitted_plot[1]]
        data_list_x = [doses, fitted_plot[0]]
        tikz_colors = ['blue', 'cyan']

        tikz = utils.get_plt_as_tex(data_list_y=data_list_y, data_list_x=data_list_x, plot_colors=tikz_colors,
                                    title=plt_title.replace('\n', ' - ').replace('  ', ' ').replace('_', '-'),
                                    plot_titles=legend_entries_tex,
                                    max_y=y_axis_max + 0.15, max_x=int(x_axis_max) + 1,
                                    min_x=x_axis_min, tick_count_x=len(well_ticks),
                                    legend_pos='south west',
                                    label_y='\\% Control', label_x='Well Index')
        f = open(out_name + '.tex', 'w')
        f.write(tikz)
        f.close()
        del f

        # Saving fit TO CSV
        sigmoid_out_csv = out_name + '-sigmoid_fit.csv'
        f = open(sigmoid_out_csv, 'w')
        f.write('i;x;y\n')
        [f.write(str(i) + ';' + str(fitted_plot[0][i]) + ';' + str(fitted_plot[1][i]) + '\n')
         for i in range(len(fitted_plot[0]))]
        f.close()
        del f

        # writing data to CSV
        out_csv = out_name + '-raw-measurements.csv'
        f = open(out_csv, 'w')
        # headers
        f.write(self.experiment_id + ' [Score: ' + str(sigmoid_score) + '];')
        [f.write(str(i) + ';') for i in all_well_indices]
        # responses
        f.write('\nResponse;')
        [f.write(str(self.compound_oligo_diff[i - self.well_control]) + ';') for i in all_well_indices]
        # concentrations
        f.write('\nConcentration;')
        [f.write(str(self.get_concentration_at(i)) + ';') for i in all_well_indices]
        f.close()
        del f

        # Writing the BMCs:
        f = open(out_name + '-bmc.txt', 'w')
        f.write(bmc_out_text_content.replace(utils.mu, 'u'))
        f.close()

        # Notifying the user by writing to the log
        log.write('Saved baked plate bmc evaluations here: ' + out_name + '.png', print_to_console=verbose)

    def get_concentration_at(self, well_index: int) -> float:
        # check for control case
        if well_index == self.well_control:
            return 0.0

        # Check if in range of well indices
        assert well_index >= self.well_control
        assert well_index <= self.well_max_compound_concentration

        current_concentration = self.compound_concentration_max
        for i in range(self.well_max_compound_concentration - well_index):
            current_concentration = current_concentration / self.compound_concentration_dilution

        return current_concentration

    def interpolate_well_index_to_concentration(self, well: float) -> float:
        if math.isnan(well):
            return float('NaN')

        # check if we want to have for an exact well entry
        if well.is_integer():
            return self.get_concentration_at(int(well))

        current_well = int(math.floor(well))
        well_remainder = well - float(current_well)

        current_concentration = self.get_concentration_at(current_well)
        next_concentration = self.get_concentration_at(current_well + 1)
        concentration_step = next_concentration - current_concentration

        interpolated_concentration = lerp(a=next_concentration, b=current_concentration, c=well_remainder)
        return interpolated_concentration

    def interpolate_concentration_to_well(self, concentration: float) -> float:
        if math.isnan(concentration):
            return float('NaN')

        closest_well = float('NaN')
        concentration_remainder = float('NaN')
        for i in range(self.well_control, self.well_max_compound_concentration):
            current_concentration = self.get_concentration_at(i)
            next_concentration = self.get_concentration_at(i + 1)

            # checking if the given concentration matches one in the metadata
            if round(concentration, 8) == round(current_concentration, 8):
                return float(i)
            if round(concentration, 8) == round(next_concentration, 8):
                return float(i + 1)

            if current_concentration <= concentration <= next_concentration:
                closest_well = i

                # concentration_step = next_concentration - current_concentration
                # step_progress = lerp(concentration_step, 0, current_concentration - concentration)
                # step_progress = lerp(concentration_step, 0, (concentration - current_concentration) / (
                #             next_concentration - current_concentration))
                step_progress = (concentration - current_concentration) / (next_concentration - current_concentration)

                lerped_well = lerp(closest_well + 1, closest_well, step_progress)
                concentration_remainder = current_concentration / next_concentration
                return lerped_well

        # closest_well=float(closest_well)
        # concentration_min = self.get_concentration_at(self.well_control)
        # concentration_max = self.get_concentration_at(self.well_max_compound_concentration)
        # if concentration_min <= concentration <= concentration_max:
        #    concentration_normalized = (concentration + concentration_min) / concentration_max
        #    concentration_lerp = lerp(a=self.well_max_compound_concentration, b=self.well_control,
        #                              c=concentration_normalized)
        #    return concentration_lerp
        #    # print(str(current_concentration) + ' - ' + str(next_concentration))

        # the concentration we are looking for, is not in the concentration range
        return float('NaN')


class TileMetadata:

    def __init__(self, experiment_name: str, well_letter: str = '', well_number: int = math.nan, pos_x: int = 0,
                 pos_y: int = 0, well_image_width: int = 0, well_image_height: int = 0,
                 count_nuclei: int = 0, count_oligos: int = 0, count_neurons: int = 0,
                 plate_metadata: PlateMetadata = None,
                 read_from_source: bool = True) -> None:
        super().__init__()

        # Setting fields
        self.plate_metadata = plate_metadata
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
            return '<Well info n/a. Runtime generated tile>'

        if len(self.well_letter) == 1 and long:
            return self.well_letter + '0' + str(self.well_number)

        return self.well_letter + str(self.well_number)

    def get_bag_name(self):
        return self.experiment_name + ' - ' + self.get_formatted_well()

    def has_valid_position(self):
        return not math.isnan(self.pos_x) and not math.isnan(self.pos_y)

    def __str__(self) -> str:
        if self.read_from_source:
            metadata_text = 'No plate metadata available.'
            if self.has_plate_metadata():
                metadata_text = 'Plate metadata: '
                str(self.plate_metadata) + '.'

            return "Tile from bag: " + self.get_bag_name() + " at well " + str(self.well_letter) + str(
                self.well_number) + ". Pos: x: " + str(self.pos_x) + ' y: ' + str(
                self.pos_y) + '. Original image size: ' + str(self.well_image_width) + 'x' + str(
                self.well_image_height) + '. ' + metadata_text
        else:
            return 'Unknown tile, created at runtime.'

    def has_plate_metadata(self) -> PlateMetadata:
        return self.plate_metadata is not None

    def get_concentration(self) -> float:
        if not self.has_plate_metadata():
            return float('nan')

        return self.plate_metadata.get_concentration_at(self.well_number)


def extract_well_info(well: str, verbose: bool = False) -> (str, int):
    m = re.findall(well_regex, well)[0]
    well_letter = m[0]
    well_number = int(m[1])
    if verbose:
        log.write('Reconstructing well: "' + well + '" -> "' + well_letter + str(well_number) + '".')

    return well_letter, well_number


def lerp(a: float,  # the maximum value to be lerped
         b: float,  # the minimum value to be lerped
         c: float  # the lerping index (should be between 0.0 and 1.0)
         ):
    a = float(a)
    b = float(b)
    c = float(c)
    return (c * a) + ((1 - c) * b)


def main():
    print('This file contains no functioning code. It is a holder for the WellMetadata class containing well metadata.')


if __name__ == '__main__':
    main()
