import os
import sys
from torch import Tensor

import mil_metrics
import models
from util import log


#########################
# Base class to inherit
#########################


class BaseTorchCallback:
    def __init__(self):
        super().__init__()
        self.__request_cancellation: bool = False

    def on_training_start(self, model):
        pass

    def on_training_finished(self, model, was_canceled: bool, history):
        pass

    def on_epoch_start(self, model, epoch: int):
        pass

    def on_epoch_finished(self, model, epoch: int, epoch_result, history):
        pass

    def on_batch_start(self, model, batch_id: int, data: Tensor, label: Tensor):
        pass

    def on_batch_finished(self, model, batch_id: int, data: Tensor, label: Tensor, batch_acc: float, batch_loss: float):
        pass

    def is_cancel_requested(self):
        return self.__request_cancellation

    def request_cancellation(self):
        self.__request_cancellation = True

    def reset(self):
        pass

    def __str__(self) -> str:
        return "Torch Callback: " + self._describe()

    def _describe(self) -> str:
        return None


##########################################################
# Cancels training, if metrics become unreasonable
##########################################################

class EarlyStopping(BaseTorchCallback):

    def __init__(self, epoch_threshold: int, metric: str = 'val_loss'):
        super().__init__()
        self.metric: str = metric
        self.epoch_threshold: int = epoch_threshold

        self.epochs_without_improvement = 0
        self.best_metric = sys.float_info.max
        self.reset()

    def reset(self):
        self.epochs_without_improvement = 0
        self.best_metric = sys.float_info.max

    def on_training_start(self, model):
        print('Early stopping initiated. Epochs: ' + str(self.epoch_threshold) + '. Metric: ' + self.metric)

    def on_epoch_finished(self, model, epoch: int, epoch_result, history):
        super().on_epoch_finished(model, epoch, epoch_result, history)

        current_metric: float = epoch_result[self.metric]
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement = self.epochs_without_improvement + 1
            if self.epochs_without_improvement >= self.epoch_threshold:
                log.write('Epoch threshold without metric improvement met. Early stopping training.')
                self.request_cancellation()

    def _describe(self) -> str:
        return 'Early Stopping. Metric: "' + self.metric + ". Threshold: " + str(self.epoch_threshold)


class SigmoidPredictionCallback(BaseTorchCallback):

    def __init__(self, out_dir: str, experiment_holders: dict, all_well_letters: [str], all_well_numbers: [int],
                 experiment_names_unique: [str], epoch_interval: int = 5, workers: int = 2, verbose: bool = True):
        super().__init__()
        self.epoch_interval: int = epoch_interval
        self.out_dir: str = out_dir
        self.verbose = verbose
        self.workers = workers

        self._experiment_holders = experiment_holders
        self._all_well_letters = all_well_letters
        self._all_well_numbers = all_well_numbers
        self._experiment_names_unique = experiment_names_unique
        self.error_out_filename = None

        os.makedirs(out_dir, exist_ok=True)

    def _describe(self) -> str:
        return 'Sigmoid Prediction Validation. Experiments: "' + str(
            self._experiment_names_unique) + ". Epochs: " + str(self.epoch_interval)

    def on_training_start(self, model):
        self.error_out_filename = self.out_dir + os.sep + 'errors.txt'
        f = open(self.error_out_filename, 'w')
        f.write('Errors:')
        f.close()

        if self.verbose:
            log.write('Saving sigmoid prediction errors to: '+self.error_out_filename)

    def on_epoch_finished(self, model, epoch: int, epoch_result, history):
        super().on_epoch_finished(model, epoch, epoch_result, history)

        if not (epoch % self.epoch_interval == 0):
            return

        if self.verbose:
            log.write('Predicting Sigmoid Previews')

        # Predicting every experiment and saving results
        for current_experiment in self._experiment_names_unique:
            try:
                current_holder = self._experiment_holders[current_experiment]
                all_wells, prediction_dict_bags, _, prediction_dict_well_names = models.predict_dose_response(
                    experiment_holder=current_holder, experiment_name=current_experiment, model=model,
                    max_workers=self.workers)

                # Writing the predictions to disk
                out_file_base = self.out_dir + os.sep + current_experiment + '-bags-ep' + str(epoch)
                mil_metrics.save_sigmoid_prediction_csv(experiment_name=current_experiment,
                                                        all_well_letters=self._all_well_letters,
                                                        prediction_dict=prediction_dict_bags,
                                                        file_path=out_file_base + '.csv',
                                                        prediction_dict_well_names=prediction_dict_well_names)
                mil_metrics.save_sigmoid_prediction_img(out_file_base + '.png', prediction_dict=prediction_dict_bags,
                                                        title='Dose Response: ' + current_experiment + ': '
                                                        + 'Whole Well\nEpoch: ' + str(epoch),
                                                        prediction_dict_well_names=prediction_dict_well_names,
                                                        x_ticks_angle=15, x_ticks_font_size=9)
            except Exception as e:
                error_text = 'Error during Epoch ' + str(epoch) + ' for experiment ' + current_experiment + '!'
                log.write('SIGMOID PREDICTION ERROR! ' + error_text)
                log.write(str(e))

                f = open(self.error_out_filename, 'a')
                f.write(error_text)
                f.write(str(e))
                f.close()


class UnreasonableLossCallback(BaseTorchCallback):

    def __init__(self, loss_max: float = 15.0):
        super().__init__()
        self.loss_max = loss_max

    def on_epoch_finished(self, model, epoch: int, epoch_result, history):
        super().on_epoch_finished(model, epoch, epoch_result, history)
        val_loss = epoch_result['val_loss']
        loss = epoch_result['train_loss']

        if val_loss > self.loss_max or loss > self.loss_max:
            log.write('The current loss has exceeded its maximum! Aborting training!!')
            self.request_cancellation()

    def _describe(self) -> str:
        return 'Unreasonable Loss. Max Loss: "' + str(self.loss_max)


if __name__ == "__main__":
    print('There are some pytorch callback functions and classes in this file. Nothing to execute. Have a nice day. :)')
