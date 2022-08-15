import sys

from torch import Tensor
from torch.optim import Optimizer

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
        return "Torch Callback: " + self.describe()

    def describe(self) -> str:
        return None


##########################################################
# Cancels training, if metrics become too stagnant
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

    def describe(self) -> str:
        return 'Early Stopping. Metric: "' + self.metric + ". Threshold: " + str(self.epoch_threshold)


##########################################################
# Cancels training, if metrics become too stagnant
##########################################################

class ReduceLearnRate(BaseTorchCallback):

    def __init__(self, epoch_threshold: int, optimizer: Optimizer, initial_lr: float, lr_scaling_factor: float = 0.5,
                 metric: str = 'val_loss'):
        super().__init__()
        self.metric: str = metric
        self.epoch_threshold: int = epoch_threshold
        self.optimizer = optimizer

        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.lr_scaling_factor = lr_scaling_factor

        self.epochs_without_improvement = 0
        self.best_metric = sys.float_info.max
        self.reset()

    def reset(self):
        self.current_lr = self.initial_lr
        self.epochs_without_improvement = 0
        self.best_metric = sys.float_info.max

    def on_training_start(self, model):
        print('Automated learning rate reduction enabled. Epochs: ' + str(
            self.epoch_threshold) + '. Metric: ' + self.metric + '. LR scaling factor: ' + str(
            self.lr_scaling_factor) + ' starting from ' + str(self.initial_lr))

        self._set_lr(new_lr=self.initial_lr)

    def on_epoch_finished(self, model, epoch: int, epoch_result, history):
        super().on_epoch_finished(model, epoch, epoch_result, history)
        # log.write(' ## DEBUG ##\nTorch callback optimizer state: ' + str(self.optimizer))

        current_metric: float = epoch_result[self.metric]
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement = self.epochs_without_improvement + 1
            if self.epochs_without_improvement >= self.epoch_threshold:
                log.write('Epoch threshold without metric improvement met. Reducing learn rate.')
                log.write('Optimizer state before reduction: ' + str(self.optimizer))
                self._apply_lr_reduction()
                self.epochs_without_improvement = 0
                log.write('Optimizer state after reduction: ' + str(self.optimizer))

    def describe(self) -> str:
        return 'Automated learning rate reduction enabled. Epochs: ' + str(
            self.epoch_threshold) + '. Metric: ' + self.metric + '. LR scaling factor: ' + str(
            self.lr_scaling_factor) + ' starting from ' + str(self.initial_lr)

    def _apply_lr_reduction(self):
        self.current_lr = self.current_lr * self.lr_scaling_factor
        self._set_lr(new_lr=self.current_lr)

    def _set_lr(self, new_lr):
        log.write('New LR applied: ' + str(new_lr))
        self.current_lr = new_lr

        # See: https://stackoverflow.com/a/48324389
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr


##########################################################
# Cancels training, if metrics become too spiky
##########################################################

class SpikingLossCallback(BaseTorchCallback):

    def __init__(self, loss_max: float = 15.0, tolerance: int = 15):
        super().__init__()
        self.loss_max = loss_max
        self.tolerance = tolerance
        self.irrational_epochs_count = 0

    def on_epoch_finished(self, model, epoch: int, epoch_result, history):
        super().on_epoch_finished(model, epoch, epoch_result, history)
        val_loss = epoch_result['val_loss']
        loss = epoch_result['train_loss']

        if val_loss > self.loss_max or loss > self.loss_max:
            self.irrational_epochs_count = self.irrational_epochs_count + 1
            log.write('The current loss has exceeded its maximum! Irrational spree: ' + str(
                self.irrational_epochs_count) + '/' + str(self.tolerance))

            if self.irrational_epochs_count >= self.tolerance:
                log.write('The spree has exceeded the tolerance. Aborting training.')
                self.request_cancellation()
        else:
            self.irrational_epochs_count = 0

    def reset(self):
        self.irrational_epochs_count = 0

    def describe(self) -> str:
        return 'Unreasonable Loss. Max Loss: "' + str(self.loss_max) + ' for ' + str(self.tolerance) + ' epochs.'


if __name__ == "__main__":
    print('There are some pytorch callback functions and classes in this file. Nothing to execute. Have a nice day. :)')
