import sys

from torch import Tensor

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
