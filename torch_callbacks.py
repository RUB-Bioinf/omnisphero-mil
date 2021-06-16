import torch
from torch import Tensor


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


##########################################################
# Cancels training, if metrics become unreasonable
##########################################################

class UnreasonableLossCallback(BaseTorchCallback):

    def __init__(self, loss_max: float = 5.0):
        super().__init__()
        self.loss_max = loss_max

    def on_epoch_finished(self, model, epoch: int, epoch_result, history):
        super().on_epoch_finished(model, epoch, epoch_result, history)
        val_loss = epoch_result['val_loss']
        loss = epoch_result['train_loss']

        if val_loss > self.loss_max or loss > self.loss_max:
            self.request_cancellation()


if __name__ == "__main__":
    print('There are some pytorch callback functions and classes in this file. Nothing to execute. Have a nice day. :)')
