import math
import os
import sys
from datetime import datetime
from datetime import timedelta
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import hardware
import loader
import mil_metrics
import torch_callbacks
from util import log
from util import utils
from util.omnisphero_data_loader import OmniSpheroDataLoader
from util.paths import training_metrics_live_dir_name
from util.utils import line_print


def save_model(state, save_path: str, verbose: bool = False):
    if verbose:
        log.write('Saving model: ' + save_path)
    torch.save(state, save_path)


# MODEL
#######

device_ordinals_local = [0, 0, 0, 0]
device_ordinals_ehrlich = [0, 1, 2, 3]
device_ordinals_cluster = [0, 1, 2, 3]


class OmniSpheroMil(nn.Module):

    def __init__(self, device, device_ordinals=None):
        super().__init__()
        self.device = device
        self._device_ordinals = device_ordinals

    def get_device_ordinal(self, index: int) -> str:
        if self._device_ordinals is None:
            return 'cpu'

        if self.is_cpu():
            return 'cpu'

        return 'cuda:' + str(self._device_ordinals[index])

    def is_cpu(self) -> bool:
        return self.device.type == 'cpu'

    def get_device_ordinals(self) -> [int]:
        return self._device_ordinals.copy()

    def compute_loss(self, X: Tensor, y: Tensor):
        pass

    def compute_accuracy(self, X: Tensor, y: Tensor, y_tiles: Tensor):
        pass


####

class BaselineMIL(OmniSpheroMil):
    def __init__(self, input_dim, device, loss_function: str, accuracy_function: str, use_bias=True, use_max=True,
                 device_ordinals=None, enable_attention: bool = False):
        super().__init__(device, device_ordinals)
        self.linear_nodes: int = 512
        self.num_classes: int = 1  # 3
        self.input_dim = input_dim

        self.use_bias: bool = use_bias
        self.use_max: bool = use_max
        self.loss_function: str = loss_function
        self.accuracy_function: str = accuracy_function
        self.enable_attention: bool = enable_attention
        self.attention_nodes = 128

        if self.enable_attention:
            self.use_max = False

        # add batch norm? increases model complexity but possibly speeds up convergence
        self.feature_extractor_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim[0], out_channels=16, kernel_size=3, bias=self.use_bias),
            nn.LeakyReLU(0.01),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=self.use_bias),
            # nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),

            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, bias=self.use_bias),
            # nn.LeakyReLU(0.01),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=self.use_bias),
            # nn.LeakyReLU(0.01),
            # nn.MaxPool2d(2, stride=2),

            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=self.use_bias),
            # nn.LeakyReLU(0.01),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=self.use_bias),
            # nn.LeakyReLU(0.01),
            # nn.MaxPool2d(2, stride=2)
        )
        self.feature_extractor_0 = self.feature_extractor_0.to(self.get_device_ordinal(0))

        self.feature_extractor_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, bias=self.use_bias),
            nn.LeakyReLU(0.01),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, bias=self.use_bias),
            # nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, bias=self.use_bias),
            nn.LeakyReLU(0.01),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=self.use_bias),
            # nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),

            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, bias=self.use_bias),
            # #nn.LeakyReLU(0.01),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, bias=self.use_bias),
            # nn.LeakyReLU(0.01),
            # nn.MaxPool2d(2, stride=2)
        )
        self.feature_extractor_1 = self.feature_extractor_1.to(self.get_device_ordinal(1))

        self.feature_extractor_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=self.use_bias),
            nn.LeakyReLU(0.01),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, bias=self.use_bias),
            # nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),

            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, bias=self.use_bias),
            # nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=(2, 2), bias=self.use_bias),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, bias=self.use_bias),
            nn.LeakyReLU(0.01),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, bias=self.use_bias),
            # nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2)
        )
        self.feature_extractor_2 = self.feature_extractor_2.to(self.get_device_ordinal(2))

        size_after_conv = self._get_conv_output(self.input_dim)

        self.feature_extractor_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=size_after_conv, out_features=self.linear_nodes, bias=self.use_bias),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),

            # nn.Linear(in_features=self.linear_nodes, out_features=self.linear_nodes, bias=self.use_bias),
            # nn.LeakyReLU(0.01),
            # nn.Dropout(0.5),

            nn.Linear(in_features=self.linear_nodes, out_features=self.linear_nodes, bias=self.use_bias),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5)
        )
        self.feature_extractor_3 = self.feature_extractor_3.to(self.get_device_ordinal(3))  # bag of embeddings

        # if not self.use_max:
        #    self.mean_pool = nn.Sequential(
        #        torch.mean(dim=0, keepdim=True)
        #    ).to('cuda:3') # two-layer NN that replaces the permutation invariant pooling operator( max or mean normally, which are pre-defined and non trainable) with an adaptive weighting attention mechanism

        # elif self.use_max:
        #    self.max_pool = nn.Sequential(
        #        torch.max(dim=0, keepdim=True)
        #    ).to('cuda:3')

        if self.enable_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.linear_nodes, self.attention_nodes, bias=self.use_bias),
                nn.Tanh(),
                nn.Linear(self.attention_nodes, 1)  # self.num_classes, bias=self.use_bias)
            )
            self.attention = self.attention.to(self.get_device_ordinal(3))  # two-layer NN that

        self.classifier = nn.Sequential(
            nn.Linear(self.linear_nodes, self.num_classes),  # * self.num_classes, self.num_classes),
            # nn.Softmax()
            nn.Sigmoid()
        )
        self.classifier = self.classifier.to(self.get_device_ordinal(3))

    def forward(self, x):
        """ Forward NN pass, declaring the exact interplay of model components
        """
        # Running the bag through the hidden layers
        x = x.squeeze(
            0)  # necessary? compresses unnecessary dimensions eg. (1,batch,channel,x,y) -> (batch,channel,x,y)
        hidden = self.feature_extractor_0(x)
        hidden = self.feature_extractor_1(hidden.to(self.get_device_ordinal(1)))
        hidden = self.feature_extractor_2(hidden.to(self.get_device_ordinal(2)))
        hidden = self.feature_extractor_3(hidden.to(self.get_device_ordinal(3)))  # N x linear_nodes

        # Predicting the whole bag
        if not self.use_max:
            pooled = torch.mean(hidden, dim=[0, 1], keepdim=True)  # N x num_classes
            attention = torch.tensor([[0.5]])
        elif self.use_max:
            pooled = torch.max(hidden)  # N x num_classes
            pooled = pooled.unsqueeze(dim=0)
            pooled = pooled.unsqueeze(dim=0)
            attention = torch.tensor([[0.5]])

        if self.enable_attention:
            attention = self.attention(hidden)
            attention = torch.transpose(attention, 1, 0)
            attention = F.softmax(attention, dim=1)

            pooled = torch.mm(attention, hidden)

        y_hat = self.classifier(pooled)

        # Predicting every single tile
        y_tiles = []
        y_tiles_binarized = []
        s = hidden.shape
        for i in range(s[0]):
            h = hidden[i]
            h = h.unsqueeze(dim=0)

            if not self.use_max:
                h = torch.mean(h, dim=[0, 1], keepdim=True)
            elif self.use_max:
                h = torch.max(h)
                h = h.unsqueeze(dim=0)
                h = h.unsqueeze(dim=0)

            if self.enable_attention:
                a = attention[0, i]
                h = hidden[i]
                h = h.unsqueeze(dim=0)

                h = a * h

            h = self.classifier(h)
            y_tiles.append(h)
            y_tiles_binarized.append(torch.ge(h, 0.5).float())

        # Returning the predictions
        y_hat_binarized = torch.ge(y_hat, 0.5).float()
        return y_hat, y_hat_binarized, attention, y_tiles, y_tiles_binarized

    def _get_conv_output(self, shape):
        """ generate a single fictional input sample and do a forward pass over
        Conv layers to compute the input shape for the Flatten -> Linear layers input size
        """
        bs = 1
        test_input = torch.autograd.Variable(torch.rand(bs, *shape)).to(self.get_device_ordinal(0))
        output_features = self.feature_extractor_0(test_input)
        output_features = self.feature_extractor_1(output_features.to(self.get_device_ordinal(1)))
        output_features = self.feature_extractor_2(output_features.to(self.get_device_ordinal(2)))
        n_size = int(output_features.data.view(bs, -1).size(1))
        del test_input, output_features
        return n_size

    # COMPUTATION METHODS
    def compute_loss(self, X: Tensor, y: Tensor):
        """ otherwise known as loss_fn
        Takes a data input of X,y (batches or bags) computes a forward pass and the resulting error.
        """
        y = y.float()
        # y = y.unsqueeze(dim=0)
        # y = torch.argmax(y, dim=1)

        y_hat, y_hat_binarized, attention, _, _ = self.forward(X)
        # y_prob = torch.ge(y_hat, 0.5).float() # for binary classification only. Rounds prediction output to 0 or 1
        y_prob = torch.clamp(y_hat, min=1e-5, max=1. - 1e-5)
        # y_prob = y_prob.squeeze(dim=0)

        method = self.loss_function
        loss: Tensor = None
        if method == 'negative_log_bernoulli':
            loss = -1. * (
                    y * torch.log(y_prob) + (1.0 - y) * torch.log(1.0 - y_prob))  # negative log bernoulli loss
        elif method == 'binary_cross_entropy':
            # loss = y * torch.log(y_hat) + (1.0 - y) * torch.log(1.0 - y_hat)
            # loss = torch.clamp(loss, min=-1.0, max=1.0)
            # loss = loss * -1

            b = nn.BCELoss()
            loss = b(y_hat.squeeze(), y)

        # loss = F.cross_entropy(y_hat, y)
        # loss = F.binary_cross_entropy(y_hat_binarized, y)
        # loss_func = nn.BCELoss()
        # loss = loss_func(y_hat, y)
        return loss, attention

    def compute_accuracy(self, X: Tensor, y: Tensor, y_tiles: Tensor) -> (Tensor, float, [float], [int], float):
        """ compute accuracy
        """
        y = y.float()
        y = y.unsqueeze(dim=0)
        # y = torch.argmax(y, dim=1)

        y_hat, y_hat_binarized, _, y_hat_tiles, _ = self.forward(X)
        y_hat = y_hat.squeeze(dim=0)

        bag_acc = None
        method = self.accuracy_function
        if method == 'binary':
            bag_acc = _binary_accuracy(y_hat, y)
        else:
            raise Exception('Illegal accuracy method selected: "' + method + '"!')

        tile_acc = None
        tile_accuracy_list = None
        tile_prediction_list = None
        if y_tiles is not None:
            tile_acc, tile_accuracy_list, tile_prediction_list = self.compute_accuracy_tiles(y_targets=y_tiles,
                                                                                             y_predictions=y_hat_tiles)

        return bag_acc, tile_acc, tile_accuracy_list, tile_prediction_list, float(y_hat), y_hat_binarized

    def compute_accuracy_tiles(self, y_targets: Tensor, y_predictions: [Tensor]) -> (float, [float], [int]):
        accuracy_list = []
        tile_hat_list = []

        if not (len(y_predictions) == y_targets.shape[1]):
            raise Exception('Prediction and target sizes must match!')

        for i in range(len(y_predictions)):
            hat: Tensor = y_predictions[i].squeeze()
            hat = torch.ge(hat, 0.5).float()  # TODO use binary from forward pass here??
            y: Tensor = y_targets[0, i].float()

            correct = float(y) == float(hat)
            accuracy_list.append(float(correct))
            tile_hat_list.append(int(hat))

        accuracy = sum(accuracy_list) / len(accuracy_list)
        return accuracy, accuracy_list, tile_hat_list


####

def load_checkpoint(load_path: str, model: OmniSpheroMil, optimizer: Optimizer = None, map_location=None) -> (
        OmniSpheroMil, Optimizer, int, float):
    ''' loads the model and its optimizer states from disk.
    '''
    checkpoint = torch.load(load_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss


####

def choose_optimizer(model: OmniSpheroMil, selection: str) -> Optimizer:
    """ Chooses an optimizer according to the string specifed in the model CLI argument and build with specified args
    """
    if selection == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                     amsgrad=False)
    elif selection == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    elif selection == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay=0,
                                    nesterov=False)
    else:
        raise Exception("Error! Chosen optimizer or its parameters are unclear: '" + selection + "'")
    return optimizer


def fit(model: OmniSpheroMil, optimizer: Optimizer, epochs: int, training_data: OmniSpheroDataLoader,
        validation_data: OmniSpheroDataLoader, out_dir_base: str, callbacks: [torch_callbacks.BaseTorchCallback],
        checkpoint_interval: int = 1, clamp_min: float = None, clamp_max: float = None,
        augment_training_data: bool = False, augment_validation_data: bool = False):
    """ Trains a model on the previously preprocessed train and val sets.
    Also calls evaluate in the validation phase of each epoch.
    """
    best_loss = sys.float_info.max
    history = []
    history_keys = ['train_loss', 'train_acc', 'val_acc', 'val_loss', 'train_roc_auc', 'val_roc_auc',
                    'train_dice_score', 'val_dice_score']

    checkpoint_out_dir = out_dir_base + 'checkpoints' + os.sep
    metrics_dir_live = out_dir_base + training_metrics_live_dir_name + os.sep
    epoch_data_dir_live = metrics_dir_live + 'epochs_live' + os.sep
    os.makedirs(checkpoint_out_dir, exist_ok=True)
    os.makedirs(metrics_dir_live, exist_ok=True)
    os.makedirs(epoch_data_dir_live, exist_ok=True)

    # Writing Live Loss CSV
    batch_headers = ';'.join(
        ['Batch ' + str(batch_id) for batch_id, (data, label, tile_labels, batch_index) in enumerate(training_data)])
    batch_losses_file = metrics_dir_live + 'batch_losses.csv'
    f = open(batch_losses_file, 'w')
    f.write('Epoch;' + batch_headers)
    f.close()

    # Writing Live Tile Accuracy CSV
    tile_accuracy_file = metrics_dir_live + 'tile_accuracy.csv'
    f = open(tile_accuracy_file, 'w')
    f.write('Epoch;' + batch_headers)
    f.write('\nValidation;' + ';'.join(
        [str(tile_labels.cpu().numpy()[0]).replace('\n', '') for batch_id, (data, label, tile_labels, tile_index) in
         enumerate(training_data)]))
    f.close()

    # Preparing detailed epoch output files
    data_batch_dirs = [epoch_data_dir_live + 'data_batch_' + str(batch_id) + '.csv' for
                       batch_id, (data, label, tile_labels, original_bag_index) in enumerate(training_data)]
    [open(d, 'w').close() for d in data_batch_dirs]

    # Fields and param setup
    cancel_requested = False
    loss = None
    os.makedirs(checkpoint_out_dir, exist_ok=True)
    model_save_path_best = out_dir_base + 'model_best.h5'
    epoch_durations = []

    if model.is_cpu():
        log.write('Training on CPU.')
    else:
        log.write('Training on GPU, using these ordinals: ' + str(model.get_device_ordinals()))

    # Notifying callbacks
    for i in range(len(callbacks)):
        callback: torch_callbacks.BaseTorchCallback = callbacks[i]
        callback.on_training_start(model=model)

    epoch = 0
    while (epoch - 1 <= epochs) and not cancel_requested:
        epoch = epoch + 1

        # TRAINING PHASE
        model.train()
        train_losses = []
        train_acc = []
        train_acc_tiles = []
        all_labels = []
        predicted_labels = []
        train_FP = 0
        train_TP = 0
        train_FN = 0
        train_TN = 0
        start_time_epoch = datetime.now()
        epochs_remaining = epochs - epoch

        for batch_id, (data, label, tile_labels, bag_index) in enumerate(training_data):
            # torch.cuda.empty_cache()

            # Notifying Callbacks
            for i in range(len(callbacks)):
                callback: torch_callbacks.BaseTorchCallback = callbacks[i]
                callback.on_batch_start(model=model, batch_id=batch_id, data=data, label=label)

            label = label.squeeze()
            bag_label = label

            # writing data from the batch
            f = open(data_batch_dirs[batch_id], 'a')
            f.write('Epoch: ' + str(epoch))
            f.write('\nTile Index;Tile Hash;Bag Label;Tile Labels;Tile Labels (Sum)')
            for i in range(data.shape[1]):
                tile = data[0, i].cpu().numpy()
                tile_label = tile_labels[0, i].numpy()
                f.write('\n' + str(i) + ';' + np.format_float_positional(hash(str(tile))) + ';' + str(
                    label.numpy()) + ';' + str(tile_label))
            f.write('\n\n')
            f.close()

            data = data.to(model.get_device_ordinal(0))
            bag_label = bag_label.to(model.get_device_ordinal(3))
            tile_labels = tile_labels.to(model.get_device_ordinal(3))

            # resets gradients
            model.zero_grad()

            loss, _ = model.compute_loss(X=data, y=bag_label)  # forward pass
            if clamp_min is not None:
                loss = torch.clamp(loss, min=clamp_min)
            if clamp_max is not None:
                loss = torch.clamp(loss, max=clamp_max)

            # https://github.com/yhenon/pytorch-retinanet/issues/3
            train_losses.append(float(loss))
            acc, acc_tiles, _, _, y_hat, y_hat_binarized = model.compute_accuracy(data, bag_label, tile_labels)
            train_acc.append(float(acc))
            train_acc_tiles.append(float(acc_tiles))
            predicted_labels.append(float(y_hat))
            all_labels.append(float(bag_label))

            # Checking if prediction is TP / FP ... etc
            binary_label = bool(label)
            binary_prediction = bool(y_hat_binarized)
            if binary_label:
                if binary_prediction:
                    train_TP += 1
                else:
                    train_FP += 1
            else:
                if binary_prediction:
                    train_FN += 1
                else:
                    train_TN += 1

            # Notifying Callbacks
            for i in range(len(callbacks)):
                callback: torch_callbacks.BaseTorchCallback = callbacks[i]
                callback.on_batch_finished(model=model, batch_id=batch_id, data=data, label=label, batch_acc=float(acc),
                                           batch_loss=float(loss))

            loss.backward()  # backward pass
            optimizer.step()  # update parameters
            # optim.zero_grad() # reset gradients (alternative if all grads are contained in the optimizer)
            # for p in model.parameters(): p.grad=None # alternative for model.zero_grad() or optim.zero_grad()
            del data, bag_label, label, y_hat

        # VALIDATION PHASE
        result, _, all_losses, all_tile_lists, _ = evaluate(model, validation_data, clamp_max=clamp_max,
                                                            clamp_min=clamp_min,
                                                            apply_data_augmentation=augment_validation_data)  # returns a results dict for metrics

        fpr, tpr, thresholds = mil_metrics.binary_roc_curve(all_labels, predicted_labels)
        roc_auc = float('NaN')
        try:
            roc_auc = auc(fpr, tpr)
        except Exception as e:
            log.write(str(e))
        result['train_roc_auc'] = roc_auc

        result['train_FP'] = train_FP
        result['train_TP'] = train_TP
        result['train_FN'] = train_FN
        result['train_TN'] = train_TN
        result['train_dice_score'] = mil_metrics.calculate_dice_score(TP=train_TP, FP=train_FP, FN=train_FN)
        result['train_loss'] = sum(train_losses) / len(train_losses)  # torch.stack(train_losses).mean().item()
        result['train_acc'] = sum(train_acc) / len(train_acc)
        result['train_acc_tiles'] = sum(train_acc_tiles) / len(train_acc_tiles)

        time_diff = utils.get_time_diff(start_time_epoch)
        duration = (datetime.now() - start_time_epoch).total_seconds()
        epoch_durations.append(duration)
        result['duration'] = duration

        remaining_time_eta = timedelta(seconds=np.mean(epoch_durations) * epochs_remaining)
        eta_timestamp = datetime.now() + remaining_time_eta
        eta_timestamp = eta_timestamp.strftime("%Y/%m/%d, %H:%M:%S")

        history.append(result)
        log.write(
            'Epoch {}/{}: Train Loss: {:.4f}, Train Acc (Bags): {:.4f}, Val Loss: {:.4f}, '
            'Val Acc (Bags): {:.4f}, Train AUC: {:.4f}, Val AUC: {:.4f}. Duration: {}. ETA: {}'.format(
                epoch, epochs, result['train_loss'], result['train_acc'], result['val_loss'],
                result['val_acc'], result['train_roc_auc'], result['val_roc_auc'], time_diff,
                eta_timestamp))

        # Notifying Callbacks
        for i in range(len(callbacks)):
            callback: torch_callbacks.BaseTorchCallback = callbacks[i]
            callback.on_epoch_finished(model=model, epoch=epoch, epoch_result=result, history=history)
            cancel_requested = cancel_requested or callback.is_cancel_requested()

        # Save best model / checkpointing stuff
        is_best = bool(result['val_loss'] < best_loss)
        best_loss = min(result['val_loss'], best_loss)
        state = {
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }

        # Saving raw history
        try:
            mil_metrics.write_history(history, history_keys, metrics_dir=metrics_dir_live)
            mil_metrics.plot_accuracy(history, metrics_dir_live, include_raw=False, include_tikz=False)
            mil_metrics.plot_accuracy_tiles(history, metrics_dir_live, include_raw=False, include_tikz=False)
            mil_metrics.plot_losses(history, metrics_dir_live, include_raw=False, include_tikz=False)
            mil_metrics.plot_accuracies(history, metrics_dir_live, include_tikz=False)
            mil_metrics.plot_dice_scores(history, metrics_dir_live, include_tikz=True)
            mil_metrics.plot_binary_roc_curves(history, metrics_dir_live, include_tikz=False)
        except Exception as e:
            print('Failed to write metrics for this epoch.')
            print(str(e))

        # Printing each loss for every batch
        f = open(batch_losses_file, 'a')
        f.write('\n' + str(epoch) + ';' + ';'.join([str(l) for l in all_losses]))
        f.close()

        f = open(tile_accuracy_file, 'a')
        f.write('\n' + str(epoch) + ';' + ';'.join([str(np.asarray(li)).replace('\n', '') for li in all_tile_lists]))
        f.close()

        # Saving model checkpoints
        if checkpoint_interval is not None and checkpoint_interval > 0:
            if epoch % checkpoint_interval == 0:
                save_model(state, checkpoint_out_dir + 'checkpoint-' + str(epoch) + '.h5', verbose=True)

        if is_best:
            log.write('New best model! Saving...')
            save_model(state, model_save_path_best, verbose=False)

        if cancel_requested:
            log.write('Model was canceled before reaching all epochs.')

    # Notifying callbacks
    for i in range(len(callbacks)):
        callback: torch_callbacks.BaseTorchCallback = callbacks[i]
        callback.on_training_finished(model=model, was_canceled=cancel_requested, history=history)

    return history, history_keys, model_save_path_best


@torch.no_grad()
def get_predictions(model: OmniSpheroMil, data_loader: DataLoader, verbose: bool = False):
    """ takes a trained model and validation or test dataloader
    and applies the model on the data producing predictions

    binary version
    """
    model.eval()

    all_y_hats = []
    all_predictions = []
    all_true = []
    all_attentions = []
    all_y_tiles_binarized = []
    all_y_tiles = []
    all_tiles_true = []
    original_bag_indices = []

    if verbose:
        log.write('Predicting bags. Count: ' + str(len(data_loader)))
        print('')

    for batch_id, (data, label, label_tiles, original_bag_index) in enumerate(data_loader):
        if verbose:
            line_print('Predicting bag ' + str(batch_id + 1) + '/' + str(len(data_loader)), include_in_log=True)

        label = label.squeeze()
        # bag_label = label[0]
        bag_label = label
        bag_label = bag_label.cpu()

        data = data.to(model.get_device_ordinal(0))
        y_hat, predictions, attention, prediction_tiles, prediction_tiles_binarized = model.forward(data)

        y_hat = y_hat.squeeze(dim=0)  # for binary setting
        y_hat = y_hat.cpu()
        predictions = predictions.squeeze(dim=0)  # for binary setting
        predictions = predictions.cpu()

        all_y_hats.append(y_hat.numpy().item())
        all_predictions.append(predictions.numpy().item())
        all_true.append(bag_label.numpy().item())
        # all_y_tiles = prediction_tiles.cpu().data.numpy()[0]
        attention_scores = np.round(attention.cpu().data.numpy()[0], decimals=3)
        all_attentions.append(attention_scores)
        # all_tiles_true.append(label_tiles.cpu().numpy()[0])
        original_bag_indices.append(int(original_bag_index.cpu()))

        all_tiles_true.append(label_tiles.cpu().numpy()[0])
        all_y_tiles_binarized.append(
            np.asarray([int(prediction_tiles_binarized[i].cpu()) for i in range(len(prediction_tiles_binarized))]))
        all_y_tiles.append(
            np.asarray([float(prediction_tiles[i].cpu()) for i in range(len(prediction_tiles))]))

        # log.write('Bag Label:' + str(bag_label))
        # log.write('Predicted Label:' + str(predictions.numpy().item()))
        # log.write('attention scores (unique ones):')
        # log.write(attention_scores)

        del data, bag_label, label, label_tiles, prediction_tiles_binarized, prediction_tiles

    if verbose:
        log.write('Predicting bags: Done.')

    return all_y_hats, all_predictions, all_true, all_y_tiles, all_y_tiles_binarized, all_tiles_true, all_attentions, original_bag_indices


@torch.no_grad()
def evaluate(model: OmniSpheroMil, data_loader: OmniSpheroDataLoader, clamp_max: float = None, clamp_min: float = None,
             apply_data_augmentation=False):
    ''' Evaluate model / validation operation
    Can be used for validation within fit as well as testing.
    '''
    model.eval()
    test_losses = []
    test_acc = []
    test_acc_tiles = []
    acc_tiles_list_list = []
    tiles_prediction_list_list = []
    val_FP = 0
    val_TP = 0
    val_FN = 0
    val_TN = 0

    bag_label_list = []
    y_hat_list = []

    result = {}
    attention_weights = None

    for batch_id, (data, label, tile_labels, original_bag_index) in enumerate(data_loader):
        label = label.squeeze()
        bag_label = label

        data = data.to(model.get_device_ordinal(0))
        bag_label = bag_label.to(model.get_device_ordinal(3))
        tile_labels = tile_labels.to(model.get_device_ordinal(3))

        loss, attention_weights = model.compute_loss(data, bag_label)  # forward pass
        if clamp_min is not None:
            loss = torch.clamp(loss, min=clamp_min)
        if clamp_max is not None:
            loss = torch.clamp(loss, max=clamp_max)

        # https://github.com/yhenon/pytorch-retinanet/issues/3
        test_losses.append(float(loss))
        acc, acc_tiles, acc_tiles_list, tiles_prediction_list, y_hat, y_hat_binarized = model.compute_accuracy(data,
                                                                                                               bag_label,
                                                                                                               tile_labels)
        test_acc_tiles.append(float(acc_tiles))
        acc_tiles_list_list.append(acc_tiles_list)
        tiles_prediction_list_list.append(tiles_prediction_list)
        test_acc.append(float(acc))

        # Checking if prediction is TP / FP ... etc
        binary_label = bool(label)
        binary_prediction = bool(y_hat_binarized)
        if binary_label:
            if binary_prediction:
                val_TP += 1
            else:
                val_FP += 1
        else:
            if binary_prediction:
                val_FN += 1
            else:
                val_TN += 1

        bag_label_list.append(float(label))
        y_hat_list.append(float(y_hat))

        del data, bag_label, tile_labels, label, y_hat

    fpr, tpr, thresholds = mil_metrics.binary_roc_curve(bag_label_list, y_hat_list)
    roc_auc = float('NaN')
    try:
        roc_auc = auc(fpr, tpr)
    except Exception as e:
        log.write(str(e))

    result['val_dice_score'] = mil_metrics.calculate_dice_score(TP=val_TP, FP=val_FP, FN=val_FN)
    result['val_FP'] = val_FP
    result['val_TP'] = val_TP
    result['val_FN'] = val_FN
    result['val_TN'] = val_TN
    result['val_roc_auc'] = roc_auc
    result['val_loss'] = sum(test_losses) / len(test_losses)  # torch.stack(test_losses).mean().item()
    result['val_acc'] = sum(test_acc) / len(test_acc)
    result['val_acc_tiles'] = sum(test_acc_tiles) / len(test_acc_tiles)

    return result, attention_weights, test_losses, acc_tiles_list_list, tiles_prediction_list_list


# deprecated
def binary_cross_entropy(y: Union[float, list], y_predicted: Union[float, list]) -> float:
    if not isinstance(y, list):
        y = [y]
    if not isinstance(y_predicted, list):
        y_predicted = [y_predicted]

    losses = []
    for i in range(len(y)):
        loss = y[i] * math.log(y_predicted[i]) + (1 - y[i]) * math.log(1 - y_predicted[i])
        losses.append(loss * -1)

    return float(np.mean(losses))


def _binary_accuracy(outputs: Tensor, targets: Tensor) -> Tensor:
    # assert targets.size() == outputs.size()
    y_prob: Tensor = torch.ge(outputs, 0.5).float()
    return (targets == y_prob).sum().item() / targets.size(0)


def debug_all_models(gpu_enabled: bool = True):
    device = hardware.get_hardware_device(gpu_preferred=gpu_enabled)
    print('Selected device: ' + str(device))

    print('Checking the baseline model')
    m = BaselineMIL()
    print(m)


def predict_dose_response(experiment_holder: dict, experiment_name: str, model,
                          max_workers: int = 1, verbose: bool = False):
    if verbose:
        log.write('Predicting experiment: ' + experiment_name)
    well_indices = list(experiment_holder.keys())
    well_indices.sort()

    prediction_dict_bags = {}
    prediction_dict_samples = {}
    prediction_dict_well_names = {}
    all_wells = []

    # Iterating over all well numbers
    for current_well_index in well_indices:
        well_letters = list(experiment_holder[current_well_index])
        well_letters.sort()

        if verbose:
            log.write(
                experiment_name + ' - Discovered ' + str(len(well_letters)) + ' replicas(s) for well index ' + str(
                    current_well_index) + ': ' + str(well_letters))

        replica_predictions_bags = []
        replica_predictions_samples = []
        replica_well_names = []

        # Iterating over all replcas for this concentration and adding raw predictions to the list
        for current_well_letter in well_letters:
            current_well = current_well_letter + str(current_well_index)
            if verbose:
                log.write('Predicting: ' + experiment_name + ' - ' + current_well)

            if current_well not in all_wells:
                all_wells.append(current_well)

            current_x = [experiment_holder[current_well_index][current_well_letter]]
            dataset, bag_input_dim = loader.convert_bag_to_batch(bags=current_x, labels=None, y_tiles=None)
            predict_dl = OmniSpheroDataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False,
                                              num_workers=max_workers,
                                              transform_enabled=False, transform_data_saver=False)
            del current_x, dataset
            if verbose:
                log.write('Model input dim: ' + str(model.input_dim))
                log.write('Loaded data input dim: ' + str(bag_input_dim))
            assert bag_input_dim == model.input_dim

            all_y_hats, all_predictions, all_true, all_y_tiles, all_y_tiles_binarized, all_tiles_true, all_attentions, original_bag_indices = get_predictions(
                model, predict_dl, verbose=False)
            del predict_dl

            if verbose:
                log.write('Finished predicting: ' + experiment_name + ' - ' + current_well)

            # Taking the means of the predictions and adding them to the list
            replica_predictions_bags.append(np.mean(all_y_hats))
            replica_predictions_samples.append(np.mean(all_y_tiles))
            replica_well_names.append(current_well)
            del all_y_hats, all_predictions, all_true, all_y_tiles, all_y_tiles_binarized, all_tiles_true, all_attentions, original_bag_indices

        # Adding the lists to the dict, so the dict will include means of the replicas
        prediction_dict_bags[current_well_index] = replica_predictions_bags
        prediction_dict_samples[current_well_index] = replica_predictions_samples
        prediction_dict_well_names[current_well_index] = replica_well_names
    del experiment_holder
    all_wells.sort()

    return all_wells, prediction_dict_bags, prediction_dict_samples, prediction_dict_well_names


if __name__ == "__main__":
    print('Debugging the models, to see if everything is available.')

    debug_all_models()

    print('Finished debugging.')
