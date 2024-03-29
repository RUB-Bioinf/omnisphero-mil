import math
import os
import sys
from datetime import datetime
from datetime import timedelta
from typing import Union

import numpy as np
from sklearn.metrics import auc

import hardware
import loader
import mil_metrics
import r
import torch_callbacks
import video_render_ffmpeg
from util import data_renderer
from util import log
from util import utils
from util.omnisphero_data_loader import OmniSpheroDataLoader
from util.paths import training_metrics_live_dir_name
from util.utils import line_print

# setting env before importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def save_model(state, save_path: str, verbose: bool = False):
    if verbose:
        log.write('Saving model: ' + save_path)
    torch.save(state, save_path)


device_ordinals_cpu = None
device_ordinals_local = [0, 0, 0, 0]
device_ordinals_ehrlich = [0, 1, 2, 3]
device_ordinals_cluster = [0, 1, 2, 3]


def build_single_card_device_ordinals(card_index: int):
    return [card_index, card_index, card_index, card_index]


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


##############
# MODEL CLASS
##############

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

        self.synchronize_gpu()
        return 'cuda:' + str(self._device_ordinals[index])

    def is_cpu(self) -> bool:
        return self.device.type == 'cpu'

    def get_device_ordinals(self) -> [int]:
        return self._device_ordinals.copy()

    def compute_loss(self, X: Tensor, y: Tensor) -> (Tensor, [Tensor]):
        # abstract function
        pass

    def compute_accuracy(self, X: Tensor, y: Tensor, y_tiles: Tensor):
        # abstract function
        pass

    def synchronize_gpu(self):
        torch.cuda.synchronize()

    def _sum_layers(self, input_dim):
        from torchsummary import summary
        summary(self, input_dim)


####

class BaselineMIL(OmniSpheroMil):
    def __init__(self, input_dim, device, loss_function: str, accuracy_function: str, use_bias=True, use_max=True,
                 device_ordinals=None, enable_attention: bool = False):
        super().__init__(device, device_ordinals)
        log.write('Creating "BaselineMIL". Device: ' + str(device) + '. Input Dim: ' + str(
            input_dim) + '. Device ordinals: ' + str(device_ordinals))

        self.linear_nodes: int = 512
        self.num_classes: int = 1  # 3
        self.input_dim = input_dim

        self.use_bias: bool = use_bias
        self.use_max: bool = use_max
        self.loss_function: str = loss_function
        self.accuracy_function: str = accuracy_function
        self.enable_attention: bool = enable_attention
        self.attention_nodes = 128

        self.loss_cache = None

        if self.enable_attention:
            self.use_max = False

        #################
        # SETUP LAYERS
        #################
        log.write('Setting up Layers.')

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
        log.write('Finished: feature_extractor_0 on ' + str(self.get_device_ordinal(0)))

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
        log.write('Finished: feature_extractor_1 on ' + str(self.get_device_ordinal(1)))

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
        log.write('Finished: feature_extractor_2 on ' + str(self.get_device_ordinal(2)))

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
        log.write('Finished: feature_extractor_3 on ' + str(self.get_device_ordinal(3)))

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
        log.write('Finished: attention on ' + str(self.get_device_ordinal(3)))

        self.classifier = nn.Sequential(
            nn.Linear(self.linear_nodes, self.num_classes),  # * self.num_classes, self.num_classes),
            # nn.Softmax()
            nn.Sigmoid()
        )
        self.classifier = self.classifier.to(self.get_device_ordinal(3))
        log.write('Finished: classifier on ' + str(self.get_device_ordinal(3)))
        log.write('Finished all layers.')

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
        pooled = None
        attention = None
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
    def compute_loss(self, X: Tensor, y: Tensor) -> (Tensor, [Tensor]):
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

        cache = y_hat.cpu().detach().numpy()
        self.loss_cache = cache

        method = self.loss_function
        loss: Tensor = None
        if method == 'negative_log_bernoulli':
            loss = -1. * (
                    y * torch.log(y_prob) + (1.0 - y) * torch.log(1.0 - y_prob))  # negative log bernoulli loss
        elif method == 'binary_cross_entropy':
            # loss = y * torch.log(y_hat) + (1.0 - y) * torch.log(1.0 - y_hat)
            # loss = torch.clamp(loss, min=-1.0, max=1.0)
            # loss = loss * -1
            log.write('Calculating BCE Loss: ' + str(cache) + ', ' + str(y.cpu().detach().numpy()),
                      print_to_console=False)

            try:
                b = nn.BCELoss()
                loss = b(y_hat.squeeze(), y)
            except Exception as e:
                log.write(e)
                log.write('Cannot execute loss function!')
        elif method == 'mean_square_error':
            b = nn.MSELoss()
            loss = b(y_hat.squeeze(), y)
        else:
            raise Exception('Unknown loss function!')

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

        y_hat, y_hat_binarized, attention, y_hat_tiles, _ = self.forward(X)
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

        if not self.enable_attention:
            attention = None

        return bag_acc, tile_acc, tile_accuracy_list, tile_prediction_list, float(y_hat), y_hat_binarized, attention

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

    def sum_layers(self):
        self._sum_layers(self.input_dim)


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

def choose_optimizer(model: OmniSpheroMil, selection: str) -> (Optimizer, float):
    """ Chooses an optimizer according to the string specifed in the model CLI argument and build with specified args
    """
    initial_lr = float('NaN')
    if selection == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                     amsgrad=False)
        initial_lr = 0.001
    elif selection == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
        initial_lr = 1.0
    elif selection == 'momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay=0,
                                    nesterov=False)
        initial_lr = 0.01
    else:
        raise Exception("Error! Chosen optimizer or its parameters are unclear: '" + selection + "'")
    return optimizer, initial_lr


def fit(model: OmniSpheroMil, optimizer: Optimizer, epochs: int, training_data: OmniSpheroDataLoader, bag_names: [str],
        validation_data: OmniSpheroDataLoader, out_dir_base: str, callbacks: [torch_callbacks.BaseTorchCallback],
        checkpoint_interval: int = 1, clamp_min: float = None, clamp_max: float = None,
        save_sigmoid_plot_interval: int = 5, data_loader_sigmoid: OmniSpheroDataLoader = None,
        X_metadata_sigmoid: [np.ndarray] = None, augment_training_data: bool = False, hist_bins_override=None,
        sigmoid_video_render_enabled: bool = True, render_fps: int = video_render_ffmpeg.default_fps,
        sigmoid_evaluation_enabled: bool = False, augment_validation_data: bool = False):
    """ Trains a model on the previously preprocessed train and val sets.
    Also calls evaluate in the validation phase of each epoch.
    """
    best_loss = sys.float_info.max
    history = []
    history_keys = ['train_loss', 'train_acc', 'val_acc', 'val_loss', 'train_roc_auc', 'val_roc_auc',
                    'train_entropy_attention_label0', 'val_entropy_attention_label0', 'train_otsu_threshold_label0',
                    'val_otsu_threshold_label0', 'train_entropy_attention_label1', 'val_entropy_attention_label1',
                    'train_otsu_threshold_label1', 'val_otsu_threshold_label1', 'val_mean_sigmoid_scores',
                    'train_dice_score', 'val_dice_score', 'duration', 'timestamp']
    history_keys.sort()

    sigmoid_evaluation_enabled: bool = sigmoid_evaluation_enabled and X_metadata_sigmoid is not None
    log.write('Training a new model. Sigmoid validation enabled: ' + str(sigmoid_evaluation_enabled))
    log.write('Training duration: ' + str(epochs) + ' epochs.')

    checkpoint_out_dir = out_dir_base + 'checkpoints' + os.sep
    metrics_dir_live = out_dir_base + training_metrics_live_dir_name + os.sep
    epoch_data_dir_live = metrics_dir_live + 'epochs_live' + os.sep
    sigmoid_data_dir_live = metrics_dir_live + 'sigmoid_live' + os.sep
    sigmoid_data_dir_live_best = sigmoid_data_dir_live + 'best' + os.sep
    sigmoid_data_dir_naive_live = sigmoid_data_dir_live + 'naive' + os.sep
    sigmoid_data_dir_naive_live_best = sigmoid_data_dir_naive_live + 'best' + os.sep
    os.makedirs(checkpoint_out_dir, exist_ok=True)
    os.makedirs(metrics_dir_live, exist_ok=True)
    os.makedirs(epoch_data_dir_live, exist_ok=True)
    os.makedirs(sigmoid_data_dir_live, exist_ok=True)
    os.makedirs(sigmoid_data_dir_live_best, exist_ok=True)
    os.makedirs(sigmoid_data_dir_naive_live, exist_ok=True)
    os.makedirs(sigmoid_data_dir_naive_live_best, exist_ok=True)

    best_epoch_log = metrics_dir_live + 'best_epochs.txt'
    f = open(best_epoch_log, 'w')
    f.write('Best epochs:')
    f.close()

    # Writing Live Loss CSV
    batch_headers = ';'.join(
        ['Batch ' + str(batch_id) for batch_id, (data, label, tile_labels, batch_index) in enumerate(training_data)])
    batch_losses_file = metrics_dir_live + 'batch_losses.csv'
    f = open(batch_losses_file, 'w')
    f.write('Epoch;' + batch_headers)
    f.close()

    # Setting up sigmoid CSV
    batch_sigmoid_evaluation_error_file = sigmoid_data_dir_live + 'sigmoid_evaluations_errors.txt'
    batch_sigmoid_evaluation_file = sigmoid_data_dir_live + 'sigmoid_evaluations.csv'
    if not sigmoid_evaluation_enabled:
        f = open(batch_sigmoid_evaluation_error_file, 'w')
        f.write('Not running sigmoid evaluations.')
        f.close()
    else:
        sigmoid_experiment_names: [str] = list(dict.fromkeys([m[0].experiment_name for m in X_metadata_sigmoid]))
        f = open(batch_sigmoid_evaluation_file, 'w')
        f.write('Epoch;')
        for sigmoid_experiment in sigmoid_experiment_names:
            f.write(sigmoid_experiment + ';')
            del sigmoid_experiment
        f.close()
        f = open(batch_sigmoid_evaluation_error_file, 'w')
        f.write('Errors:')
        f.close()

    # Setting up sigmoid instructions log file
    batch_sigmoid_instructions_file = sigmoid_data_dir_live + 'sigmoid_instructions.csv'
    f = open(batch_sigmoid_instructions_file, 'w')
    f.write('Epoch;Experiment;Best?;Instructions: Dose;Instructions: Response')
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
    sigmoid_render_out_dirs = []

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
        # Beginning of a new epoch!
        epoch = epoch + 1

        # TRAINING PHASE
        model.train()
        train_losses = []
        train_acc = []
        train_acc_tiles = []
        all_labels = []
        predicted_labels = []
        otsu_threshold_label0_list = []
        entropy_attention_label0_list = []
        otsu_threshold_label1_list = []
        entropy_attention_label1_list = []
        train_FP = 0
        train_TP = 0
        train_FN = 0
        train_TN = 0
        start_time_epoch = datetime.now()
        epochs_remaining = epochs - epoch
        # log.write(' ## DEBUG ##\nModel training callback optimizer state: ' + str(optimizer))

        for batch_id, (data, label, tile_labels, bag_index) in enumerate(training_data):
            # TODO check if running on GPU and clear cache
            # torch.cuda.empty_cache()

            # Notifying Callbacks
            for i in range(len(callbacks)):
                callback: torch_callbacks.BaseTorchCallback = callbacks[i]
                callback.on_batch_start(model=model, batch_id=batch_id, data=data, label=label)

            label = label.squeeze()
            bag_label = label
            bag_name = bag_names[int(bag_index)]

            # writing data from the batch
            f = open(data_batch_dirs[batch_id], 'a')
            f.write('Epoch: ' + str(epoch))
            f.write('\nTile Index;Tile Hash;Bag Label;Finite array;Bag Name')
            for i in range(data.shape[1]):
                tile = data[0, i].cpu()
                tile_finite = tile.isfinite().numpy()
                tile_finite = bool(tile_finite.all())

                # Checking if the tile in this bag is finite!
                if not tile_finite:
                    warn_text = 'WARNING! Not all tiles in this bag are completely finite! Epoch: ' + str(
                        epoch) + '. Batch: ' + str(batch_id) + '. Bag: ' + str(int(bag_index)) + '. CSV index: ' + str(
                        i) + '. Bag Name: ' + bag_name
                    log.write(warn_text)
                    error_dir = data_batch_dirs[batch_id][:-4] + '-errors'
                    os.makedirs(error_dir, exist_ok=True)

                    out_name = 'data_ep' + str(epoch) + '_bag' + str(int(bag_index)) + '_batch' + str(batch_id)
                    torch.save(data, error_dir + os.sep + out_name + '.prt')
                    torch.save(label.cpu(), error_dir + os.sep + out_name + '-label.prt')

                    data_list = data.tolist()
                    np.save(error_dir + os.sep + out_name + '.npy', np.asarray(data_list))
                    with open(error_dir + os.sep + out_name + '.txt', 'w') as err:
                        err.write(str(data_list))
                    with open(error_dir + os.sep + out_name + '-meta.txt', 'w') as err:
                        err.write(str(warn_text))
                    del error_dir, out_name, data_list, warn_text

                tile = tile.numpy()
                tile_label = tile_labels[0, i].numpy()

                f.write('\n' + str(i) + ';' + np.format_float_positional(hash(str(tile))) + ';' + str(
                    label.numpy()) + ';' + str(tile_label) + ';' + str(tile_finite) + ';' + bag_name)
                del i, tile, tile_finite, tile_label
            f.write('\n\n')
            f.close()

            data = data.to(model.get_device_ordinal(0))
            bag_label = bag_label.to(model.get_device_ordinal(3))
            tile_labels = tile_labels.to(model.get_device_ordinal(3))

            # resets gradients
            model.zero_grad()

            cuda_device_error = False
            loss = None
            try:
                loss, _ = model.compute_loss(X=data, y=bag_label)  # forward pass
                if loss is None:
                    log.write("WARNING: Loss is None in epoch " + str(epoch) + "!")
                    log.write('Did you forget to define a loss function?')
                    # If you reach here, the loss is "None". That mostly means, there is no loss function defined??
                    cuda_device_error = True
            except Exception as e:
                log.write('Error while getting the loss for bag: ' + bag_name)
                log.write(str(e))
                cuda_device_error = True

            model_loss_cache = model.loss_cache
            try:
                torch.isfinite(loss)
                str(loss)
                loss.item()
            except Exception as e:
                cuda_device_error = True
                log.write('#### LOSS ERROR ON THE DEVICE ###')
                log.write('cached Loss: ' + str(model_loss_cache))
                log.write('(raw) Loss: ' + str(loss))
                log.write('Exception: "' + str(e) + '"')

            device_loss: float = float('NaN')
            if cuda_device_error:
                log.write('loss error in epoch ' + str(epoch) + ', batch ' + str(batch_id) + '! Bag name: ' + bag_name)
                error_dir = out_dir_base + os.sep + 'error_recovery-ep' + str(epoch) + '-batch-' + str(
                    batch_id) + os.sep
                os.makedirs(error_dir, exist_ok=True)
                log.write('Saving errors here: ' + error_dir)

                # writing the recovery files as pytorch tensors
                torch.save(data, error_dir + 'data.prt')
                torch.save(data.isfinite(), error_dir + 'data_finite.prt')
                torch.save(label, error_dir + 'label.prt')

                data_list = data.tolist()
                data_finite_list = data.isfinite().tolist()
                label_list = label.tolist()

                # writing recovery files as numpy formatted files
                np.save(error_dir + 'data.npy', np.asarray(data_list))
                np.save(error_dir + 'data_finite.npy', np.asarray(data_finite_list))
                np.save(error_dir + 'label.npy', np.asarray(label_list))

                # writing the recovery files in the log
                log.write(str(label_list), print_to_console=False)
                log.write(str(data_list), print_to_console=False)
                log.write(str(data_finite_list), print_to_console=False)

                # writing recovery files as text files
                with open(error_dir + 'data.txt', 'w') as f:
                    f.write(str(data_list))
                with open(error_dir + 'data_finite.txt', 'w') as f:
                    f.write(str(data_finite_list))
                with open(error_dir + 'label.txt', 'w') as f:
                    f.write(str(label_list))
                with open(error_dir + 'raw_loss_cache.txt', 'w') as f:
                    f.write(str(model_loss_cache))

                if loss is not None:
                    torch.save(loss, error_dir + 'loss.prt')
                    loss_list = loss.tolist()
                    np.save(error_dir + 'loss.npy', np.asarray(loss_list))
                    log.write(str(loss_list), print_to_console=False)
                    with open(error_dir + 'loss.txt', 'w') as f:
                        f.write(str(loss_list))
            else:
                log.write('loss epoch ' + str(epoch) + ', batch ' + str(batch_id) + ': "' + str(loss) + '".',
                          print_to_console=False)

                # Clamping the loss
                # https://github.com/yhenon/pytorch-retinanet/issues/3
                if clamp_min is not None:
                    loss = torch.clamp(loss, min=clamp_min)
                if clamp_max is not None:
                    loss = torch.clamp(loss, max=clamp_max)

                # Moving the loss from a device tensor to float in RAM
                try:
                    if torch.isfinite(loss):
                        # device_loss = float(torch.Tensor.float(loss).cpu().detach().numpy())
                        device_loss = float(loss.item())
                    else:
                        log.write(' ==== LOSS WARNING === Loss is not finite: ' + str(loss))
                except Exception as e:
                    log.write('#### LOSS ERROR ###\n=======================\nFailed to move loss from tensor to float!')
                    log.write('(raw) Loss: ' + str(loss))
                    log.write('Exception: "' + str(e) + '"')
                log.write('Loss processing: "' + str(loss) + '" -> "' + str(device_loss) + '".', print_to_console=False)

            train_losses.append(float(device_loss))
            acc, acc_tiles, _, _, y_hat, y_hat_binarized, attention = model.compute_accuracy(data, bag_label,
                                                                                             tile_labels)
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

            # Evaluating Attention Histogram
            if model.enable_attention:
                attention = attention.cpu().squeeze().detach().numpy()
                _, _, _, otsu_threshold, entropy_attention, _ = mil_metrics.attention_metrics(attention=attention,
                                                                                              hist_bins_override=hist_bins_override,
                                                                                              normalized=True)
            else:
                otsu_threshold = float('NaN')
                entropy_attention = float('NaN')

            # Adding the params to the respective lists, based on the current label
            if label == 1:
                otsu_threshold_label0_list.append(otsu_threshold)
                entropy_attention_label0_list.append(entropy_attention)
            else:
                otsu_threshold_label1_list.append(otsu_threshold)
                entropy_attention_label1_list.append(entropy_attention)

            # Notifying Callbacks
            for i in range(len(callbacks)):
                callback: torch_callbacks.BaseTorchCallback = callbacks[i]
                callback.on_batch_finished(model=model, batch_id=batch_id, data=data, label=label, batch_acc=float(acc),
                                           batch_loss=float(device_loss))

            loss.backward()  # backward pass
            optimizer.step()  # update parameters
            # optim.zero_grad() # reset gradients (alternative if all grads are contained in the optimizer)
            # for p in model.parameters(): p.grad=None # alternative for model.zero_grad() or optim.zero_grad()
            del data, bag_label, label, y_hat, attention, otsu_threshold, entropy_attention, device_loss

        # VALIDATION PHASE
        result, _, all_losses, all_tile_lists, _ = evaluate(model, validation_data, clamp_max=clamp_max,
                                                            clamp_min=clamp_min,
                                                            hist_bins_override=hist_bins_override,
                                                            apply_data_augmentation=augment_validation_data)  # returns a results dict for metrics

        # Sigmoid evaluation for this epoch
        sigmoid_mean = float('nan')
        sigmoid_successes = 0
        sigmoid_scores_sanitized = None
        sigmoid_scores_raw = None
        y_hats_sigmoid = None
        val_mean_sigmoid_scores = float('nan')
        if sigmoid_evaluation_enabled:
            save_sigmoid_plot = epoch % save_sigmoid_plot_interval == 0 or epoch == 1

            if r.has_connection() and X_metadata_sigmoid is not None and data_loader_sigmoid is not None:
                y_hats_sigmoid, _, _, _, _, _, _, _ = get_predictions(model, data_loader_sigmoid)
                sigmoid_score_map, sigmoid_score_detail_map, sigmoid_plot_estimation_map, sigmoid_plot_data_map, sigmoid_instructions_map, sigmoid_bmc30_map = r.prediction_sigmoid_evaluation(
                    X_metadata=X_metadata_sigmoid,
                    y_pred=y_hats_sigmoid,
                    save_sigmoid_plot=False,
                    file_name_suffix='-epoch' + str(epoch),
                    out_dir=sigmoid_data_dir_live)

                f = open(batch_sigmoid_instructions_file, 'a')
                for key in sigmoid_instructions_map.keys():
                    sigmoid_instructions = sigmoid_instructions_map[key]
                    f.write('\n' + str(epoch) + ';'
                            + key + ';False;' + sigmoid_instructions[0] + ';' + sigmoid_instructions[1])
                    del key, sigmoid_instructions
                f.close()

                if save_sigmoid_plot:
                    all_render_out_dirs = data_renderer.render_response_curves(X_metadata=X_metadata_sigmoid,
                                                                               y_pred=y_hats_sigmoid,
                                                                               sigmoid_score_map=sigmoid_score_map,
                                                                               dpi=350,
                                                                               sigmoid_plot_estimation_map=sigmoid_plot_estimation_map,
                                                                               sigmoid_plot_fit_map=sigmoid_plot_data_map,
                                                                               sigmoid_score_detail_map=sigmoid_score_detail_map,
                                                                               sigmoid_bmc30_map=sigmoid_bmc30_map,
                                                                               file_name_suffix='-epoch' + str(epoch),
                                                                               title_suffix='\nTraining Epoch ' + str(
                                                                                   epoch),
                                                                               out_dir=sigmoid_data_dir_naive_live)

                    # The maybe newly created dirs are added to the list of known sigmoid render dirs
                    for all_render_out_dir in all_render_out_dirs:
                        if all_render_out_dir not in sigmoid_render_out_dirs:
                            sigmoid_render_out_dirs.append(all_render_out_dir)
                    del all_render_out_dirs

                try:
                    sigmoid_scores_raw = list(sigmoid_score_map.values())
                    sigmoid_scores_sanitized = []
                    for score in sigmoid_scores_raw:
                        if not math.isnan(score):
                            sigmoid_scores_sanitized.append(score)

                    sigmoid_successes = len(sigmoid_scores_sanitized)
                    if sigmoid_successes == 0:
                        sigmoid_mean = float('nan')
                        log.write('WARNING: ALL SIGMOID FITS FAILED!')
                    else:
                        sigmoid_mean = np.mean(np.asarray(sigmoid_scores_sanitized))
                    log.write('Sigmoid successes: ' + str(sigmoid_successes) + '/' + str(
                        len(sigmoid_scores_raw)) + '. Score: ' + str(sigmoid_mean))

                    f = open(batch_sigmoid_evaluation_file, 'a')
                    f.write('\n' + str(epoch))
                    for key in sigmoid_score_map.keys():
                        f.write(';' + str(sigmoid_score_map[key]))
                    f.write(';' + str(sigmoid_mean))

                    val_mean_sigmoid_scores = sigmoid_mean
                    f.close()
                except Exception as e:
                    err_text = 'FATAL ERROR: Sigmoid evaluation failed! Reason: "' + str(e) + '"'
                    log.write(err_text)
                    f = open(batch_sigmoid_evaluation_error_file, 'a')
                    f.write('\nEpoch: ' + str(epoch) + ' - "' + str(e) + '"')
                    f.close()

                del sigmoid_mean, sigmoid_successes, sigmoid_scores_sanitized, sigmoid_scores_raw
            else:
                log.write(
                    'Warning: Not running sigmoid evaluation. Data missing or no rServe connection.Connected: ' + str(
                        r.has_connection()))
                log.write('X_metadata_sigmoid is None: ' + str(X_metadata_sigmoid is None))
                log.write('data_loader_sigmoid is None: ' + str(data_loader_sigmoid is None))
                f = open(batch_sigmoid_evaluation_error_file, 'a')
                f.write('\nEpoch: ' + str(epoch) + ' - Data missing or no rServe connection. Connected: ' + str(
                    r.has_connection()) + '. X_metadata_sigmoid is None: ' + str(
                    X_metadata_sigmoid is None) + '. data_loader_sigmoid is None: ' + str(data_loader_sigmoid is None))
                f.close()

        # ROC curve for this epoch
        fpr, tpr, thresholds = mil_metrics.binary_roc_curve(all_labels, predicted_labels)
        roc_auc = float('NaN')
        try:
            roc_auc = auc(fpr, tpr)
        except Exception as e:
            log.write(str(e))
        result['train_roc_auc'] = roc_auc

        # Adding the sigmoid scores to the results dict.
        # NOTE: This says "VAL" and that's correct, because it comes from a separate validation set
        result['val_mean_sigmoid_scores'] = val_mean_sigmoid_scores

        if len(otsu_threshold_label0_list) == 0:
            otsu_threshold_label0_list = [float('NaN')]
        if len(entropy_attention_label0_list) == 0:
            entropy_attention_label0_list = [float('NaN')]
        if len(otsu_threshold_label1_list) == 0:
            otsu_threshold_label1_list = [float('NaN')]
        if len(entropy_attention_label1_list) == 0:
            entropy_attention_label1_list = [float('NaN')]
        assert len(entropy_attention_label1_list) == len(otsu_threshold_label1_list)
        assert len(entropy_attention_label0_list) == len(otsu_threshold_label0_list)

        # Adding (mean) evaluations for this epoch
        result['train_otsu_threshold_label0'] = sum(otsu_threshold_label0_list) / len(otsu_threshold_label0_list)
        result['train_entropy_attention_label0'] = sum(entropy_attention_label0_list) / len(
            entropy_attention_label0_list)
        result['train_otsu_threshold_label1'] = sum(otsu_threshold_label1_list) / len(otsu_threshold_label1_list)
        result['train_entropy_attention_label1'] = sum(entropy_attention_label1_list) / len(
            entropy_attention_label1_list)
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
        result['timestamp'] = str(utils.gct())

        remaining_time_eta = timedelta(seconds=np.mean(epoch_durations) * epochs_remaining)
        eta_timestamp = datetime.now() + remaining_time_eta
        eta_timestamp = eta_timestamp.strftime("%Y/%m/%d, %H:%M:%S")

        history.append(result)
        log.write(
            'Epoch {}/{}: Train Loss: {:.4f}, Train Acc (Bags): {:.4f}, Val Loss: {:.4f}, '
            'Val Acc (Bags): {:.4f}, Sigmoid Scores: {:.4f}, Train AUC: {:.4f}, Val AUC: {:.4f}, '
            'Train Otsu 0: {:.4f}, Val Otsu 0: {:.4f}, Train Entropy 0: {:.4f}, Val Entropy 0: {:.4f}. '
            'Train Otsu 1: {:.4f}, Val Otsu 1: {:.4f}, Train Entropy 1: {:.4f}, Val Entropy 1: {:.4f}. '
            'Duration: {}. ETA: {}'.format(
                epoch, epochs, result['train_loss'], result['train_acc'], result['val_loss'],
                result['val_acc'], result['val_mean_sigmoid_scores'], result['train_roc_auc'], result['val_roc_auc'],
                result['train_otsu_threshold_label0'], result['val_otsu_threshold_label0'],
                result['train_entropy_attention_label0'], result['val_entropy_attention_label0'],
                result['train_otsu_threshold_label1'], result['val_otsu_threshold_label1'],
                result['train_entropy_attention_label1'], result['val_entropy_attention_label1'], time_diff,
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
            mil_metrics.plot_accuracy_bags(history, metrics_dir_live, include_raw=False, include_tikz=False,
                                           include_line_fit=False)
            mil_metrics.plot_accuracy_tiles(history, metrics_dir_live, include_raw=False, include_tikz=False,
                                            include_line_fit=False)
            mil_metrics.plot_losses(history, metrics_dir_live, include_raw=False, include_tikz=False,
                                    include_line_fit=False)
            mil_metrics.plot_accuracies(history, metrics_dir_live, include_tikz=False, include_line_fit=False)
            mil_metrics.plot_dice_scores(history, metrics_dir_live, include_tikz=False, include_line_fit=False)
            mil_metrics.plot_sigmoid_scores(history, metrics_dir_live, include_tikz=False, include_line_fit=False)
            mil_metrics.plot_binary_roc_curves(history, metrics_dir_live, include_tikz=False)

            if model.enable_attention:
                mil_metrics.plot_attention_otsu_threshold(history, metrics_dir_live, label=0, include_tikz=False)
                mil_metrics.plot_attention_entropy(history, metrics_dir_live, label=0, include_tikz=False)
                mil_metrics.plot_attention_otsu_threshold(history, metrics_dir_live, label=1, include_tikz=False)
                mil_metrics.plot_attention_entropy(history, metrics_dir_live, label=1, include_tikz=False)
        except Exception as e:
            log.write('Failed to write metrics for this epoch.')
            log.write(str(e))

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

            # Updating best epochs file
            f = open(best_epoch_log, 'a')
            f.write('\n' + str(utils.gct()) + ' - Epoch: ' + str(epoch))
            f.close()

            # Rendering the sigmoid curves again, because of new best performance
            if r.has_connection() and X_metadata_sigmoid is not None and data_loader_sigmoid is not None and y_hats_sigmoid is not None:
                sigmoid_score_map, sigmoid_score_detail_map, sigmoid_plot_estimation_map, sigmoid_plot_data_map, sigmoid_instructions_map, sigmoid_bmc30_map = r.prediction_sigmoid_evaluation(
                    X_metadata=X_metadata_sigmoid,
                    y_pred=y_hats_sigmoid,
                    save_sigmoid_plot=False,
                    file_name_suffix='-best-epoch' + str(epoch),
                    out_dir=sigmoid_data_dir_live_best)

                # Writing it to the CSV
                f = open(batch_sigmoid_instructions_file, 'a')
                for key in sigmoid_instructions_map.keys():
                    sigmoid_instructions = sigmoid_instructions_map[key]
                    f.write(
                        '\n' + str(epoch) + ';' + key + ';True;' + sigmoid_instructions[0] + ';' + sigmoid_instructions[
                            1])
                    del key, sigmoid_instructions
                f.close()

                # Rendering the data
                all_render_out_dirs = data_renderer.render_response_curves(X_metadata=X_metadata_sigmoid,
                                                                           y_pred=y_hats_sigmoid,
                                                                           sigmoid_score_map=sigmoid_score_map, dpi=350,
                                                                           sigmoid_plot_estimation_map=sigmoid_plot_estimation_map,
                                                                           sigmoid_plot_fit_map=sigmoid_plot_data_map,
                                                                           sigmoid_score_detail_map=sigmoid_score_detail_map,
                                                                           sigmoid_bmc30_map=sigmoid_bmc30_map,
                                                                           file_name_suffix='-best-epoch' + str(epoch),
                                                                           title_suffix='\nTraining Epoch ' + str(
                                                                               epoch) + ' (New Best)',
                                                                           out_dir=sigmoid_data_dir_naive_live_best)

                del sigmoid_score_map, sigmoid_plot_estimation_map, sigmoid_plot_data_map, sigmoid_score_detail_map
                # The maybe newly created dirs are added to the list of known sigmoid render dirs
                for all_render_out_dir in all_render_out_dirs:
                    if all_render_out_dir not in sigmoid_render_out_dirs:
                        sigmoid_render_out_dirs.append(all_render_out_dir)
                del all_render_out_dirs
        del y_hats_sigmoid

        if cancel_requested:
            log.write('Model was canceled before reaching all epochs.')

    ###################
    # TRAINING FINISHED
    ###################

    # Rendering video files
    log.write('Newly created sigmoid dirs:')
    for sigmoid_render_out_dir in sigmoid_render_out_dirs:
        log.write('  # ' + str(sigmoid_render_out_dir))
    log.write('End of list.')

    if sigmoid_video_render_enabled and len(sigmoid_render_out_dirs) > 0:
        try:
            # video_render.render_images_to_video_multiple(image_paths=sigmoid_render_out_dirs, fps=render_fps, verbose=True)
            video_render_ffmpeg.render_image_dir_to_video_multiple(image_paths=sigmoid_render_out_dirs, fps=render_fps,
                                                                   verbose=True)
        except Exception as e:
            log.write(str(e))
            log.write("ERROR! FAILED TO RENDER VIDEO!")
    else:
        log.write('Video render not done. Enabled: ' + str(sigmoid_video_render_enabled) + '.')

    # Notifying callbacks
    for i in range(len(callbacks)):
        callback: torch_callbacks.BaseTorchCallback = callbacks[i]
        callback.on_training_finished(model=model, was_canceled=cancel_requested, history=history)

    return history, history_keys, model_save_path_best


@torch.no_grad()
def get_predictions(model: OmniSpheroMil, data_loader: DataLoader, verbose: bool = False,
                    attention_round_decimals: int = None):
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

        # Predictions
        all_y_hats.append(y_hat.numpy().item())
        all_predictions.append(predictions.numpy().item())
        all_true.append(bag_label.numpy().item())
        # all_y_tiles = prediction_tiles.cpu().data.numpy()[0]

        # Attention
        attention_scores = attention.cpu().data.numpy()[0]
        if attention_round_decimals is not None:
            attention_round_decimals = int(attention_round_decimals)
            attention_scores = np.round(attention_scores, decimals=attention_round_decimals)
        all_attentions.append(attention_scores)
        # all_tiles_true.append(label_tiles.cpu().numpy()[0])

        # Bag indices
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
def evaluate(model: OmniSpheroMil, data_loader: OmniSpheroDataLoader, hist_bins_override: int = None,
             clamp_max: float = None, clamp_min: float = None, apply_data_augmentation=False):
    ''' Evaluate model / validation operation
    Can be used for validation within fit as well as testing.
    '''
    model.eval()
    test_losses = []
    test_acc = []
    test_acc_tiles = []
    otsu_threshold_label0_list = []
    entropy_attention_label0_list = []
    otsu_threshold_label1_list = []
    entropy_attention_label1_list = []
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
        if loss is None:
            log.write(' =================')
            log.write('WARNING: Validation loss is none!')
            log.write('Check if you picked the right loss function and optimizer!')
            log.write(' =================')

        if clamp_min is not None:
            loss = torch.clamp(loss, min=clamp_min)
        if clamp_max is not None:
            loss = torch.clamp(loss, max=clamp_max)

        # https://github.com/yhenon/pytorch-retinanet/issues/3
        test_losses.append(float(loss))
        acc, acc_tiles, acc_tiles_list, tiles_prediction_list, y_hat, y_hat_binarized, attention = model.compute_accuracy(
            data,
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

        # Appending evaluations to lists
        bag_label_list.append(float(label))
        y_hat_list.append(float(y_hat))

        # Evaluating Attention Histogram
        if model.enable_attention:
            attention = attention.cpu().squeeze().numpy()
            _, _, _, otsu_threshold, entropy_attention, _ = mil_metrics.attention_metrics(attention=attention,
                                                                                          hist_bins_override=hist_bins_override,
                                                                                          normalized=True)
        else:
            otsu_threshold = float('NaN')
            entropy_attention = float('NaN')

        if label == 1:
            otsu_threshold_label1_list.append(otsu_threshold)
            entropy_attention_label1_list.append(entropy_attention)
        else:
            otsu_threshold_label0_list.append(otsu_threshold)
            entropy_attention_label0_list.append(entropy_attention)

        del data, bag_label, tile_labels, label, y_hat, attention, otsu_threshold, entropy_attention

    fpr, tpr, thresholds = mil_metrics.binary_roc_curve(bag_label_list, y_hat_list)
    roc_auc = float('NaN')
    try:
        roc_auc = auc(fpr, tpr)
    except Exception as e:
        log.write(str(e))

    if len(otsu_threshold_label0_list) == 0:
        otsu_threshold_label0_list = [float('NaN')]
    if len(entropy_attention_label0_list) == 0:
        entropy_attention_label0_list = [float('NaN')]
    if len(otsu_threshold_label1_list) == 0:
        otsu_threshold_label1_list = [float('NaN')]
    if len(entropy_attention_label1_list) == 0:
        entropy_attention_label1_list = [float('NaN')]
    assert len(entropy_attention_label1_list) == len(otsu_threshold_label1_list)
    assert len(entropy_attention_label0_list) == len(otsu_threshold_label0_list)

    result['val_otsu_threshold_label0'] = sum(otsu_threshold_label0_list) / len(entropy_attention_label0_list)
    result['val_entropy_attention_label0'] = sum(entropy_attention_label0_list) / len(entropy_attention_label0_list)
    result['val_otsu_threshold_label1'] = sum(otsu_threshold_label1_list) / len(entropy_attention_label1_list)
    result['val_entropy_attention_label1'] = sum(entropy_attention_label1_list) / len(entropy_attention_label1_list)
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
    log.write('Debugging all models.')
    device = hardware.get_hardware_device(gpu_preferred=gpu_enabled)
    device_ordinals = build_single_card_device_ordinals(0)
    accuracy_function = 'binary'
    loss_function = 'mean_square_error'
    log.write('Selected device: ' + str(device))

    log.write('Checking the baseline model')
    m = BaselineMIL(input_dim=(3, 150, 150),
                    device=device,
                    loss_function=loss_function,
                    accuracy_function=accuracy_function,
                    enable_attention=True, use_max=False,
                    device_ordinals=device_ordinals)

    # Writing to the sdo
    print(m)
    log.write(str(m))

    # writing layer summary
    m.sum_layers()


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
