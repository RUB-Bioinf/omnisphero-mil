from itertools import chain

import torch
import torch.nn as nn
import os
import numpy as np

from torch.optim import Optimizer

from util.utils import get_hardware_device


####

def save_model(state, save_path):
    print('--> Saving new best model')
    torch.save(state, save_path)


####

def load_checkpoint(load_path, model, optimizer):
    ''' loads the model and its optimizer states from disk.
    '''
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss


# MODEL
#######

device_ordinals_local = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
device_ordinals_cluster = (0, 1, 2, 3, 3, 1, 2, 3, 0, 1, 2)


class BaselineMIL(nn.Module):
    def __init__(self, input_dim, device, use_bias=True, use_max=True, device_ordinals=None):
        super().__init__()
        self.linear_nodes = 512
        self.num_classes = 1  # 3
        self._device_ordinals = device_ordinals
        self.input_dim = input_dim
        self.device = device

        self.use_bias = use_bias
        self.use_max = use_max

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

        self.classifier = nn.Sequential(
            nn.Linear(1, self.num_classes),  # * self.num_classes, self.num_classes),
            # nn.Softmax()
            nn.Sigmoid()
        )
        self.classifier = self.classifier.to(self.get_device_ordinal(4))

    def forward(self, x):
        ''' Forward NN pass, declaring the exact interplay of model components
        '''
        x = x.squeeze(
            0)  # necessary? compresses unnecessary dimensions eg. (1,batch,channel,x,y) -> (batch,channel,x,y)
        hidden = self.feature_extractor_0(x)
        hidden = self.feature_extractor_1(hidden.to(self.get_device_ordinal(5)))
        hidden = self.feature_extractor_2(hidden.to(self.get_device_ordinal(6)))
        hidden = self.feature_extractor_3(hidden.to(self.get_device_ordinal(7)))  # N x linear_nodes

        if not self.use_max:
            pooled = torch.mean(hidden, dim=[0, 1], keepdim=True)  # N x num_classes

        elif self.use_max:
            pooled = torch.max(hidden)  # N x num_classes
            pooled = pooled.unsqueeze(dim=0)
            pooled = pooled.unsqueeze(dim=0)

        attention = torch.tensor([[0.5]])
        y_hat = self.classifier(pooled)
        y_hat_binarized = torch.ge(y_hat, 0.5).float()
        return y_hat, y_hat_binarized, attention

    def get_device_ordinal(self, index: int) -> str:
        if self._device_ordinals is None:
            return 'cpu'

        if self.device.type == 'cpu':
            return 'cpu'

        return 'cuda:' + str(self._device_ordinals(index))

    def _get_conv_output(self, shape):
        ''' generate a single fictional input sample and do a forward pass over
        Conv layers to compute the input shape for the Flatten -> Linear layers input size
        '''
        bs = 1
        test_input = torch.autograd.Variable(torch.rand(bs, *shape)).to(self.get_device_ordinal(8))
        output_features = self.feature_extractor_0(test_input)
        output_features = self.feature_extractor_1(output_features.to(self.get_device_ordinal(9)))
        output_features = self.feature_extractor_2(output_features.to(self.get_device_ordinal(10)))
        n_size = int(output_features.data.view(bs, -1).size(1))
        del test_input, output_features
        return n_size

    # COMPUTATION METHODS
    def compute_loss(self, X, y):
        ''' otherwise known as loss_fn
        Takes a data input of X,y (batches or bags) computes a forward pass and the resulting error.
        '''
        y = y.float()
        # y = y.unsqueeze(dim=0)
        # y = torch.argmax(y, dim=1)

        y_hat, y_hat_binarized, attention = self.forward(X)
        # y_prob = torch.ge(y_hat, 0.5).float() # for binary classification only. Rounds prediction output to 0 or 1
        y_prob = torch.clamp(y_hat, min=1e-5, max=1. - 1e-5)
        # y_prob = y_prob.squeeze(dim=0)

        loss = -1. * (y * torch.log(y_prob) + (1. - y) * torch.log(1. - y_prob))  # negative log bernoulli loss
        # loss = F.cross_entropy(y_hat, y)
        # loss = F.binary_cross_entropy(y_hat_binarized, y)
        # loss_func = nn.BCELoss()
        # loss = loss_func(y_hat, y)
        return loss, attention

    def compute_accuracy(self, X, y):
        ''' compute accuracy
        '''
        y = y.float()
        y = y.unsqueeze(dim=0)
        # y = torch.argmax(y, dim=1)

        y_hat, y_hat_binarized, _ = self.forward(X)
        # y_hat = torch.ge(y_hat, 0.5).float() # for binary classification only. Rounds prediction output to 0 or 1
        y_hat = y_hat.squeeze(dim=0)

        # acc = 1. - y_hat_binarized.eq(y).cpu().float().mean().item() # accuracy for neg. log bernoulli loss
        # acc = mil_metrics.multiclass_accuracy(y_hat, y)
        acc = _binary_accuracy(y_hat, y)
        return acc


def apply_optimizer(model: nn.Module, selection: str = 'adam') -> Optimizer:
    ''' Chooses an optimizer according to the string specifed in the model CLI argument and build with specified args
    '''
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


def fit(model, optimizer, epochs, training_data, validation_data, checkpoint_out_dir,
        device_ordinals=None):
    ''' Trains a model on the previously preprocessed train and val sets.
    Also calls evaluate in the validation phase of each epoch.
    '''
    best_acc = 0
    history = []
    loss = None
    os.makedirs(checkpoint_out_dir, exist_ok=True)
    model_save_path_best = checkpoint_out_dir + os.sep + 'model_best.h5'

    gpu_enabled = True
    if device_ordinals is None:
        print('Training on CPU.')
        gpu_enabled = False
    else:
        print('Training on GPU, using these ordinals: ' + str(device_ordinals))

    for epoch in range(1, epochs + 1):
        # TRAINING PHASE
        model.train()
        train_losses = []
        train_acc = []

        for batch_id, (data, label) in enumerate(training_data):
            # torch.cuda.empty_cache()

            label = label.squeeze()
            # bag_label = label[0] //TODO this causes error
            bag_label = label

            data = data.to(model.get_device_ordinal(0))
            bag_label = bag_label.to(model.get_device_ordinal(3))

            model.zero_grad()  # resets gradients

            loss, _ = model.compute_loss(data, bag_label)  # forward pass
            train_losses.append(float(loss))
            acc = model.compute_accuracy(data, bag_label)
            train_acc.append(float(acc))

            loss.backward()  # backward pass
            optimizer.step()  # update parameters
            # optim.zero_grad() # reset gradients (alternative if all grads are contained in the optimizer)
            # for p in model.parameters(): p.grad=None # alternative for model.zero_grad() or optim.zero_grad()
            del data, bag_label, label

        # VALIDATION PHASE
        result, _ = evaluate(model, validation_data)  # returns a results dict for metrics
        result['train_loss'] = sum(train_losses) / len(train_losses)  # torch.stack(train_losses).mean().item()
        result['train_acc'] = sum(train_acc) / len(train_acc)
        history.append(result)

        print('Epoch [{}] : Train Loss {:.4f}, Train Acc {:.4f}, Val Loss {:.4f}, Val Acc {:.4f}'.format(epoch, result[
            'train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))

        # Save best model / checkpointing stuff
        is_best = bool(result['val_acc'] >= best_acc)
        best_acc = max(result['val_acc'], best_acc)
        state = {
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }

        model_save_path_checkpoint = checkpoint_out_dir + os.sep + 'model_checkpoint-' + str(epoch) + '.h5'
        save_model(state, model_save_path_checkpoint)
        if is_best:
            save_model(state, model_save_path_best)

    return history, model_save_path_best


@torch.no_grad()
def get_predictions(model, data_loader, device_ordinal: [int] = None):
    ''' takes a trained model and validation or test dataloader
    and applies the model on the data producing predictions

    binary version
    '''
    model.eval()

    all_y_hats = []
    all_predictions = []
    all_true = []
    all_attention = []

    for batch_id, (data, label) in enumerate(data_loader):
        label = label.squeeze()
        # bag_label = label[0]
        bag_label = label
        bag_label = bag_label.cpu()

        data = data.to(model.get_device_ordinal(0))

        y_hat, predictions, attention = model(data)
        y_hat = y_hat.squeeze(dim=0)  # for binary setting
        y_hat = y_hat.cpu()
        predictions = predictions.squeeze(dim=0)  # for binary setting
        predictions = predictions.cpu()

        all_y_hats.append(y_hat.numpy().item())
        all_predictions.append(predictions.numpy().item())
        all_true.append(bag_label.numpy().item())
        attention_scores = np.round(attention.cpu().data.numpy()[0], decimals=3)
        all_attention.append(attention_scores)

        print('Bag Label:' + str(bag_label))
        print('Predicted Label:' + str(predictions.numpy().item()))
        print('attention scores (unique ones):')
        # print(np.unique(attention_scores))
        print(attention_scores)

        del data, bag_label, label

    return all_y_hats, all_predictions, all_true


@torch.no_grad()
def evaluate(model, data_loader):
    ''' Evaluate model / validation operation
    Can be used for validation within fit as well as testing.
    '''
    model.eval()
    test_losses = []
    test_acc = []
    result = {}
    attention_weights = None

    for batch_id, (data, label) in enumerate(data_loader):
        label = label.squeeze()
        # bag_label = label[0]
        bag_label = label

        # instance_labels = label
        # if torch.cuda.is_available():
        #    data, bag_label = data.cuda(), bag_label.cuda()
        # data, bag_label = torch.autograd.Variable(data), torch.autograd.Variable(bag_label)
        # data = data.to(device=device)
        # bag_label = bag_label.to(device=device)

        data = data.to(model.get_device_ordinal(0))
        bag_label = bag_label.to(model.get_device_ordinal(3))

        loss, attention_weights = model.compute_loss(data, bag_label)  # forward pass
        test_losses.append(float(loss))
        acc = model.compute_accuracy(data, bag_label)
        test_acc.append(float(acc))

        del data, bag_label, label

    result['val_loss'] = sum(test_losses) / len(test_losses)  # torch.stack(test_losses).mean().item()
    result['val_acc'] = sum(test_acc) / len(test_acc)
    return result, attention_weights


def _binary_accuracy(outputs, targets):
    assert targets.size() == outputs.size()
    y_prob = torch.ge(outputs, 0.5).float()
    return (targets == y_prob).sum().item() / targets.size(0)


def debug_all_models(gpu_enabled: bool = True):
    device = get_hardware_device(gpu_enabled=gpu_enabled)
    print('Selected device: ' + str(device))

    print('Checking the baseline model')
    m = BaselineMIL()
    print(m)


if __name__ == "__main__":
    print('Debugging the models, to see if everything is available.')

    debug_all_models()

    print('Finished debugging.')