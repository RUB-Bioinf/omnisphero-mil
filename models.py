import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# project specific imports (repo-internal)

####
from util.utils import get_hardware_device


####

def save_checkpoint(state, is_best, save_path):
    ''' Save model and state stuff if a new best is achieved
    Used in fit function in main.
    '''
    if is_best:
        print('--> Saving new best model')
        torch.save(state, save_path)


####

def load_checkpoint(load_path, model, optim):
    ''' loads the model and its optimizer states from disk.
    '''
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optim_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optim, epoch, loss


# MODEL
#######

device_ordinals_local = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
device_ordinals_cluster = (0, 1, 2, 3, 3, 1, 2, 3, 0, 1, 2)


class BaselineMIL(nn.Module):
    def __init__(self, use_bias=True, use_max=True, device_ordinals=device_ordinals_local):
        super().__init__()
        self.linear_nodes = 512
        self.num_classes = 1  # 3
        self.device_ordinals = device_ordinals

        # self.input_dim = (3,224,224)
        # self.input_dim = (1,224,224)
        # self.input_dim = (1,350,350)
        self.input_dim = (1, 150, 150)

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
        ).to('cuda:' + str(self.device_ordinals[0]))

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
        ).to('cuda:' + str(self.device_ordinals[1]))

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
        ).to('cuda:' + str(self.device_ordinals[2]))

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
        ).to('cuda:' + str(self.device_ordinals[3]))  # bag of embeddings

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
        ).to('cuda:'+str(self.device_ordinals[4]))

    def forward(self, x):
        ''' Forward NN pass, declaring the exact interplay of model components
        '''
        x = x.squeeze(
            0)  # necessary? compresses unnecessary dimensions eg. (1,batch,channel,x,y) -> (batch,channel,x,y)
        hidden = self.feature_extractor_0(x)
        hidden = self.feature_extractor_1(hidden.to('cuda:'+str(self.device_ordinals[5])))
        hidden = self.feature_extractor_2(hidden.to('cuda:'+str(self.device_ordinals[6])))
        hidden = self.feature_extractor_3(hidden.to('cuda:'+str(self.device_ordinals[7])))  # N x linear_nodes

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

    def _get_conv_output(self, shape):
        ''' generate a single fictional input sample and do a forward pass over
        Conv layers to compute the input shape for the Flatten -> Linear layers input size
        '''
        bs = 1
        test_input = torch.autograd.Variable(torch.rand(bs, *shape)).to('cuda:'+str(self.device_ordinals[8]))
        output_features = self.feature_extractor_0(test_input)
        output_features = self.feature_extractor_1(output_features.to('cuda:'+str(self.device_ordinals[9])))
        output_features = self.feature_extractor_2(output_features.to('cuda:'+str(self.device_ordinals[10])))
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