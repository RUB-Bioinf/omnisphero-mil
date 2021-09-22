# IMPORTS
#########

import math
import numpy as np
import torch
import torchvision.models as model_zoo
from sklearn.cluster import KMeans
from torch import Tensor
import random

from models import OmniSpheroMil
from util import log
from util.omnisphero_data_loader import OmniSpheroDataLoader


# FUNCTIONS
###########

@torch.no_grad()
def get_false_positive_bags(trained_model: OmniSpheroMil, train_dl: OmniSpheroDataLoader) -> ([Tensor], [Tensor]):
    """ After training access all bags falsely classified by DeepAttentionMIL as positive
    (= false positive bags).
    Also extract their respective attention weights for the bags and all instances in them.
    """
    # prediction run with trained model
    attention_weights_list = []
    false_positive_bags = []

    trained_model.eval()
    for batch_id, (data, label, tile_labels, bag_index) in enumerate(train_dl):
        print(str(batch_id) + '/' + str(len(train_dl)))
        # label = label.squeeze()
        # bag_label = label[0]

        label = label.squeeze()
        # bag_label = label[0] //TODO this causes error
        bag_label = label

        # Moving to GPU if it's available
        data = data.cpu()
        bag_label = bag_label.cpu()
        # if torch.cuda.is_available():
        #    data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = torch.autograd.Variable(data), torch.autograd.Variable(bag_label)

        # _, predicted_label, attention_weights = trained_model.forward(data)  # forward prediction pass
        y_hat, predictions, attention, _, prediction_tiles_binarized = trained_model.forward(data)
        predictions = predictions.squeeze(dim=0)

        # check Ground Truth bag label and compare with model prediction ## BINARY CASE !!!
        # if predictions == bag_label:
        if float(predictions) == float(bag_label) and random.choice(
                [True, False, False, False]):  # TODO resolve this debug part
            continue
        elif predictions == 1 and bag_label == 0:
            # A false positive bag has been found!
            false_positive_bags.append(data.squeeze().cpu())
            attention_weights_list.append(attention.squeeze().cpu())

        assert len(false_positive_bags) == len(attention_weights_list)

    log.write('Found {} false positive bags.'.format(len(false_positive_bags)))
    return false_positive_bags, attention_weights_list


def determine_hard_negative_instances(false_positive_bags: [Tensor], attention_weights: [Tensor],
                                      magnitude: float = 5.0) -> [Tensor]:
    """ Compute the hard negative instances of each false positive with the attention weights.
    Where H_bag = {a_bag_i | a_bag_i >= std_a_bag + mean_a_bag}
    """
    hard_negative_instances = []
    magnitude = float(magnitude)

    # iterate over all FP Bags individually and compute the hard negative instances
    for bag, attention_vector in zip(false_positive_bags, attention_weights):
        assert len(bag) == len(attention_vector)
        attention_vector = attention_vector.numpy()
        mean_attention = np.mean(attention_vector)
        std_attention = np.std(attention_vector)

        h_bag = [bag_tile for bag_tile, attention_weight in zip(bag, attention_vector) if
                 attention_weight >= (mean_attention + std_attention / magnitude)]

        if h_bag:  # check if not empty
            hard_negative_instances.append(h_bag)

    # unroll nested list to contain all hard negative instances in a single structure
    hard_negative_instances = [instance for sublist in hard_negative_instances for instance in sublist]

    return hard_negative_instances


def compute_bag_size(training_ds: [np.ndarray]) -> int:
    """ To conform to training bag size a newly generated (hard negative) bag
    is empirically generated with a Gaussian random size of sigma and mu where sigma is
    std and mu is mean of all training bag sizes.
    """
    tile_amount = []
    for data in training_ds:
        tile_amount.append(data[0].shape[0])
    mu = np.mean(tile_amount)
    sigma = np.std(tile_amount)

    hard_negative_bag_size = np.random.normal(mu, sigma, size=1)
    chosen_size = math.ceil(hard_negative_bag_size)
    log.write('Calculating negative bag size. Samples: ' + str(sum(tile_amount)) + ' Mean bag size: ' + str(
        mu) + '. Std: ' + str(sigma) + '. Size: ' + str(chosen_size))
    return chosen_size


@torch.no_grad()
def new_bag_generation(hard_negative_instances: [Tensor], training_ds, n_clusters: int = 10, random_seed: int = 1337):
    """ Use a pretrained CNN w/o last layer to extract feature vectors from
    the determined hard negative instances.
    These feature vectors are then clustered with k-Means to obtain feature clusters.
    From the feature clusters one can employ weighted instance sampling to generate
    new Hard Negative Bags.
    """
    # Produce feature vectors
    #########################
    # load pretrained model from pytorch zoo
    feature_extractor = model_zoo.vgg16(pretrained=True)

    feature_extractor.classifier = feature_extractor.classifier[:-1]  # drop last layer (classification)
    # FeatureExtractor.features[0] = torch.nn.Conv2d(1,64,3,1,1) # modify input to grayscale image either the above
    # new Conv2d layer for 1 channel images <OR> concatenate the same grayscale image 3 times to achieve faux-RGB (
    # see below in loop)

    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False

    # assert model output
    assert feature_extractor(torch.rand((1, 3, 224, 224))).size() == torch.Size([1, 4096])
    # TODO why?
    # model done

    all_feature_vectors = []
    c = 0
    for instance in hard_negative_instances:
        c = c + 1
        print('Mining VGG16 instance: ' + str(c) + '/' + str(len(hard_negative_instances)))

        instance = instance.unsqueeze(dim=0).cpu()
        instance_input = torch.cat((instance, instance, instance), dim=0)  # achieve faux-RGB
        # TODO Did I RGB right?

        # feature_vector = feature_extractor(instance_input.unsqueeze(dim=0))  # produces [1,4096] torch tensor
        feature_vector = feature_extractor(instance)
        feature_vector = feature_vector.squeeze()
        all_feature_vectors.append(feature_vector)

    # Feature Clustering with k-Means
    all_feature_vectors = np.vstack(all_feature_vectors)  # (n, 4096) array
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=random_seed).fit_predict(
        all_feature_vectors)  # assigns cluster label for each sample == for each hard negative instance feature vector
    assert len(cluster_labels) == len(hard_negative_instances)

    # Weighted Instance sampling
    ############################
    # randomly select hard negative instances from each feature cluster and puts it into a new bag
    # number of new_bags = n_clusters
    new_bags = []
    np_instances = np.asarray([np.asarray(i) for i in hard_negative_instances.cpu()])

    for i in range(n_clusters):
        new_bag = []
        new_bag_size = compute_bag_size(training_ds)

        for sample in range(new_bag_size):
            chosen_cluster = np.random.choice(cluster_labels, size=1, replace=False, p=None)  # get a random cluster
            chosen_instances = np.array(np_instances)[np.where(cluster_labels == chosen_cluster)[
                0]]  # get all hard negative instances assigned to this kMeans cluster
            new_instance = chosen_instances[np.random.choice(chosen_instances.shape[0], size=1, replace=False,
                                                             p=None), ...]  # from this cluster select a random instance
            new_bag.append(new_instance[0])
        new_bags.append(new_bag)

    return new_bags


def add_back_to_dataset(training_ds: [([np.ndarray])], new_bags: [np.ndarray]) -> [([np.ndarray])]:
    """ Add the constructed newly generated Hard Negative Bags to the original training dataset.
    """
    current_bag_index = int(max([ds[3] for ds in training_ds]))

    for bag in new_bags:
        # numpy_bag = [torch_tensor_cuda.unsqueeze(dim=0).cpu().numpy() for torch_tensor_cuda in bag]
        numpy_bag = np.asarray(bag)

        new_label = 0  # [0,0,1] # (or 0) making it 'normal' or 'negative' in multiclass or binary setting
        new_label = np.tile(new_label, (len(numpy_bag), 1))
        new_bag_data = np.asarray(numpy_bag, dtype='float32')
        new_bag_label = np.asarray(0, dtype='float32')
        new_tile_labels = np.asarray(new_label, dtype='float32').squeeze()

        if len(new_bag_data) > 3:
            current_bag_index = current_bag_index + 1

            index = np.asarray(current_bag_index, dtype='float32')
            new_ds = (new_bag_data, new_bag_label, new_tile_labels, index)

            # print('HNM generated new data:')
            # print(new_bag_data.shape)
            # print('HNM generated new label:')
            # print(new_tile_label.shape)
            training_ds.append(new_ds)

    return training_ds


if __name__ == '__main__':
    print('This file contains OmniSphero hard-negative mining and provides corresponding util functions.')
