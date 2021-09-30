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
from util.utils import line_print


@torch.no_grad()
def get_false_positive_bags(trained_model: OmniSpheroMil, train_dl: OmniSpheroDataLoader, X_raw: [np.ndarray]) -> (
        [Tensor], [Tensor], [np.ndarray]):
    """ After training access all bags falsely classified by DeepAttentionMIL as positive
    (= false positive bags).
    Also extract their respective attention weights for the bags and all instances in them.
    """
    # prediction run with trained model
    attention_weights_list = []
    false_positive_bags = []
    false_positive_bags_raw = []

    print('')
    trained_model.eval()
    for batch_id, (data, label, tile_labels, bag_index) in enumerate(train_dl):
        line_print(str(batch_id) + '/' + str(len(train_dl)))

        bag_index = int(bag_index.cpu().numpy())
        label = label.squeeze()
        bag_label = label

        # Moving to GPU if it's available
        data = data.to(trained_model.get_device_ordinal(0))
        # data = data.cpu()
        # bag_label = bag_label.cpu()
        # if torch.cuda.is_available():
        #    data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = torch.autograd.Variable(data), torch.autograd.Variable(bag_label)

        # _, predicted_label, attention_weights = trained_model.forward(data)  # forward prediction pass
        y_hat, predictions, attention, _, prediction_tiles_binarized = trained_model.forward(data)
        predictions = predictions.squeeze(dim=0)
        predictions = predictions.cpu()

        # check Ground Truth bag label and compare with model prediction ## BINARY CASE !!!
        # if predictions == bag_label:
        if float(predictions) == float(bag_label):  # and not batch_id == 0:
            # TODO resolve this debug part
            continue
        elif (predictions == 1 and bag_label == 0):  # or batch_id == 0:
            # A false positive bag has been found!
            false_positive_bags.append(data.squeeze().cpu())
            attention_weights_list.append(attention.squeeze().cpu())
            false_positive_bags_raw.append(X_raw[bag_index])

        assert len(false_positive_bags) == len(attention_weights_list)
    print('')

    log.write('Found {} false positive bags.'.format(len(false_positive_bags)))
    return false_positive_bags, attention_weights_list, false_positive_bags_raw


def determine_hard_negative_instances(false_positive_bags: [Tensor], attention_weights: [Tensor],
                                      false_positive_bags_raw: [np.ndarray], magnitude: float = 5.0) -> (
        [Tensor], [np.ndarray]):
    """ Compute the hard negative instances of each false positive with the attention weights.
    Where H_bag = {a_bag_i | a_bag_i >= std_a_bag + mean_a_bag}
    """
    hard_negative_instances = []
    hard_negative_instances_raw = []
    magnitude = float(magnitude)

    # iterate over all FP Bags individually and compute the hard negative instances
    for bag, attention_vector, bag_raw in zip(false_positive_bags, attention_weights, false_positive_bags_raw):
        assert len(bag) == len(attention_vector)
        assert len(attention_vector) == len(bag_raw)

        attention_vector = attention_vector.numpy()
        mean_attention = np.mean(attention_vector)
        std_attention = np.std(attention_vector)

        h_index = np.where(attention_vector >= (mean_attention + std_attention / magnitude))
        h_bag = bag[h_index]
        h_bag_raw = bag_raw[h_index]

        # h_bag = [bag_tile for bag_tile, attention_weight,bag_tile_raw in zip(bag, attention_vector,bag_raw) ifattention_weight >= (mean_attention + std_attention / magnitude)]

        if h_bag.shape[0] > 0:  # check if not empty
            hard_negative_instances.append(h_bag)
            hard_negative_instances_raw.append(h_bag_raw)

    # unroll nested list to contain all hard negative instances in a single structure
    hard_negative_instances = [instance for sublist in hard_negative_instances for instance in sublist]
    hard_negative_instances_raw = [np.copy(instance) for sublist in hard_negative_instances_raw for instance in sublist]
    assert len(hard_negative_instances) == len(hard_negative_instances_raw)

    return hard_negative_instances, hard_negative_instances_raw


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
def new_bag_generation(hard_negative_instances: [Tensor], training_ds: OmniSpheroDataLoader,
                       hard_negative_instances_raw: [np.ndarray], n_clusters: int = 10, random_seed: int = 1337) -> (
[[np.ndarray]], [[np.ndarray]]):
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

    log.write('Mining VGG16 instances: ' + str(len(hard_negative_instances)))
    print('')  # Printing an empty line, just to console, not to log.
    all_feature_vectors = []
    c = 0
    for instance in hard_negative_instances:
        c = c + 1
        line_print('Mining VGG16 instance: ' + str(c) + '/' + str(len(hard_negative_instances)))

        instance = instance.unsqueeze(dim=0).cpu()
        instance_input = torch.cat((instance, instance, instance), dim=0)  # achieve faux-RGB
        # TODO Did I RGB right?

        # feature_vector = feature_extractor(instance_input.unsqueeze(dim=0))  # produces [1,4096] torch tensor
        feature_vector = feature_extractor(instance)
        feature_vector = feature_vector.squeeze()
        all_feature_vectors.append(feature_vector)

    print('')  # Printing an empty line, just to console, not to log.
    log.write('Finished mining VGG16 instances: ' + str(len(hard_negative_instances)))

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
    new_bags_raw = []
    new_bag_names = []
    np_instances = np.asarray([np.asarray(i.cpu()) for i in hard_negative_instances])

    for i in range(n_clusters):
        new_bag = []
        new_bag_raw = []
        new_bag_size = compute_bag_size(training_ds)
        print('')  # Printing an empty line, just to console, not to log.

        for sample in range(new_bag_size):
            line_print('Repacking sample ' + str(sample + 1) + '/' + str(new_bag_size) + ' from cluster ' + str(
                i + 1) + '/' + str(n_clusters))

            chosen_cluster = np.random.choice(cluster_labels, size=1, replace=False, p=None)  # get a random cluster
            chosen_indexes = np.where(cluster_labels == chosen_cluster)[0]
            chosen_instances = np.array(np_instances)[
                chosen_indexes]  # get all hard negative instances assigned to this kMeans cluster
            chosen_instances_raw = np.array(hard_negative_instances_raw)[chosen_indexes]

            instance_index = np.random.choice(chosen_instances.shape[0], size=1, replace=False, p=None)
            new_instance = chosen_instances[instance_index, ...]
            new_instance_raw = chosen_instances_raw[instance_index, ...]
            # from this cluster select a random instance

            new_bag.append(new_instance[0])
            new_bag_raw.append(new_instance_raw[0])

        new_bags.append(new_bag)
        new_bags_raw.append(new_bag_raw)
        new_bag_names.append('hnm-' + str(i))

    assert len(new_bags) == len(new_bags_raw)
    assert len(new_bags) == len(new_bag_names)
    return new_bags, new_bags_raw, new_bag_names


def add_back_to_dataset(training_ds: [([np.ndarray])], new_bags: [[np.ndarray]], X_raw: [np.ndarray], bag_names: [str],
                        new_bag_names: [str], new_bags_raw: [[np.ndarray]]) -> ([([np.ndarray])], [np.ndarray], [str]):
    """ Add the constructed newly generated Hard Negative Bags to the original training dataset.
    """
    for i in range(len(new_bags)):
        # numpy_bag = [torch_tensor_cuda.unsqueeze(dim=0).cpu().numpy() for torch_tensor_cuda in bag]
        bag = new_bags[i]
        bag_raw = new_bags_raw[i]
        new_bag_name = str(new_bag_names[i])
        numpy_bag = np.asarray(bag)

        new_label = 0  # [0,0,1] # (or 0) making it 'normal' or 'negative' in multiclass or binary setting
        new_label = np.tile(new_label, (len(numpy_bag), 1))
        new_bag_data = np.asarray(numpy_bag, dtype='float32')
        new_bag_label = np.asarray(0, dtype='float32')
        new_tile_labels = np.asarray(new_label, dtype='float32').squeeze()
        new_bag_raw = np.asarray(bag_raw, dtype='uint8')

        if len(new_bag_data) > 3:
            current_bag_index = len(X_raw)

            index = np.asarray(current_bag_index, dtype='float32')
            new_ds = (new_bag_data, new_bag_label, new_tile_labels, index)

            # print('HNM generated new data:')
            # print(new_bag_data.shape)
            # print('HNM generated new label:')
            # print(new_tile_label.shape)
            training_ds.append(new_ds)
            X_raw.append(new_bag_raw)
            bag_names.append(new_bag_name)

    return training_ds, X_raw, bag_names


if __name__ == '__main__':
    print('This file contains OmniSphero hard-negative mining and provides corresponding util functions.')
