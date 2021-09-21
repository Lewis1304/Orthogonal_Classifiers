"""
Import dependentcies
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import quimb
import numpy as np
import math
import matplotlib.pyplot as plt
import idx2numpy as idn
import autograd.numpy as anp

import quimb.tensor as qtn
from quimb.tensor.tensor_core import rand_uuid
from quimb.tensor.optimize import TNOptimizer

from oset import oset
from xmps.fMPS import fMPS
from fMPO_reduced import fMPO
from tools import *

"""
Encode Bitstrings
"""

def create_hairy_bitstrings_data(
    possible_labels, n_hairysites, n_sites, one_site=False
):

    bitstrings = create_bitstrings(possible_labels, n_hairysites)

    if one_site:
        num_qubits = int(np.log2(len(possible_labels))) + 1
        hairy_sites = np.expand_dims(
            [i for i in np.eye(2 ** num_qubits)][: len(possible_labels)], 1
        )

        other_sites = np.array(
            [
                [np.eye(2 ** num_qubits)[0] for pixel in range(n_sites - 1)]
                for _ in possible_labels
            ]
        )

    else:

        hairy_sites = np.array(
            [
                [
                    [1, 0, 0, 0]
                    * (1 - int(bitstring[i : i + 2][0]))
                    * (1 - int(bitstring[i : i + 2][1]))
                    + [0, 1, 0, 0]
                    * (1 - int(bitstring[i : i + 2][0]))
                    * (int(bitstring[i : i + 2][1]))
                    + [0, 0, 1, 0]
                    * (int(bitstring[i : i + 2][0]))
                    * (1 - int(bitstring[i : i + 2][1]))
                    + [0, 0, 0, 1]
                    * (int(bitstring[i : i + 2][0]))
                    * (int(bitstring[i : i + 2][1]))
                    for i in range(0, len(bitstring), 2)
                ]
                for bitstring in bitstrings
            ]
        )

        other_sites = np.array(
            [
                [[1, 0, 0, 0] for pixel in range(n_sites - n_hairysites)]
                for _ in possible_labels
            ]
        )
    # .shape = #classes, #sites, dim(s)
    untruncated = np.append(other_sites, hairy_sites, axis=1)

    return untruncated


"""
MPS Encoding
"""

def mps_encoding(images, D=2):
    mps_images = [fMPS_to_QTN(image_to_mps(image, D)) for image in images]
    return mps_images


"""
MPO Encoding
"""

def class_encode_mps_to_mpo(mps, label, q_hairy_bitstrings, n_sites):
    mps_data = [tensor.data for tensor in mps.tensors]
    # class_encoded_mps.shape = #pixels, dim(d), d(s), i, j
    class_encoded_mps = [
        np.array(
            [mps_data[site] * i for i in q_hairy_bitstrings[label].tensors[site].data]
        ).transpose(1, 0, 2, 3)
        for site in range(n_sites)
    ]
    return class_encoded_mps


def mpo_encoding(mps_train, y_train, q_hairy_bitstrings):
    n_samples = len(mps_train)
    n_sites = mps_train[0].num_tensors
    mpo_train = [
        data_to_QTN(
            class_encode_mps_to_mpo(
                mps_train[i], y_train[i], q_hairy_bitstrings, n_sites
            )
        )
        for i in range(n_samples)
    ]
    return mpo_train


"""
Create Random Classifier
"""


def create_mpo_classifier(mps_train, q_hairy_bitstrings, seed=None, full_sized=False):

    n_sites = mpo_train[0].num_tensors
    # Create MPO classifier
    tensors = []
    previous_ind = rand_uuid()
    D_max = max([site.shape[-1] for site in mps_train[0].tensors])

    for pixel in range(n_sites):
        # Uses shape of mpo_train images
        # Quimb squeezes, thus need to specifiy size at the ends
        d, i, j = mps_train[0].tensors[pixel].data.shape
        s = q_hairy_bitstrings[0].tensors[pixel].data.shape[0]
        next_ind = rand_uuid()

        if full_sized:

            if pixel == 0:
                site_tensor = qtn.Tensor(
                    quimb.gen.rand.randn([d, s, 1, D_max], seed=seed),
                    inds=(f"k{pixel}", f"s{pixel}", previous_ind, next_ind),
                    tags=[f"{pixel}"],
                )
            elif pixel == (n_sites - 1):
                site_tensor = qtn.Tensor(
                    quimb.gen.rand.randn([d, s, D_max, 1], seed=seed),
                    inds=(f"k{pixel}", f"s{pixel}", previous_ind, next_ind),
                    tags=[f"{pixel}"],
                )
            else:
                site_tensor = qtn.Tensor(
                    quimb.gen.rand.randn([d, s, D_max, D_max], seed=seed),
                    inds=(f"k{pixel}", f"s{pixel}", previous_ind, next_ind),
                    tags=[f"{pixel}"],
                )

        else:
            site_tensor = qtn.Tensor(
                quimb.gen.rand.randn([d, s, i, j], seed=seed),
                inds=(f"k{pixel}", f"s{pixel}", previous_ind, next_ind),
                tags=[f"{pixel}"],
            )
        tensors.append(site_tensor)
        previous_ind = next_ind

    mpo_classifier = qtn.TensorNetwork(tensors)
    mpo_classifier /= (mpo_classifier.H @ mpo_classifier) ** 0.5
    return mpo_classifier


"""
Train Classifier
"""


def green_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    overlaps = [
        anp.real(mps_train[i].H @ (classifier @ q_hairy_bitstrings[y_train[i]])) ** 2
        for i in range(len(mps_train))
    ]
    return -np.sum(overlaps) / len(mps_train)


def abs_green_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    overlaps = [
        abs(mps_train[i].H @ (classifier @ q_hairy_bitstrings[y_train[i]])) ** 2
        for i in range(len(mps_train))
    ]
    return -np.sum(overlaps) / len(mps_train)


def mse_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    overlaps = [
        (anp.real(mps_train[i].H @ (classifier @ q_hairy_bitstrings[y_train[i]])) - 1)
        ** 2
        for i in range(len(mps_train))
    ]
    return np.sum(overlaps) / len(mps_train)


def abs_mse_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    overlaps = [
        (abs(mps_train[i].H @ (classifier @ q_hairy_bitstrings[y_train[i]])) - 1) ** 2
        for i in range(len(mps_train))
    ]
    return np.sum(overlaps) / len(mps_train)


def cross_entropy_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    overlaps = [
        anp.log(abs(mps_train[i].H @ (classifier @ q_hairy_bitstrings[y_train[i]])))
        for i in range(len(mps_train))
    ]
    return -np.sum(overlaps) / len(mps_train)


def stoudenmire_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    possible_labels = list(set(y_train))
    overlaps = [
        [
            (
                anp.real(mps_train[i].H @ (classifier @ q_hairy_bitstrings[y_train[i]]))
                - int(y_train[i] == label)
            )
            ** 2
            for label in possible_labels
        ]
        for i in range(len(mps_train))
    ]
    return np.sum(overlaps) / len(mps_train)


def abs_stoudenmire_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    possible_labels = list(set(y_train))
    overlaps = [
        [
            (
                abs(mps_train[i].H @ (classifier @ q_hairy_bitstrings[y_train[i]]))
                - int(y_train[i] == label)
            )
            ** 2
            for label in possible_labels
        ]
        for i in range(len(mps_train))
    ]
    return np.sum(overlaps) / len(mps_train)


def normalize_tn(tn):
    return tn / (tn.H @ tn) ** 0.5


def orthogonalise_and_normalize(tn):
    tn = compress_QTN(tn, D=None, orthogonalise=True)
    return tn / (tn.H @ tn) ** 0.5


"""
Evaluate classifier
"""


def classifier_predictions(mpo_classifier, mps_test, q_hairy_bitstrings):
    # assumes mps_test is aligned with appropiate labels, y_test
    predictions = [
        [(test_image.H @ (mpo_classifier @ b)).norm() for b in q_hairy_bitstrings]
        for test_image in mps_test
    ]
    return predictions


def squeezed_classifier_predictions(mpo_classifier, mps_test, q_hairy_bitstrings):
    # assumes mps_test is aligned with appropiate labels, y_test
    predictions = [
        [abs(test_image.H @ (mpo_classifier @ b)) for b in q_hairy_bitstrings]
        for test_image in mps_test
    ]
    return predictions


def evaluate_classifier_accuracy(predictions, y_test):
    argmax_predictions = [
        np.argmax(image_prediction) for image_prediction in predictions
    ]
    results = np.mean([int(i == j) for i, j in zip(y_test, argmax_predictions)])
    return results


def evaluate_classifier_top_k_accuracy(predictions, y_test, k):
    top_k_predicitions = [
        np.argpartition(image_prediction, -k)[-k:] for image_prediction in predictions
    ]
    results = np.mean([int(i in j) for i, j in zip(y_test, top_k_predicitions)])
    return results


def evaluate_prediction_variance(predictions):
    prediction_variance = [np.var(image_prediction) for image_prediction in predictions]
    return np.mean(prediction_variance)


if __name__ == "__main__":
    pass
