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
from tqdm import tqdm
from collections import Counter

from oset import oset
from xmps.fMPS import fMPS
from fMPO import fMPO
from tools import *

"""
Encode Bitstrings
"""


def create_hairy_bitstrings_data(possible_labels, n_sites):

    bitstrings = create_bitstrings(possible_labels)

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
    untruncated = np.append(other_sites, hairy_sites, axis=1)
    return untruncated


def create_padded_bitstrings_data(possible_labels,uclassifier):
    #Only for onesite at the moment.
    if not (uclassifier.tensors[-2].shape[1] < uclassifier.tensors[-1].shape[1]):
        raise Exception('Only works for one site classifiers at the moment')

    max_s = max([site.shape[1] for site in uclassifier.tensors])
    max_s_others = max([site.shape[1] for site in uclassifier.tensors[:-1]])
    n_paddings = np.sum([site.shape[1] for site in uclassifier.tensors[:-1]]) - len(uclassifier.tensors[:-1])

    bitstrings_others = [bin(k)[2:].zfill(len(uclassifier.tensors[:-1])) for k in range(2**n_paddings)]

    #other_sites has shape (10,16,9,64) = (labels,padding configrations, sites, max_S from Unitaryfying ())
    other_sites = np.array([[ [([1,0] + [0]*(max_s - 2)) * (1 - int(bstr_site)) + ([0,1]+ [0]*(max_s - 2))  * int(bstr_site) for bstr_site in bitstring] for bitstring in bitstrings_others] for _ in possible_labels])


    num_qubits = int(np.log2(len(possible_labels))) + 1
    #hairy_site has shape (10, 48, 1,64) = (padding configrations, sites, max_S from Unitaryfying)

    #from scipy.linalg import null_space

    #test = null_space(hairy_sites,10)
    #assert()
    #hairy_site = np.array([[list(i) + list(j) for j in np.eye(uclassifier.tensors[-1].shape[1] - 2 ** num_qubits)] for i in np.eye(2 ** num_qubits)][: len(possible_labels)])
    hairy_site = np.pad([i for i in np.eye(2 ** num_qubits)][:len(possible_labels)], ((0,0), (0,uclassifier.tensors[-1].shape[1] - 2**num_qubits)))


    hairy_site = np.expand_dims(np.expand_dims(hairy_site,1),1)
    #print(other_sites.shape)
    #print(hairy_site.shape)
    untruncated = []
    for label1, label2 in zip(other_sites, hairy_site):
        padded_configs = []
        for padded_other in label1:
            for padded_hairy in label2:
                padded_configs.append(np.append(padded_other, padded_hairy, axis = 0))
        untruncated.append(padded_configs)

    return np.array(untruncated).transpose(1,0,2,3)



"""
MPS Encoding
"""


def mps_encoding(images, D=2):
    mps_images = [fMPS_to_QTN(image_to_mps(image, D)) for image in images]
    return mps_images


"""
Create Random Classifier
"""


def create_mpo_classifier(mps_train, q_hairy_bitstrings, seed=None, full_sized=False):

    n_sites = mps_train[0].num_tensors
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


def create_mpo_classifier_from_initialised_classifier(initialised_classifier, seed=420):

    # Create MPO classifier
    tensors = []
    previous_ind = rand_uuid()
    n_sites = initialised_classifier.num_tensors

    for pixel in range(n_sites):
        # Uses shape of mpo_train images
        # Quimb squeezes, thus need to specifiy size at the ends
        d, s, i, j = initialised_classifier.tensors[pixel].data.shape
        next_ind = rand_uuid()

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
        anp.real(
            mps_train[i].squeeze().H
            @ (classifier @ q_hairy_bitstrings[y_train[i]].squeeze())
        )
        ** 2
        for i in range(len(mps_train))
    ]
    return -np.sum(overlaps) / len(mps_train)


def abs_green_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    overlaps = [
        abs(
            mps_train[i].squeeze().H
            @ (classifier @ q_hairy_bitstrings[y_train[i]].squeeze())
        )
        ** 2
        for i in range(len(mps_train))
    ]
    return -np.sum(overlaps) / len(mps_train)


def mse_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    overlaps = [
        (
            anp.real(
                mps_train[i].squeeze().H
                @ (classifier @ q_hairy_bitstrings[y_train[i]].squeeze())
            )
            - 1
        )
        ** 2
        for i in range(len(mps_train))
    ]
    return np.sum(overlaps) / len(mps_train)


def abs_mse_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    overlaps = [
        (
            abs(
                mps_train[i].squeeze().H
                @ (classifier @ q_hairy_bitstrings[y_train[i]].squeeze())
            )
            - 1
        )
        ** 2
        for i in range(len(mps_train))
    ]
    return np.sum(overlaps) / len(mps_train)


def cross_entropy_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    overlaps = [
        anp.log(
            abs(
                mps_train[i].squeeze().H
                @ (classifier @ q_hairy_bitstrings[y_train[i]].squeeze())
            )
        )
        for i in range(len(mps_train))
    ]
    return -np.sum(overlaps) / len(mps_train)


def stoudenmire_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    possible_labels = list(set(y_train))
    overlaps = [
        [
            (
                anp.real(
                    mps_train[i].squeeze().H
                    @ (classifier @ q_hairy_bitstrings[label].squeeze())
                )
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
                abs(
                    mps_train[i].squeeze().H
                    @ (classifier @ q_hairy_bitstrings[label].squeeze())
                )
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


"""
Evaluate classifier
"""


def classifier_predictions(mpo_classifier, mps_test, q_hairy_bitstrings):
    # assumes mps_test is aligned with appropiate labels, y_test
    predictions = [
        [
            abs(test_image.squeeze().H @ (mpo_classifier @ b.squeeze()))
            for b in q_hairy_bitstrings
        ]
        for test_image in tqdm(mps_test)
    ]
    return predictions


def ensemble_predictions(ensemble, mps_test, q_hairy_bitstrings):
    # assumes mps_test is aligned with appropiate labels, y_test
    predictions = [[
        [
            abs(test_image.squeeze().H @ (classifier @ b.squeeze()))
            for b in q_hairy_bitstrings
        ]
        for test_image in mps_test
    ]
    for classifier in tqdm(ensemble)]
    normalised_predictions = [[j / np.sum(j) for j in i] for i in predictions]
    return normalised_predictions

def padded_classifier_predictions(mpo_classifier, mps_test, padded_q_hairy_bitstrings):
    # assumes mps_test is aligned with appropiate labels, y_test
    predictions = [np.sum([[abs(test_image.squeeze().H @ (mpo_classifier @ b.squeeze())) for b in paddings] for paddings in padded_q_hairy_bitstrings],axis = 0) for test_image in tqdm(mps_test)]
    return predictions


def evaluate_classifier_top_k_accuracy(predictions, y_test, k):
    top_k_predictions = [
        np.argpartition(image_prediction, -k)[-k:] for image_prediction in predictions
    ]
    results = np.mean([int(i in j) for i, j in zip(y_test, top_k_predictions)])
    return results


def evaluate_soft_ensemble_top_k_accuracy(e_predictions, y_test, k):
    top_k_ensemble_predictions = np.sum(e_predictions, axis = 0)
    top_k_predictions = [
        np.argpartition(image_prediction, -k)[-k:] for image_prediction in top_k_ensemble_predictions
    ]
    results = np.mean([int(i in j) for i, j in zip(y_test, top_k_predictions)])
    return results

def evaluate_hard_ensemble_top_k_accuracy(e_predictions, y_test, k):
    top_k_ensemble_predictions = np.array([[
        np.argpartition(image_prediction, -k)[-k:] for image_prediction in top_k_predictions
    ] for top_k_predictions in e_predictions]).transpose(1,0,2).reshape(len(y_test), -1)

    top_k_ensemble_predictions = [[t[0] for t in Counter(i).most_common(k)] for i in top_k_ensemble_predictions]
    results = np.mean([int(i in j) for i, j in zip(y_test, top_k_ensemble_predictions)])
    return results

if __name__ == "__main__":
    pass
