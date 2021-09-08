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


def create_padded_hairy_bitstrings_data(possible_labels, n_hairysites, n_sites):

    bitstrings = create_bitstrings(possible_labels, n_hairysites)
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
    bitstrings_others = [
        bin(k)[2:].zfill(n_sites - n_hairysites)
        for k in range(2 ** (n_sites - n_hairysites))
    ]
    other_sites = [
        [
            [
                [1, 0, 0, 0] * (1 - int(bstr_site)) + [0, 1, 0, 0] * int(bstr_site)
                for bstr_site in bitstring
            ]
            for label in possible_labels
        ]
        for bitstring in bitstrings_others
    ]

    untruncated = np.array(
        [
            np.append(np.array(other_site), hairy_sites, axis=1)
            for other_site in other_sites
        ]
    )
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


def create_mpo_classifier(mpo_train, seed=None):

    n_sites = mpo_train[0].num_tensors
    # Create MPO classifier
    tensors = []
    previous_ind = rand_uuid()
    for pixel in range(n_sites):
        # Uses shape of mpo_train images
        # Quimb squeezes, thus need to specifiy size at the ends
        d, s, i, j = mpo_train[0].tensors[pixel].data.shape
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
Create Initialised Classifier (New Way)
"""


def compress_QTN(projected_q_mpo, D, orthogonalise):
    # ASSUMES q_mpo IS ALREADY PROJECTED ONTO |0> STATE FOR FIRST (n_sites - n_hairysites) SITES
    # compress procedure leaves mpo in mixed canonical form
    # center site is at left most hairest site.

    if projected_q_mpo.tensors[-1].shape[1] > 4:
        return fMPO_to_QTN(
            QTN_to_fMPO(projected_q_mpo).compress_one_site(
                D=D, orthogonalise=orthogonalise
            )
        )
    return fMPO_to_QTN(
        QTN_to_fMPO(projected_q_mpo).compress(D=D, orthogonalise=orthogonalise)
    )


def batch_adding_mpos(grouped_mpos, D, orthogonalise, compress=True):
    # Add all MPOs in list squentially. I.e. add first batch_num together,
    # then second batch_num etc. Then add result from first batch_num
    # and second batch_num etc. Until everything is added.
    batch_num = len(grouped_mpos[0])

    # "Flatten" grouped_mpos in order to add together recursively
    # Flattens as [sublist_0, sublist_1, ...]
    flattened_grouped_mpos = [item for sublist in grouped_mpos for item in sublist]
    added_mpos = flattened_grouped_mpos
    while len(added_mpos) > 1:
        results = []
        for i in range(int(len(added_mpos) / batch_num) + 1):
            sublist = added_mpos[i * batch_num : (i + 1) * batch_num]
            if len(sublist) > 0:
                if compress:
                    results.append(
                        compress_QTN(adding_sublist(sublist), D, orthogonalise)
                    )
                else:
                    results.append(adding_sublist(sublist))
        added_mpos = results
    return compress_QTN(added_mpos[0], D, orthogonalise)


def create_initialised_mpo_classifier(mpo_train, y_train, D, orthogonalise):
    possible_labels = list(set(y_train))
    n_samples = len(mpo_train)
    n_hairysites = int(np.ceil(math.log(len(possible_labels), 4)))
    n_sites = mpo_train[0].num_tensors

    # Project first (n_sites - n_hairysites) sites onto |0> state.
    # This is nessercary in order for compression to not explode.
    mpos = []
    for mpo in mpo_train:
        truncated_mpo_data = [
            site.data[:, :1, :, :] if i < (n_sites - n_hairysites) else site.data
            for i, site in enumerate(mpo.tensors)
        ]
        truncated_mpo = data_to_QTN(truncated_mpo_data)
        truncated_mpo /= (truncated_mpo.H @ truncated_mpo) ** 0.5
        mpos.append(truncated_mpo)

    mpo_train = mpos

    # Add images in equally divided batches.
    # Sort images by label/encodings
    grouped_images = [
        [mpo_train[i] for i in range(n_samples) if y_train[i] == label]
        for label in possible_labels
    ]

    # Ensure all images are included
    assert sum([len(i) for i in grouped_images]) == n_samples

    # Sorted MPOs, i.e. a list containing lists of mpos,
    # with each mpo form a different class within the list
    # Number of sublists is equal to #labels * min(#images in a class)
    # All other mpo images are thrown away. :c
    # Batch num dictated by length of sublist
    grouped_mpos = [mpo_from_each_class for mpo_from_each_class in zip(*grouped_images)]

    # Sequentially add batches of mpo images together to form initial classifier
    initialised_classifier = batch_adding_mpos(grouped_mpos, D, orthogonalise)
    return initialised_classifier


"""
Create Initialised Classifier (old way)
"""


def initialise_sequential_mpo_classifier(train_data, train_labels, D_total):
    n_samples = len(train_data)
    possible_labels = list(set(train_labels))
    batch_num = 10

    grouped_images = [
        [train_data[i] for i in range(n_samples) if train_labels[i] == label]
        for label in possible_labels
    ]

    train_data = [images for images in zip(*grouped_images)]
    train_labels = [possible_labels for _ in train_data]

    train_data = np.array([item for sublist in train_data for item in sublist])
    train_labels = np.array([item for sublist in train_labels for item in sublist])

    train_spread_mpo_product_states = mps_encoded_data(
        train_data, train_labels, D_total
    )
    mpo_classifier = sequential_mpo_classifier(
        train_labels=train_labels, mps_images=train_spread_mpo_product_states
    )

    train_spread_mpo_product_states = [
        image for label in train_spread_mpo_product_states.values() for image in label
    ]

    MPOs = train_spread_mpo_product_states
    while len(MPOs) > 1:
        MPOs = mpo_classifier.adding_batches(
            MPOs, D_total, batch_num, orthogonalise=False
        )
    classifier = MPOs[0].left_spread_canonicalise(D=D_total, orthogonalise=False)

    return fMPO_to_QTN(classifier)


"""
Train Classifier
"""


def green_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    # Loss for more than one class
    # Trains over all images in classes specified by y_train
    n_samples = len(mps_train)
    summed_overlaps = np.sum(
        [
            (mps_train[i].H @ (classifier @ q_hairy_bitstrings[y_train[i]])).norm() ** 2
            for i in range(n_samples)
        ]
    )
    return -summed_overlaps / n_samples

def squeezed_green_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    # Loss for more than one class
    # Trains over all images in classes specified by y_train

    n_samples = len(mps_train)
    summed_overlaps = np.sum(
        [
            (mps_train[i].H @ (classifier @ q_hairy_bitstrings[y_train[i]])) ** 2
            for i in range(n_samples)
        ]
    )
    return -summed_overlaps / n_samples

def squeezed_delta_green_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    # Loss for more than one class
    # Trains over all images in classes specified by y_train

    n_samples = len(mps_train)
    possible_labels = list(set(y_train))
    summed_overlaps = np.sum(
        [
            [int(y_train[i] == l) * (mps_train[i].H @ (classifier @ q_hairy_bitstrings[y_train[i]])) ** 2 for l in possible_labels]
            for i in range(n_samples)
        ]
    )
    return -summed_overlaps / n_samples


def padded_green_loss(classifier, mps_train, q_padded_hairy_bitstrings, y_train):
    n_samples = len(mps_train)
    possible_paddings = len(q_padded_hairy_bitstrings)
    summed_overlaps = np.sum(
        [
            [
                (
                    mps_train[i].H
                    @ (classifier @ q_padded_hairy_bitstrings[k][y_train[i]])
                ).norm()
                ** 2
                for i in range(n_samples)
            ]
            for k in range(possible_paddings)
        ]
    )
    return -summed_overlaps / n_samples


def stoundenmire_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    # Loss for more than one class
    # Trains over all images in classes specified by y_train
    n_samples = len(mps_train)
    possible_labels = list(set(y_train))
    overlaps = []
    # summed_overlaps = np.sum([ [((train_image.H @ (classifier @ q_hairy_bitstrings[y_train[i]])).norm() - (1 if y_train[i] == label else 0))**2 for train_image in mps_train] for i in range(n_samples)])
    # return -summed_overlaps/n_samples
    for i in range(len(mps_train)):
        for label in possible_labels:
            overlap = ((mps_train[i].H @ (classifier @ q_hairy_bitstrings[label])).norm() - int(y_train[i] == label)) ** 2
            overlaps.append(overlap)
    return 0.5 * np.sum(overlaps)

def squeezed_stoundenmire_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
    # Loss for more than one class
    # Trains over all images in classes specified by y_train
    n_samples = len(mps_train)
    possible_labels = list(set(y_train))
    overlaps = []
    # summed_overlaps = np.sum([ [((train_image.H @ (classifier @ q_hairy_bitstrings[y_train[i]])).norm() - (1 if y_train[i] == label else 0))**2 for train_image in mps_train] for i in range(n_samples)])
    # return -summed_overlaps/n_samples
    for i in range(len(mps_train)):
        for label in possible_labels:
            overlap = (abs(mps_train[i].H @ (classifier @ q_hairy_bitstrings[label])) - int(y_train[i] == label)) ** 2
            overlaps.append(overlap)
    return 0.5 * np.sum(overlaps)


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

def padded_classifier_predictions(mpo_classifier, mps_test, q_padded_hairy_bitstrings):
    # assumes mps_test is aligned with appropiate labels, y_test
    predictions = [
        np.sum(
            [
                [
                    (test_image.H @ (mpo_classifier @ b)).norm()
                    for b in q_padded_hairy_bitstrings[k]
                ]
                for k in range(len(q_padded_hairy_bitstrings))
            ],
            axis=0,
        )
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
