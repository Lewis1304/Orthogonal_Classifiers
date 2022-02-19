import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import quimb
import numpy as np
from oset import oset
from xmps.fMPS import fMPS
from quimb.tensor.tensor_core import rand_uuid
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer
import math
import pytest
import sys

sys.path.append("../")
from tools import *
from variational_mpo_classifiers import *
from deterministic_mpo_classifier import mpo_encoding, prepare_batched_classifier, unitary_qtn

from xmps.ncon import ncon as nc
from functools import reduce

def ncon(*args, **kwargs):
    return nc(
        *args, check_indices=False, **kwargs
    )  # make default ncon not check indices


n_train = 100
n_test = 100
D_total = 10

x_train, y_train, x_test, y_test = load_data(n_train, n_test)

possible_labels = list(set(y_train))
n_sites = int(np.ceil(math.log(x_train.shape[-1], 2)))
n_pixels = len(x_train[0])

hairy_bitstrings_data_untruncated_data = create_hairy_bitstrings_data(
    possible_labels, n_sites
)

one_site_bitstrings_data_untruncated_data = create_hairy_bitstrings_data(
    possible_labels, n_sites
)


quimb_hairy_bitstrings = bitstring_data_to_QTN(
    hairy_bitstrings_data_untruncated_data, n_sites, truncated=False
)
truncated_quimb_hairy_bitstrings = bitstring_data_to_QTN(
    hairy_bitstrings_data_untruncated_data, n_sites, truncated=True
)
truncated_one_site_quimb_hairy_bitstrings = bitstring_data_to_QTN(
    one_site_bitstrings_data_untruncated_data, n_sites, truncated=True
)


fmps_images = [image_to_mps(image, D_total) for image in x_train]

mps_train = mps_encoding(x_train, D_total)
mpo_train = mpo_encoding(mps_train, y_train, quimb_hairy_bitstrings)

mpo_classifier = create_mpo_classifier(mps_train, quimb_hairy_bitstrings, seed=420)
truncated_mpo_classifier = create_mpo_classifier(
    mps_train, truncated_quimb_hairy_bitstrings, seed=420
)
one_site_truncated_mpo_classifier = create_mpo_classifier(
    mps_train, truncated_one_site_quimb_hairy_bitstrings, seed=420
)

predictions = np.array(
    classifier_predictions(mpo_classifier.squeeze(), mps_train, quimb_hairy_bitstrings)
)


def test_create_hairy_bitstrings_data():

    # Test one_site
    dim_s = 16
    assert one_site_bitstrings_data_untruncated_data.shape == (
        len(possible_labels),
        n_sites,
        dim_s,
    )

    # Test whether first (n_sites - n_hairysites) sites are in the correct state. i.e. |00>
    # Check for all classes
    for label in possible_labels:
        for site in hairy_bitstrings_data_untruncated_data[label][
            : (n_sites - 1)
        ]:
            assert np.array_equal(site, [1] + [0]*(dim_s - 1))

    # Test one_site
    for label in possible_labels:
        for site in one_site_bitstrings_data_untruncated_data[label][: (n_sites - 1)]:
            assert np.array_equal(site, np.eye(16)[0])


def test_create_padded_bitstrings_data():
    mps_train = mps_encoding(x_train, 32)

    fmpo_classifier = prepare_batched_classifier(
            mps_train, y_train, 32, 10
        ).compress_one_site(D = None, orthogonalise = True)
    qtn_classifier = data_to_QTN(fmpo_classifier.data)

    uclassifier = unitary_qtn(qtn_classifier)

    bitstrings_data = create_padded_bitstrings_data(possible_labels, uclassifier)

    #test correct shape: #padding combinations, #classes, #sites, max_s
    ss = [site.shape[1] for site in uclassifier.tensors]
    max_s = max(ss)
    number_of_paddings = reduce(lambda x, y: x*y, ss[:-1])
    assert(bitstrings_data.shape == (number_of_paddings, len(possible_labels), uclassifier.num_tensors, max_s))

    bitstrings_others = [bin(k)[2:].zfill(len(uclassifier.tensors[:-1])) for k in range(number_of_paddings)]
    #test paddings are encoded correctly
    for padding1, padding2 in zip(bitstrings_data, bitstrings_others):
        for label in padding1:
            for site1, site2, site3 in zip(label, uclassifier.tensors[:-1], padding2):
                if site2.shape[1] > 1:
                    basis = ((1 - int(site3)) * ([1,0] + [0] * (max_s - 2))) + (int(site3) * ([0,1] + [0] * (max_s - 2)))
                    assert(np.array_equal(site1, basis))


def test_bitstring_to_product_state_data():

    product_states = bitstring_to_product_state_data(
        hairy_bitstrings_data_untruncated_data
    )

    # Check whether all labels are converted to product_states
    assert len(product_states) == len(possible_labels)

    # Check correct number of sites:
    for label in possible_labels:
        assert len(product_states[label]) == n_sites

    # Check shape of each product state is correct
    # site.shape = s, i, j
    for label in possible_labels:
        for site in product_states[label]:
            assert site.shape == (
                16,
                1,
                1,
            )

    # Check data isn't altered when converted to product state
    for label in possible_labels:
        for a, b in zip(
            hairy_bitstrings_data_untruncated_data[label], product_states[label]
        ):
            assert np.array_equal(a, b.squeeze())

    # Check the same for untruncated one site
    one_site_product_states = bitstring_to_product_state_data(
        one_site_bitstrings_data_untruncated_data
    )
    for label in possible_labels:
        for a, b in zip(
            one_site_bitstrings_data_untruncated_data[label],
            one_site_product_states[label],
        ):
            assert np.array_equal(a, b.squeeze())


def test_create_mpo_classifier():

    # Untruncated
    # Multiple Sites
    # Check shape is correct
    for site_a, site_b in zip(mpo_classifier.tensors, mpo_train[0].tensors):
        assert site_a.shape == site_b.shape

    # Check classifier is normalised
    assert np.isclose(mpo_classifier.H @ mpo_classifier, 1)

    # Check full sized shape is correct
    full_sized_mpo_classifier = create_mpo_classifier(
        mps_train, quimb_hairy_bitstrings, seed=420, full_sized=True
    )
    max_D = D_total

    for k, (site_a, site_b) in enumerate(
        zip(mpo_classifier.tensors, full_sized_mpo_classifier.tensors)
    ):
        d1, s1, i1, j1 = site_a.shape
        d2, s2, i2, j2 = site_b.shape

        assert d1 == d2
        assert s1 == s2

        if k == 0:
            assert i2 == 1
            assert j2 == max_D

        elif k == (mpo_classifier.num_tensors - 1):
            assert i2 == max_D
            assert j2 == 1

        else:
            assert i2 == max_D
            assert j2 == max_D

    # Truncated, multiple sites
    # Check shape is correct
    truncated_mpo_train = mpo_encoding(
        mps_train, y_train, truncated_quimb_hairy_bitstrings
    )

    for site_a, site_b in zip(
        truncated_mpo_classifier.tensors, truncated_mpo_train[0].tensors
    ):
        assert site_a.shape == site_b.shape

    # Check classifier is normalised
    assert np.isclose(truncated_mpo_classifier.H @ truncated_mpo_classifier, 1)

    # Check full sized shape is correct
    truncated_full_sized_mpo_classifier = create_mpo_classifier(
        mps_train, truncated_quimb_hairy_bitstrings, seed=420, full_sized=True
    )
    max_D = D_total

    for k, (site_a, site_b) in enumerate(
        zip(
            truncated_mpo_classifier.tensors,
            truncated_full_sized_mpo_classifier.tensors,
        )
    ):
        d1, s1, i1, j1 = site_a.shape
        d2, s2, i2, j2 = site_b.shape

        assert d1 == d2
        assert s1 == s2

        if k == 0:
            assert i2 == 1
            assert j2 == max_D

        elif k == (mpo_classifier.num_tensors - 1):
            assert i2 == max_D
            assert j2 == 1

        else:
            assert i2 == max_D
            assert j2 == max_D

    # Truncated, one site
    # Check shape is correct
    one_site_truncated_mpo_train = mpo_encoding(
        mps_train, y_train, truncated_one_site_quimb_hairy_bitstrings
    )

    for site_a, site_b in zip(
        one_site_truncated_mpo_classifier.tensors,
        one_site_truncated_mpo_train[0].tensors,
    ):
        assert site_a.shape == site_b.shape

    # Check classifier is normalised
    assert np.isclose(
        one_site_truncated_mpo_classifier.H @ one_site_truncated_mpo_classifier, 1
    )

    # Check full sized shape is correct
    one_site_truncated_full_sized_mpo_classifier = create_mpo_classifier(
        mps_train, truncated_one_site_quimb_hairy_bitstrings, seed=420, full_sized=True
    )
    max_D = D_total

    for k, (site_a, site_b) in enumerate(
        zip(
            one_site_truncated_mpo_classifier.tensors,
            one_site_truncated_full_sized_mpo_classifier.tensors,
        )
    ):
        d1, s1, i1, j1 = site_a.shape
        d2, s2, i2, j2 = site_b.shape

        assert d1 == d2
        assert s1 == s2

        if k == 0:
            assert i2 == 1
            assert j2 == max_D

        elif k == (mpo_classifier.num_tensors - 1):
            assert i2 == max_D
            assert j2 == 1

        else:
            assert i2 == max_D
            assert j2 == max_D


def test_create_mpo_classifier_from_initialised_classifier():

    # check randomly initiaised classifier has same shape
    random_classifier = create_mpo_classifier_from_initialised_classifier(
        mpo_classifier, seed=420
    )

    for site_a, site_b in zip(mpo_classifier.tensors, random_classifier.tensors):
        d1, s1, i1, j1 = site_a.shape
        d2, s2, i2, j2 = site_b.shape

        assert d1 == d2
        assert s1 == s2
        assert i1 == i2
        assert j1 == j2

    # check random classifier is normalized
    assert np.isclose(random_classifier.H @ random_classifier, 1)

    # Check truncated
    truncated_random_classifier = create_mpo_classifier_from_initialised_classifier(
        truncated_mpo_classifier, seed=420
    )
    for site_a, site_b in zip(
        truncated_mpo_classifier.tensors, truncated_random_classifier.tensors
    ):
        d1, s1, i1, j1 = site_a.shape
        d2, s2, i2, j2 = site_b.shape

        assert d1 == d2
        assert s1 == s2
        assert i1 == i2
        assert j1 == j2

    # check random classifier is normalized
    assert np.isclose(truncated_random_classifier.H @ truncated_random_classifier, 1)

    # Check truncated, one site
    one_site_truncated_random_classifier = (
        create_mpo_classifier_from_initialised_classifier(
            one_site_truncated_mpo_classifier, seed=420
        )
    )
    for site_a, site_b in zip(
        one_site_truncated_mpo_classifier.tensors,
        one_site_truncated_random_classifier.tensors,
    ):
        d1, s1, i1, j1 = site_a.shape
        d2, s2, i2, j2 = site_b.shape

        assert d1 == d2
        assert s1 == s2
        assert i1 == i2
        assert j1 == j2

    # check random classifier is normalized
    assert np.isclose(
        one_site_truncated_random_classifier.H @ one_site_truncated_random_classifier, 1
    )


# Assumes squeezed TNs
def test_loss_functions():
    def old_stoundenmire_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
        # Loss for more than one class
        # Trains over all images in classes specified by y_train
        n_samples = len(mps_train)
        possible_labels = list(set(y_train))
        overlaps = []
        for i in range(len(mps_train)):
            for label in possible_labels:

                if y_train[i] == label:
                    overlap = (
                        np.real(
                            mps_train[i].squeeze().H
                            @ (classifier @ q_hairy_bitstrings[label].squeeze())
                        )
                        - 1
                    ) ** 2
                else:
                    overlap = (
                        np.real(
                            mps_train[i].squeeze().H
                            @ (classifier @ q_hairy_bitstrings[label].squeeze())
                        )
                        ** 2
                    )
                overlaps.append(overlap)
        # print(overlaps)
        return np.sum(overlaps) / len(mps_train)

    def old_abs_stoundenmire_loss(classifier, mps_train, q_hairy_bitstrings, y_train):
        # Loss for more than one class
        # Trains over all images in classes specified by y_train
        n_samples = len(mps_train)
        possible_labels = list(set(y_train))
        overlaps = []
        for i in range(len(mps_train)):
            for label in possible_labels:

                if y_train[i] == label:
                    overlap = (
                        abs(
                            mps_train[i].squeeze().H
                            @ (classifier @ q_hairy_bitstrings[label].squeeze())
                        )
                        - 1
                    ) ** 2
                else:
                    overlap = (
                        np.real(
                            mps_train[i].squeeze().H
                            @ (classifier @ q_hairy_bitstrings[label].squeeze())
                        )
                        ** 2
                    )
                overlaps.append(overlap)
        # print(overlaps)
        return np.sum(overlaps) / len(mps_train)

    # Check loss is -1.0 between same image as mps image and mpo classifier
    truncated_mpo_train = [
        i.squeeze()
        for i in mpo_encoding(
            mps_train, y_train, truncated_one_site_quimb_hairy_bitstrings
        )
    ]

    squeezed_bitstrings = [i for i in truncated_one_site_quimb_hairy_bitstrings]

    truncated_multi_site_mpo_train = [
        i for i in mpo_encoding(mps_train, y_train, truncated_quimb_hairy_bitstrings)
    ]

    squeezed_multi_site_bitstrings = [i for i in truncated_quimb_hairy_bitstrings]

    digit = truncated_mpo_train[0]
    multi_site_digit = truncated_multi_site_mpo_train[0]

    # List of same images
    mps_images = [mps_train[0] for _ in range(10)]
    train_label = [y_train[0] for _ in range(10)]

    # Green Loss
    loss_one_site = green_loss(
        digit.squeeze(), mps_images, squeezed_bitstrings, train_label
    )
    loss_multi_site = green_loss(
        multi_site_digit.squeeze(),
        mps_images,
        squeezed_multi_site_bitstrings,
        train_label,
    )
    assert np.round(loss_one_site, 3) == -1.0
    assert loss_one_site == loss_multi_site

    # Abs Green Loss
    loss_one_site = abs_green_loss(
        digit.squeeze(), mps_images, squeezed_bitstrings, train_label
    )
    loss_multi_site = abs_green_loss(
        multi_site_digit.squeeze(),
        mps_images,
        squeezed_multi_site_bitstrings,
        train_label,
    )
    assert np.round(loss_one_site, 3) == -1.0
    assert loss_one_site == loss_multi_site

    # MSE Loss
    loss_one_site = mse_loss(
        digit.squeeze(), mps_images, squeezed_bitstrings, train_label
    )
    loss_multi_site = mse_loss(
        multi_site_digit.squeeze(),
        mps_images,
        squeezed_multi_site_bitstrings,
        train_label,
    )
    assert np.round(loss_one_site, 3) == 0.0
    assert loss_one_site == loss_multi_site

    # Abs MSE Loss
    loss_one_site = abs_mse_loss(
        digit.squeeze(), mps_images, squeezed_bitstrings, train_label
    )
    loss_multi_site = mse_loss(
        multi_site_digit.squeeze(),
        mps_images,
        squeezed_multi_site_bitstrings,
        train_label,
    )
    assert np.round(loss_one_site, 3) == 0.0
    assert loss_one_site == loss_multi_site

    # Cross Entropy Loss
    loss_one_site = cross_entropy_loss(
        digit.squeeze(), mps_images, squeezed_bitstrings, train_label
    )
    loss_multi_site = cross_entropy_loss(
        multi_site_digit.squeeze(),
        mps_images,
        squeezed_multi_site_bitstrings,
        train_label,
    )
    assert np.round(loss_one_site, 3) == 0.0
    assert loss_one_site == loss_multi_site

    # Stoudenmire Loss
    loss_one_site = stoudenmire_loss(
        digit.squeeze(), mps_images, squeezed_bitstrings, train_label
    )
    old_loss = old_stoundenmire_loss(
        digit, mps_images, squeezed_bitstrings, train_label
    )
    loss_multi_site = stoudenmire_loss(
        multi_site_digit.squeeze(),
        mps_images,
        squeezed_multi_site_bitstrings,
        train_label,
    )
    assert np.round(loss_one_site, 3) == 0.0
    assert old_loss == loss_one_site
    assert loss_one_site == loss_multi_site

    # Abs stoudenmire Loss
    loss_one_site = abs_stoudenmire_loss(
        digit.squeeze(), mps_images, squeezed_bitstrings, train_label
    )
    old_loss = old_abs_stoundenmire_loss(
        digit, mps_images, squeezed_bitstrings, train_label
    )
    loss_multi_site = abs_stoudenmire_loss(
        multi_site_digit.squeeze(),
        mps_images,
        squeezed_multi_site_bitstrings,
        train_label,
    )
    assert np.round(loss_one_site, 3) == 0.0
    assert old_loss == loss_one_site
    assert loss_one_site == loss_multi_site

    # Check loss between images and randomly initialised mpo classifier
    # All overlaps should be roughly equal. To a degree!

    # Untruncated
    # Just test untruncated for one. No need to do all.
    squeezed_untruncated_mpo_classifier = mpo_classifier.squeeze()
    squeezed_untruncated_bitstrings = [i for i in quimb_hairy_bitstrings]
    # Truncated
    squeezed_truncated_mpo_classifier = truncated_mpo_classifier.squeeze()
    squeezed_truncated_bitstrings = [i for i in truncated_quimb_hairy_bitstrings]

    squeezed_images = [i for i in mps_train][:10]

    # Green Loss
    truncated_loss = green_loss(
        squeezed_truncated_mpo_classifier,
        squeezed_images,
        squeezed_truncated_bitstrings,
        y_train[:10],
    )
    truncated_overlap = -(
        (
            squeezed_images[0].squeeze().H
            @ (
                squeezed_truncated_mpo_classifier
                @ squeezed_truncated_bitstrings[y_train[0]].squeeze()
            )
        )
        ** 2
    )
    assert np.isclose(truncated_loss, truncated_overlap, atol=1e-03)

    # Abs Green Loss
    loss = abs_green_loss(
        squeezed_truncated_mpo_classifier,
        squeezed_images,
        squeezed_truncated_bitstrings,
        y_train[:10],
    )
    overlap = (
        -abs(
            squeezed_images[0].squeeze().H
            @ (
                squeezed_truncated_mpo_classifier
                @ squeezed_truncated_bitstrings[y_train[0]].squeeze()
            )
        )
        ** 2
    )
    assert np.isclose(loss, overlap, atol=1e-03)

    # MSE Loss
    loss = mse_loss(
        squeezed_truncated_mpo_classifier,
        squeezed_images,
        squeezed_truncated_bitstrings,
        y_train[:10],
    )
    overlap = (
        (
            squeezed_images[0].squeeze().H
            @ (
                squeezed_truncated_mpo_classifier
                @ squeezed_truncated_bitstrings[y_train[0]].squeeze()
            )
        )
        - 1
    ) ** 2
    assert np.isclose(loss, overlap, atol=1e-01)

    # Abs MSE Loss
    loss = abs_mse_loss(
        squeezed_truncated_mpo_classifier,
        squeezed_images,
        squeezed_truncated_bitstrings,
        y_train[:10],
    )
    overlap = (
        abs(
            squeezed_images[0].squeeze().H
            @ (
                squeezed_truncated_mpo_classifier
                @ squeezed_truncated_bitstrings[y_train[0]].squeeze()
            )
        )
        - 1
    ) ** 2
    assert np.isclose(loss, overlap, atol=1e-01)

    # Cross Entropy Loss
    # Variance is higher than usual with this one. Just check both numbers are of same order
    loss = cross_entropy_loss(
        squeezed_truncated_mpo_classifier,
        squeezed_images,
        squeezed_truncated_bitstrings,
        y_train[:10],
    )

    overlap = anp.log(
        abs(
            squeezed_images[0].squeeze().H
            @ (
                squeezed_truncated_mpo_classifier
                @ squeezed_truncated_bitstrings[y_train[0]].squeeze()
            )
        )
    )
    assert (int(np.log(abs(loss))) + 1) == (int(np.log(abs(overlap))) + 1)

    # Stoudenmire Loss
    loss = stoudenmire_loss(
        squeezed_truncated_mpo_classifier,
        squeezed_images,
        squeezed_truncated_bitstrings,
        y_train[:10],
    )
    old_loss = old_stoundenmire_loss(
        squeezed_truncated_mpo_classifier,
        squeezed_images,
        squeezed_truncated_bitstrings,
        y_train[:10],
    )

    mini_possible_labels = list(set(y_train[:10]))
    overlap = np.sum(
        [
            (
                anp.real(
                    squeezed_images[0].squeeze().H
                    @ (
                        squeezed_truncated_mpo_classifier
                        @ squeezed_truncated_bitstrings[y_train[0]].squeeze()
                    )
                )
                - int(y_train[0] == label)
            )
            ** 2
            for label in mini_possible_labels
        ]
    )
    assert np.isclose(loss, overlap, atol=1e-01)
    assert loss == old_loss

    # Abs Stoudenmire Loss
    loss = abs_stoudenmire_loss(
        squeezed_truncated_mpo_classifier,
        squeezed_images,
        squeezed_truncated_bitstrings,
        y_train[:10],
    )
    old_loss = old_abs_stoundenmire_loss(
        squeezed_truncated_mpo_classifier,
        squeezed_images,
        squeezed_truncated_bitstrings,
        y_train[:10],
    )

    mini_possible_labels = list(set(y_train[:10]))
    overlap = np.sum(
        [
            (
                abs(
                    squeezed_images[0].squeeze().H
                    @ (
                        squeezed_truncated_mpo_classifier
                        @ squeezed_truncated_bitstrings[y_train[0]].squeeze()
                    )
                )
                - int(y_train[0] == label)
            )
            ** 2
            for label in mini_possible_labels
        ]
    )
    assert np.isclose(loss, overlap, atol=1e-01)
    assert loss == old_loss

    untruncated_loss = abs_stoudenmire_loss(
        squeezed_untruncated_mpo_classifier,
        squeezed_images,
        squeezed_untruncated_bitstrings,
        y_train[:10],
    )
    untruncated_overlap = np.sum(
        [
            (
                abs(
                    squeezed_images[0].squeeze().H
                    @ (
                        squeezed_untruncated_mpo_classifier
                        @ squeezed_untruncated_bitstrings[y_train[0]].squeeze()
                    )
                )
                - int(y_train[0] == label)
            )
            ** 2
            for label in mini_possible_labels
        ]
    )
    assert np.isclose(untruncated_loss, untruncated_overlap, atol=1e-03)


def test_classifier_predictions():

    # Check all predictions are positive
    assert np.array(
        [[pred > 0.0 for pred in image_pred] for image_pred in predictions]
    ).all()

    # Check there's a predicition for each image
    assert predictions.shape[0] == n_train

    # Check there's a prediction from each class
    assert predictions.shape[1] == len(possible_labels)

    # Check image predicitions are not the same. Just check neighbours.
    assert np.array([i != j for i, j in zip(predictions, predictions[1:])]).all()

    # Check label predictions are not the same. Just check one image.
    assert np.array(
        [
            [i != j for i, j in zip(image_pred, image_pred[1:])]
            for image_pred in predictions[:1]
        ]
    ).all()

    # Check squeezed predicitions are the same.
    squeezed_mpo_classifier = mpo_classifier.squeeze()
    squeezed_bitstrings = [i for i in quimb_hairy_bitstrings]
    squeezed_images = [i for i in mps_train]

    squeezed_predictions = np.array(
        classifier_predictions(
            squeezed_mpo_classifier, squeezed_images, squeezed_bitstrings
        )
    )

    assert np.array(
        [np.isclose(i, j) for i, j in zip(predictions, squeezed_predictions)]
    ).all()


def test_padded_classifier_predictions():
    mps_train = mps_encoding(x_train, 32)

    fmpo_classifier = prepare_batched_classifier(
            mps_train, y_train, 32, 10
        ).compress_one_site(D = None, orthogonalise = True)
    qtn_classifier = data_to_QTN(fmpo_classifier.data)

    uclassifier = unitary_qtn(qtn_classifier)

    bitstrings_data = create_padded_bitstrings_data(possible_labels, uclassifier)
    padded_bitstrings = padded_bitstring_data_to_QTN(
        bitstrings_data, uclassifier)

    #Check predictions are the same for same padding combination.
    predictions = classifier_predictions(qtn_classifier.squeeze(), mps_train, truncated_one_site_quimb_hairy_bitstrings)
    upredictions = padded_classifier_predictions(uclassifier.squeeze(), mps_train, padded_bitstrings[:1])
    assert(np.isclose(predictions, upredictions).all())

    #Check padded predictions are same shape as normal predicitions for more than one padding
    upredictions = padded_classifier_predictions(uclassifier.squeeze(), mps_train, padded_bitstrings[:2])
    assert(np.array(upredictions).shape == np.array(predictions).shape)



def test_evaluate_classifier_top_k_accuracy():

    # Test whether an initial classifier displays correct results
    # Should have accuracy ~ 1/#classes
    assert (
        np.round(evaluate_classifier_top_k_accuracy(predictions, y_train, 1), 1) == 0.1
    )

    # Test whether an initial classifier displays correct results
    # Should have accuracy ~ 1/#classes * k
    assert (
        np.round(evaluate_classifier_top_k_accuracy(predictions, y_train, 3), 1) == 0.3
    )

    # Check if mpo_classifier is image of the same class
    # Result is 100% accuracy with test images of same class

    # Untruncated
    for label in possible_labels:
        digit = mpo_train[0]

        # Test images from same class (only use 20 images to speed up testing)
        mps_test = np.array(mps_train)[y_train == y_train[0]][:20]
        y_test = [y_train[0] for _ in range(len(mps_test))]
        prediction = np.array(
            classifier_predictions(digit.squeeze(), mps_test, quimb_hairy_bitstrings)
        )

        assert (
            np.round(evaluate_classifier_top_k_accuracy(prediction, y_test, 3), 3)
            == 1.0
        )

    # Truncated
    for label in possible_labels:
        digit = mpo_encoding(mps_train, y_train, truncated_quimb_hairy_bitstrings)[0]

        # Test images from same class (only use 20 images to speed up testing)
        mps_test = np.array(mps_train)[y_train == y_train[0]][:20]
        y_test = [y_train[0] for _ in range(len(mps_test))]
        prediction = np.array(
            classifier_predictions(
                digit.squeeze(), mps_test, truncated_quimb_hairy_bitstrings
            )
        )

        assert (
            np.round(evaluate_classifier_top_k_accuracy(prediction, y_test, 3), 3)
            == 1.0
        )


if __name__ == "__main__":
    #test_padded_classifier_predictions()
    # test_create_mpo_classifier_from_initialised_classifier()
    test_loss_functions()
