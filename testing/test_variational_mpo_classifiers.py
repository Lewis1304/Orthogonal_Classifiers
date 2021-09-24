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

from xmps.ncon import ncon as nc


def ncon(*args, **kwargs):
    return nc(
        *args, check_indices=False, **kwargs
    )  # make default ncon not check indices


n_train = 100
n_test = 100
D_total = 10

x_train, y_train, x_test, y_test = load_data(n_train, n_test)

possible_labels = list(set(y_train))
n_hairysites = int(np.ceil(math.log(len(possible_labels), 4)))
n_sites = int(np.ceil(math.log(x_train.shape[-1], 2)))
n_pixels = len(x_train[0])

hairy_bitstrings_data_untruncated_data = create_hairy_bitstrings_data(
    possible_labels, n_hairysites, n_sites
)

one_site_bitstrings_data_untruncated_data = create_hairy_bitstrings_data(
    possible_labels, n_hairysites, n_sites, one_site=True
)


quimb_hairy_bitstrings = bitstring_data_to_QTN(
    hairy_bitstrings_data_untruncated_data, n_hairysites, n_sites, truncated=False
)
truncated_quimb_hairy_bitstrings = bitstring_data_to_QTN(
    hairy_bitstrings_data_untruncated_data, n_hairysites, n_sites, truncated=True
)
truncated_one_site_quimb_hairy_bitstrings = bitstring_data_to_QTN(
    one_site_bitstrings_data_untruncated_data, n_hairysites, n_sites, truncated=True
)


fmps_images = [image_to_mps(image, D_total) for image in x_train]

mps_train = mps_encoding(x_train, D_total)
mpo_train = mpo_encoding(mps_train, y_train, quimb_hairy_bitstrings)

mpo_classifier = create_mpo_classifier(mps_train, quimb_hairy_bitstrings, seed=420)

predictions = np.array(
    classifier_predictions(mpo_classifier, mps_train, quimb_hairy_bitstrings)
)

def test_create_hairy_bitstrings_data():

    # Test correct shape: #classes, #sites, dim(s)
    dim_s = 4
    assert hairy_bitstrings_data_untruncated_data.shape == (
        len(possible_labels),
        n_sites,
        dim_s,
    )

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
            : (n_sites - n_hairysites)
        ]:
            assert np.array_equal(site, [1, 0, 0, 0])

    # Test one_site
    for label in possible_labels:
        for site in one_site_bitstrings_data_untruncated_data[label][: (n_sites - 1)]:
            assert np.array_equal(site, np.eye(16)[0])


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
                4,
                1,
                1,
            )

    # Check data isn't altered when converted to product state
    for label in possible_labels:
        for a, b in zip(
            hairy_bitstrings_data_untruncated_data[label], product_states[label]
        ):
            assert np.array_equal(a, b.squeeze())


# TODO: Add truncated_quimb_hairy_bitstrings test
def test_class_encode_mps_to_mpo():
    # Encode a single image with all different classes.
    mpo_train = [
        class_encode_mps_to_mpo(mps_train[0], label, quimb_hairy_bitstrings, n_sites)
        for label in possible_labels
    ]

    # Check whether mpo encoded data returns mps data.
    # By projecting onto corresponding label bitstring.
    # Assume |0> padding for now.
    for label in possible_labels:
        for i, (mpo_site, mps_site, bs_site) in enumerate(
            zip(mpo_train[label], mps_train[0], quimb_hairy_bitstrings[label])
        ):
            if i < (n_sites - n_hairysites):
                # same as projecting onto |00> state
                assert np.array_equal(mpo_site[:, 0, :, :], mps_site.data)
            else:
                # Projecting onto label bitstring state for that site
                proj = np.einsum("dsij, s...", mpo_site, bs_site.data)
                assert np.array_equal(proj.squeeze(), mps_site.data.squeeze())


# TODO: Add truncated_quimb_hairy_bitstrings test
def test_create_mpo_classifier():

    # Check shape is correct
    for site_a, site_b in zip(mpo_classifier.tensors, mpo_train[0].tensors):
        assert site_a.shape == site_b.shape

    # Check classifier is normalised
    assert np.isclose(mpo_classifier.H @ mpo_classifier, 1)

# TODO: Add truncated_quimb_hairy_bitstrings test
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
                        np.real(mps_train[i].H @ (classifier @ q_hairy_bitstrings[label]))
                    - 1) ** 2
                else:
                    overlap = np.real(
                        mps_train[i].H @ (classifier @ q_hairy_bitstrings[label])
                    ) ** 2
                overlaps.append(overlap)
        #print(overlaps)
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
                        abs(mps_train[i].H @ (classifier @ q_hairy_bitstrings[label]))
                    - 1) ** 2
                else:
                    overlap = np.real(
                        mps_train[i].H @ (classifier @ q_hairy_bitstrings[label])
                    ) ** 2
                overlaps.append(overlap)
        #print(overlaps)
        return np.sum(overlaps) / len(mps_train)

    # Check loss is -1.0 between same image as mps image and mpo classifier
    truncated_mpo_train = [
        i.squeeze()
        for i in mpo_encoding(
            mps_train, y_train, truncated_one_site_quimb_hairy_bitstrings
        )
    ]

    digit = truncated_mpo_train[0]

    # List of same images
    mps_images = [mps_train[0].squeeze() for _ in range(10)]
    train_label = [y_train[0] for _ in range(10)]

    squeezed_bitstrings = [
        i.squeeze() for i in truncated_one_site_quimb_hairy_bitstrings
    ]

    # Green Loss
    loss = green_loss(digit, mps_images, squeezed_bitstrings, train_label)
    assert np.round(loss, 3) == -1.0

    # Abs Green Loss
    loss = abs_green_loss(digit, mps_images, squeezed_bitstrings, train_label)
    assert np.round(loss, 3) == -1.0

    # MSE Loss
    loss = mse_loss(digit, mps_images, squeezed_bitstrings, train_label)
    assert np.round(loss, 3) == 0.0

    # Abs MSE Loss
    loss = abs_mse_loss(digit, mps_images, squeezed_bitstrings, train_label)
    assert np.round(loss, 3) == 0.0

    # Cross Entropy Loss
    loss = cross_entropy_loss(digit, mps_images, squeezed_bitstrings, train_label)
    assert np.round(loss, 3) == 0.0

    # Stoudenmire Loss
    loss = stoudenmire_loss(digit, mps_images, squeezed_bitstrings, train_label)
    old_loss = old_stoundenmire_loss(digit, mps_images, squeezed_bitstrings, train_label)
    assert np.round(loss, 3) == 0.0

    # Abs stoudenmire Loss
    loss = abs_stoudenmire_loss(digit, mps_images, squeezed_bitstrings, train_label)
    old_loss = old_abs_stoundenmire_loss(digit, mps_images, squeezed_bitstrings, train_label)
    assert np.round(loss, 3) == 0.0
    assert(old_loss == loss)

    # Check loss between images and randomly initialised mpo classifier
    # All overlaps should be roughly equal. To a degree!
    squeezed_mpo_classifier = create_mpo_classifier(mps_train,
            truncated_quimb_hairy_bitstrings
    ).squeeze()
    squeezed_images = [i.squeeze() for i in mps_train][:10]
    squeezed_bitstrings = [i.squeeze() for i in truncated_quimb_hairy_bitstrings]

    # Green Loss
    loss = green_loss(
        squeezed_mpo_classifier, squeezed_images, squeezed_bitstrings, y_train[:10]
    )
    overlap = -(
        (
            squeezed_images[0].H
            @ (squeezed_mpo_classifier @ squeezed_bitstrings[y_train[0]])
        )
        ** 2
    )
    assert np.isclose(loss, overlap, atol=1e-03)

    # Abs Green Loss
    loss = abs_green_loss(
        squeezed_mpo_classifier, squeezed_images, squeezed_bitstrings, y_train[:10]
    )
    overlap = (
        -abs(
            squeezed_images[0].H
            @ (squeezed_mpo_classifier @ squeezed_bitstrings[y_train[0]])
        )
        ** 2
    )
    assert np.isclose(loss, overlap, atol=1e-03)

    # MSE Loss
    loss = mse_loss(
        squeezed_mpo_classifier, squeezed_images, squeezed_bitstrings, y_train[:10]
    )
    overlap = (
        (
            squeezed_images[0].H
            @ (squeezed_mpo_classifier @ squeezed_bitstrings[y_train[0]])
        )
        - 1
    ) ** 2
    assert np.isclose(loss, overlap, atol=1e-01)

    # Abs MSE Loss
    loss = abs_mse_loss(
        squeezed_mpo_classifier, squeezed_images, squeezed_bitstrings, y_train[:10]
    )
    overlap = (
        abs(
            squeezed_images[0].H
            @ (squeezed_mpo_classifier @ squeezed_bitstrings[y_train[0]])
        )
        - 1
    ) ** 2
    assert np.isclose(loss, overlap, atol=1e-01)

    # Cross Entropy Loss
    # Variance is higher than usual with this one. Just check both numbers are of same order
    loss = cross_entropy_loss(
        squeezed_mpo_classifier, squeezed_images, squeezed_bitstrings, y_train[:10]
    )
    overlap = anp.log(
        abs(
            squeezed_images[0].H
            @ (squeezed_mpo_classifier @ squeezed_bitstrings[y_train[0]])
        )
    )
    assert (int(np.log(abs(loss))) + 1) == (int(np.log(abs(overlap))) + 1)

    # Stoudenmire Loss
    loss = stoudenmire_loss(
        squeezed_mpo_classifier, squeezed_images, squeezed_bitstrings, y_train[:10]
    )
    old_loss = old_stoundenmire_loss(squeezed_mpo_classifier, squeezed_images, squeezed_bitstrings, y_train[:10])

    mini_possible_labels = list(set(y_train[:10]))
    overlap = np.sum(
        [
            (
                anp.real(
                    squeezed_images[0].H
                    @ (squeezed_mpo_classifier @ squeezed_bitstrings[y_train[0]])
                )
                - int(y_train[0] == label)
            )
            ** 2
            for label in mini_possible_labels
        ]
    )
    assert np.isclose(loss, overlap, atol=1e-01)
    assert(loss == old_loss)

    # Abs Stoudenmire Loss
    loss = abs_stoudenmire_loss(
        squeezed_mpo_classifier, squeezed_images, squeezed_bitstrings, y_train[:10]
    )
    old_loss = old_abs_stoundenmire_loss(squeezed_mpo_classifier, squeezed_images, squeezed_bitstrings, y_train[:10])

    mini_possible_labels = list(set(y_train[:10]))
    overlap = np.sum(
        [
            (
                abs(
                    squeezed_images[0].H
                    @ (squeezed_mpo_classifier @ squeezed_bitstrings[y_train[0]])
                )
                - int(y_train[0] == label)
            )
            ** 2
            for label in mini_possible_labels
        ]
    )
    assert np.isclose(loss, overlap, atol=1e-01)
    assert(loss == old_loss)

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


# TODO: Add truncated_quimb_hairy_bitstrings test
def test_evaluate_classifier_accuracy():

    # Test whether an initial classifier displays correct results
    # Should have accuracy ~ 1/#classes
    assert np.round(evaluate_classifier_accuracy(predictions, y_train), 1) == 0.1

    # Check if mpo_classifier is image of the same class
    # Result is 100% accuracy with test images of same class
    for label in possible_labels:
        digit = mpo_train[0]

        # Test images from same class (only use 20 images to speed up testing)
        mps_test = np.array(mps_train)[y_train == y_train[0]][:20]
        y_test = [y_train[0] for _ in range(len(mps_test))]
        prediction = np.array(
            classifier_predictions(digit, mps_test, quimb_hairy_bitstrings)
        )

        assert np.round(evaluate_classifier_accuracy(prediction, y_test), 3) == 1.0

    # TO DO
    # Check if mpo_classifier is image of the same class.
    # Result is 100% accuracy with test images of same class. Requires padded encoding on images.


# TODO: Add truncated_quimb_hairy_bitstrings test
def test_evaluate_classifier_top_k_accuracy():

    # Test whether an initial classifier displays correct results
    # Should have accuracy ~ 1/#classes * k
    assert (
        np.round(evaluate_classifier_top_k_accuracy(predictions, y_train, 3), 1) == 0.3
    )

    # Check if mpo_classifier is image of the same class
    # Result is 100% accuracy with test images of same class
    for label in possible_labels:
        digit = mpo_train[0]

        # Test images from same class (only use 20 images to speed up testing)
        mps_test = np.array(mps_train)[y_train == y_train[0]][:20]
        y_test = [y_train[0] for _ in range(len(mps_test))]
        prediction = np.array(
            classifier_predictions(digit, mps_test, quimb_hairy_bitstrings)
        )

        assert (
            np.round(evaluate_classifier_top_k_accuracy(prediction, y_test, 3), 3)
            == 1.0
        )

    # TO DO
    # Check if mpo_classifier is image of the same class.
    # Result is 100% accuracy with test images of same class. Requires padded encoding on images.


if __name__ == "__main__":
    test_loss_functions()
