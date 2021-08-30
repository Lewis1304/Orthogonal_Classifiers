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

# Only do 2 paddings  (for speed)
hairy_bitstrings_data_padded_data = create_padded_hairy_bitstrings_data(
    possible_labels, n_hairysites, n_sites
)[:2]

quimb_hairy_bitstrings = bitstring_data_to_QTN(
    hairy_bitstrings_data_untruncated_data, n_hairysites, n_sites, truncated=False
)
truncated_quimb_hairy_bitstrings = bitstring_data_to_QTN(
    hairy_bitstrings_data_untruncated_data, n_hairysites, n_sites, truncated=True
)

quimb_padded_hairy_bitstrings = [
    bitstring_data_to_QTN(padding, n_hairysites, n_sites)
    for padding in hairy_bitstrings_data_padded_data
]

fmps_images = [image_to_mps(image, D_total) for image in x_train]

mps_train = mps_encoding(x_train, D_total)
mpo_train = mpo_encoding(mps_train, y_train, quimb_hairy_bitstrings)

mpo_classifier = create_mpo_classifier(mpo_train, seed=420)

predictions = np.array(
    classifier_predictions(mpo_classifier, mps_train, quimb_hairy_bitstrings)
)
padded_predictions = np.array(
    padded_classifier_predictions(
        mpo_classifier, mps_train, quimb_padded_hairy_bitstrings
    )
)


def test_create_hairy_bitstrings_data():

    # Test correct shape: #classes, #sites, dim(s)
    dim_s = 4
    assert hairy_bitstrings_data_untruncated_data.shape == (
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


def test_create_padded_hairy_bitstrings_data():

    # Test correct shape: #possible paddings, #classes, #sites, dim(s)
    dim_s = 4
    possible_paddings = len(hairy_bitstrings_data_padded_data)
    assert hairy_bitstrings_data_padded_data.shape == (
        possible_paddings,
        len(possible_labels),
        n_sites,
        dim_s,
    )

    # Test whether first (n_sites - n_hairysites) sites have correct padding
    # Check for all classes
    bitstrings_others = [
        bin(k)[2:].zfill(n_sites - n_hairysites) for k in range(possible_paddings)
    ]
    bitstrings_others_arrays = [
        np.array([int(i) for i in bitstring]) for bitstring in bitstrings_others
    ]

    for k in range(possible_paddings):
        for label in possible_labels:
            for site, state in zip(
                hairy_bitstrings_data_padded_data[k][label][: (n_sites - n_hairysites)],
                bitstrings_others_arrays[k],
            ):
                # Only one qubit. Therefore only 2 basis states.
                if state == 0:
                    assert np.array_equal(site, [1, 0, 0, 0])
                elif state == 1:
                    assert np.array_equal(site, [0, 1, 0, 0])
                else:
                    assert ()

    # Check if label encoding is correct.
    # Each hairy site has 2 qubits i.e. dim(s) == 2**2.
    # Basis vectors for s.shape=4 site is:
    # [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] with
    # |00>, |01>, |10>, |11> respectively.
    # So label "3" = "0011" should have [1,0,0,0] and [0,0,0,1] for the hairy sites
    basis_vectors = {
        "00": [1, 0, 0, 0],
        "01": [0, 1, 0, 0],
        "10": [0, 0, 1, 0],
        "11": [0, 0, 0, 1],
    }
    bitstrings = create_bitstrings(possible_labels, n_hairysites)

    # untruncated
    for k in range(possible_paddings):
        for label in possible_labels:
            for i, site in enumerate(
                hairy_bitstrings_data_padded_data[k][label][(n_sites - n_hairysites) :]
            ):
                site_qubits_state = bitstrings[label][2 * i : 2 * (i + 1)]
                assert np.array_equal(site, basis_vectors[site_qubits_state])


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


def test_padded_bitstring_to_product_state_data():

    # Check for 2 paddings
    padded_product_states = [
        bitstring_to_product_state_data(k)
        for k in hairy_bitstrings_data_padded_data[:2]
    ]

    # Check whether all labels are converted to product_states
    for k in padded_product_states:
        assert len(k) == len(possible_labels)

    # Check correct number of sites:
    for k in padded_product_states:
        for label in possible_labels:
            assert len(k[label]) == n_sites

    # Check shape of each product state is correct
    # site.shape = s, i, j
    for k in padded_product_states:
        for label in possible_labels:
            for site in k[label]:
                assert site.shape == (
                    4,
                    1,
                    1,
                )

    # Check data isn't altered when converted to product state
    for k in range(len(padded_product_states)):
        for label in possible_labels:
            for a, b in zip(
                hairy_bitstrings_data_padded_data[k][label],
                padded_product_states[k][label],
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
    mpo_train = mpo_encoding(mps_train, y_train[:100], quimb_hairy_bitstrings)

    mpo_classifier = create_mpo_classifier(mpo_train)

    # Check shape is correct
    for site_a, site_b in zip(mpo_classifier.tensors, mpo_train[0].tensors):
        assert site_a.shape == site_b.shape

    # Check classifier is normalised
    assert np.isclose(mpo_classifier.H @ mpo_classifier, 1)


def test_compress():

    # Truncated mpo needed otherwise dim(s) explodes when canonicalising
    truncated_mpo_classifier_data = [
        site.data[:, :1, :, :] if i < (n_sites - n_hairysites) else site.data
        for i, site in enumerate(mpo_classifier.tensors)
    ]
    truncated_mpo_classifier = data_to_QTN(truncated_mpo_classifier_data)
    truncated_mpo_classifier /= (
        truncated_mpo_classifier.H @ truncated_mpo_classifier
    ) ** 0.5

    compressed_mpo = compress_QTN(truncated_mpo_classifier, D=None, orthogonalise=False)

    # Check norm is still 1
    assert np.isclose((compressed_mpo.H @ compressed_mpo), 1)

    # Check overlap between initial classifier
    # and compressed mpo with D=None is 1.
    assert np.isclose((truncated_mpo_classifier.H @ compressed_mpo).norm(), 1)

    # Check canonicl form- compress procedure leaves mpo in mixed canonical form
    # center site is at left most hairest site.
    for n, site in enumerate(compressed_mpo.tensors):
        d, s, i, j = site.shape
        if n < (n_sites - n_hairysites):
            # reshape from (d, s, i, j) --> (d*j, s*i). As SVD was done like that.
            U = site.data.transpose(0, 3, 1, 2).reshape(d * j, s * i)
            Uh = U.conj().T
            assert np.array_equal(np.round(Uh @ U, 5), np.eye(s * i))
        else:
            # reshape from (d, s, i, j) --> (i, s*j*d). As SVD was done like that.
            U = site.data.transpose(2, 1, 3, 0).reshape(i, s * j * d)
            Uh = U.conj().T
            assert np.array_equal(np.round(U @ Uh, 5), np.eye(i))

    # Check compressed has right shape for range of different Ds
    for max_D in range(1, 5):
        compressed_mpo = compress_QTN(
            truncated_mpo_classifier, D=max_D, orthogonalise=False
        )
        for i, (site0, site1) in enumerate(
            zip(compressed_mpo.tensors, truncated_mpo_classifier)
        ):

            d0, s0, i0, j0 = site0.shape
            d1, s1, i1, j1 = site1.shape

            assert d0 == d1
            assert s0 == s1

            assert i0 <= max_D
            assert j0 <= max_D
    # TODO: Orthogonalisation test


def test_batch_adding_mpos():
    mpos = []

    # Truncate images- required for Orthogonalisation procedure
    # to work.
    for mpo in mpo_train:
        truncated_mpo_data = [
            site.data[:, :1, :, :] if i < (n_sites - n_hairysites) else site.data
            for i, site in enumerate(mpo.tensors)
        ]
        truncated_mpo = data_to_QTN(truncated_mpo_data)
        truncated_mpo /= (truncated_mpo.H @ truncated_mpo) ** 0.5
        mpos.append(truncated_mpo)

    grouped_images = [
        [mpos[i] for i in range(n_train) if y_train[i] == label]
        for label in possible_labels
    ]
    grouped_mpos = [mpo_from_each_class for mpo_from_each_class in zip(*grouped_images)]

    batch_added_mpos = batch_adding_mpos(
        grouped_mpos, D=None, orthogonalise=False, compress=False
    )

    # Check, for compress = False that batch_adding_mpos is equivalent to simply adding all mpos.
    flattened_grouped_mpos = [item for sublist in grouped_mpos for item in sublist]
    added_mpos = flattened_grouped_mpos[0]
    for mpo in flattened_grouped_mpos[1:]:
        added_mpos = add_mpos(added_mpos, mpo)
    # "compression" required since compression done in batch addng changes the local values
    added_mpos = compress_QTN(added_mpos, D=None, orthogonalise=False)
    i = 0
    for site0, site1 in zip(batch_added_mpos.tensors, added_mpos.tensors):
        assert np.array_equal(site0.data, site1.data)

    # Check shape is correct for D != None.
    max_D = 5
    batch_added_mpos = batch_adding_mpos(
        grouped_mpos, D=max_D, orthogonalise=False, compress=True
    )
    for site0, site1 in zip(batch_added_mpos.tensors, mpos[0].tensors):
        d0, s0, i0, j0 = site0.shape
        d1, s1, i1, j1 = site1.shape

        assert d0 == d1
        assert s0 == s1

        assert i0 <= max_D
        assert j0 <= max_D


# TODO: Add truncated_quimb_hairy_bitstrings test
def test_green_loss():

    # Check loss is -1.0 between same image as mps image and mpo classifier
    digit = mpo_train[0]

    # List of same images
    mps_images = [np.array(mps_train)[0] for _ in range(10)]
    train_label = [y_train[0] for _ in range(10)]

    loss = green_loss(digit, mps_images, quimb_hairy_bitstrings, train_label)
    assert np.round(loss, 3) == -1.0


# TODO: Add truncated_quimb_hairy_bitstrings test
def test_padded_green_loss():

    # Check loss is -1.0 between same image as mps image and mpo classifier
    # When image is zero-pad encoded (overlap is zero with other padding).
    # Check only for 2 paddings.
    digit = mpo_train[0]
    # List of same images
    mps_images = [np.array(mps_train)[0] for _ in range(10)]
    train_label = [y_train[0] for _ in range(10)]
    loss = padded_green_loss(
        digit, mps_images, quimb_padded_hairy_bitstrings[:2], train_label
    )
    assert np.round(loss, 3) == -1.0

    # Check that first padding is equivalent to green_loss
    padded_loss = padded_green_loss(
        mpo_classifier, mps_images, quimb_padded_hairy_bitstrings[:1], train_label
    )
    non_padded_loss = green_loss(
        mpo_classifier, mps_images, quimb_hairy_bitstrings, train_label
    )
    assert padded_loss == non_padded_loss
    # TO DO
    # Check loss is -1.0 * #possible paddings between same image as mps image and mpo classifier
    # When image is encoded as all possible paddings.


# TODO: Add truncated_quimb_hairy_bitstrings test
def test_stoundenmire_loss():

    # Check loss is 0 between same image as mps image and mpo classifier
    digit = mpo_train[0]

    # List of same images
    mps_images = [np.array(mps_train)[0] for _ in range(10)]
    train_label = [y_train[0] for _ in range(10)]

    loss = stoundenmire_loss(digit, mps_images, quimb_hairy_bitstrings, train_label)

    assert np.isclose(loss, 0)


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


def test_padded_classifier_predictions():

    # Check all predictions are positive
    assert np.array(
        [[pred > 0.0 for pred in image_pred] for image_pred in padded_predictions]
    ).all()

    # Check there's a predicition for each image
    assert padded_predictions.shape[0] == n_train

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

    # Test whether initial classifier displays correct results
    # For padded bitstrings. Should have accuracy ~1/#classes.
    # Since all labels have paddings, any contribution cancels out.
    assert np.round(evaluate_classifier_accuracy(padded_predictions, y_train), 1) == 0.1
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

    # Test whether initial classifier displays correct results
    # For padded bitstrings. Should have accuracy ~1/#classes * k.
    # Since all labels have paddings, any contribution cancels out.
    assert (
        np.round(evaluate_classifier_top_k_accuracy(padded_predictions, y_train, 3), 1)
        == 0.3
    )
    # TO DO
    # Check if mpo_classifier is image of the same class.
    # Result is 100% accuracy with test images of same class. Requires padded encoding on images.


if __name__ == "__main__":
    test_batch_adding_mpos()
