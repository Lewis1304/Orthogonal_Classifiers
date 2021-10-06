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
from deterministic_mpo_classifier import mpo_encoding, class_encode_mps_to_mpo

from xmps.ncon import ncon as nc


def ncon(*args, **kwargs):
    return nc(
        *args, check_indices=False, **kwargs
    )  # make default ncon not check indices


n_train = 100
n_test = 100
D_total = 10

x_train, y_train, x_test, y_test = load_data(n_train, n_test, equal_numbers = True)

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
one_site_quimb_hairy_bitstrings = bitstring_data_to_QTN(
    one_site_bitstrings_data_untruncated_data, n_hairysites, n_sites, truncated=False
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


def test_load_data():

    # Test data shape
    assert x_train.shape == (n_train, n_pixels)
    assert x_test.shape == (n_test, n_pixels)

    assert y_train.shape == (n_train,)
    assert y_test.shape == (n_test,)

    # Test normalised values
    assert np.max(x_train) == 1.0
    assert np.max(x_test) == 1.0

    assert np.min(x_train) == 0.0
    assert np.min(x_test) == 0.0

    # Test all labels are represented in data
    assert list(set(y_train)) == list(range(10))
    assert list(set(y_test)) == list(range(10))

    #Test for equal classes, equal number of each class.
    assert(np.array([len(x_train[y_train == label]) == (n_train // 10) for label in possible_labels]).all())



def test_arrange_data():
    #Requires equal_numbers to be true

    #test random
    random_x_train, random_y_train = arrange_data(x_train, y_train, arrangement = 'random')
    assert(not np.array_equal(random_x_train, x_train))
    assert(not np.array_equal(random_y_train, y_train))

    #test one of each
    one_of_each_x_train, one_of_each_y_train = arrange_data(x_train, y_train, arrangement = 'one of each')
    n_samples_per_class = n_train // len(possible_labels)
    assert(np.array([
            np.array_equal
            (
            one_of_each_y_train[i*len(possible_labels):(i+1)*len(possible_labels)],
            possible_labels
            )
            for i in range(n_samples_per_class)]).all())

    #test one class
    one_class_x_train, one_class_y_train = arrange_data(x_train, y_train, arrangement = 'one class')
    assert(np.array([list(set(one_class_y_train[i*n_samples_per_class:(i+1)*n_samples_per_class])) == [i] for i in possible_labels]).all())


def test_bitstrings():
    # Test for n_hairysites != 1
    bitstrings = create_bitstrings(possible_labels, n_hairysites)

    from_bitstring_to_numbers = [
        sum([int(b) * 2 ** k for k, b in enumerate(bitstrings[i][::-1])])
        for i in possible_labels
    ]
    # Test whether bitstring correctly converts labels to binary
    # By converting back to numeral
    assert from_bitstring_to_numbers == possible_labels


def test_bitstring_data_to_QTN():

    # Check data isn't altered when converted to quimb tensor
    # This checks shape and data
    for label in possible_labels:
        for a, b in zip(
            hairy_bitstrings_data_untruncated_data[label], quimb_hairy_bitstrings[label]
        ):
            assert np.array_equal(a, b.data.squeeze())

    # Check for one-site
    for label in possible_labels:
        for a, b in zip(
            one_site_bitstrings_data_untruncated_data[label],
            one_site_quimb_hairy_bitstrings[label],
        ):
            assert np.array_equal(a, b.data.squeeze())

    # Check bitstrings are orthonormal
    for i, a in enumerate(quimb_hairy_bitstrings):
        for j, b in enumerate(quimb_hairy_bitstrings):
            if i == j:
                # If 1, quimb automatically changes from tensor to int
                assert np.isclose((a @ b), 1)
            else:
                assert np.isclose((a @ b).norm(), 0)

    # Check for one-site
    for i, a in enumerate(one_site_quimb_hairy_bitstrings):
        for j, b in enumerate(one_site_quimb_hairy_bitstrings):
            if i == j:
                # If 1, quimb automatically changes from tensor to int
                assert np.isclose((a @ b), 1)
            else:
                assert np.isclose((a @ b).norm(), 0)

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
    for label in possible_labels:
        for i, site in enumerate(
            quimb_hairy_bitstrings[label].tensors[(n_sites - n_hairysites) :]
        ):
            site_qubits_state = bitstrings[label][2 * i : 2 * (i + 1)]
            assert np.array_equal(site.data.squeeze(), basis_vectors[site_qubits_state])

    # truncated
    for label in possible_labels:
        for i, site in enumerate(
            truncated_quimb_hairy_bitstrings[label].tensors[(n_sites - n_hairysites) :]
        ):
            site_qubits_state = bitstrings[label][2 * i : 2 * (i + 1)]
            assert np.array_equal(site.data.squeeze(), basis_vectors[site_qubits_state])

    # one-site
    for i, label in enumerate(possible_labels):
        site = one_site_quimb_hairy_bitstrings[label].tensors[-1]
        assert np.array_equal(site.data.squeeze(), np.eye(16)[i])

    # Test truncated sites are correct
    # This checks shape and data
    # Test that projection is not null. i.e. equal to 0
    for label in possible_labels:
        for i, site in enumerate(truncated_quimb_hairy_bitstrings[label]):
            if i < (n_sites - n_hairysites):
                assert site.data == np.array([1])

    # Check for one-site
    for label in possible_labels:
        for i, site in enumerate(truncated_one_site_quimb_hairy_bitstrings[label]):
            if i < (n_sites - 1):
                assert site.data == np.array([1])


def test_image_to_mps():

    # Check images are of class fMPS()
    # Check length of MPS is correct
    # Check bond dimension is correct
    # Check local dimension is correct
    for fmps_image in fmps_images:
        assert isinstance(fmps_image, fMPS)
        assert fmps_image.L == n_sites
        assert fmps_image.D == D_total
        assert fmps_image.d == 2

    """
    #Test mps_encoding back to image
    #Doesn't seem to work- though images look the same
    #on matplotlib
    def overlap(mps, prod_state):
        return mps.overlap(fMPS().from_product_state(prod_state))

    def bitstring(k, L):
        string = bin(k)[2:].zfill(L)
        return [[1, 0] * (1 - int(i)) + [0, 1] * int(i) for i in string]

    def fmps_to_image(fmps_image):
        image = (
            np.array([overlap(fmps_image, bitstring(i, 10)) for i in range(n_pixels)])
            ).real
        return image

    fmps_images = [image_to_mps(image, 100) for image in x_train[:5]]
    inverse_images = [fmps_to_image(fmps_image) for fmps_image in fmps_images[:5]]

    assert(np.isclose(i, j) for i, j in zip(inverse_images, x_train[:5]))
    """


def test_fMPS_to_QTN():
    quimb_mps_images = [fMPS_to_QTN(fmps) for fmps in fmps_images]

    # Check correct number of sites
    for label in possible_labels:
        assert quimb_mps_images[label].num_tensors == n_sites

    # Check data isn't altered when converted to quimb tensor
    # This checks shape and data
    for fimage, qimage in zip(fmps_images, quimb_mps_images):
        for fsite, qsite in zip(fimage.data, qimage.tensors):
            assert np.array_equal(fsite.squeeze(), qsite.data.squeeze())


def test_data_to_QTN():

    #Check untruncated multiple sites
    class_encoded_mpos = [
        class_encode_mps_to_mpo(mps_train[0], label, quimb_hairy_bitstrings)
        for label in possible_labels
    ]
    quimb_encoded_mpos = [data_to_QTN(mpo) for mpo in class_encoded_mpos]
    # Check converting to quimb tensor does not alter the data.
    # Check for all labels.
    for label in possible_labels:
        for data, q_data in zip(
            class_encoded_mpos[label], quimb_encoded_mpos[label].tensors
        ):
            assert np.array_equal(data, q_data.data)


    #Check untruncated one site
    one_site_class_encoded_mpos = [
        class_encode_mps_to_mpo(mps_train[0], label, one_site_quimb_hairy_bitstrings)
        for label in possible_labels
    ]
    one_site_quimb_encoded_mpos = [data_to_QTN(mpo) for mpo in one_site_class_encoded_mpos]
    # Check converting to quimb tensor does not alter the data.
    # Check for all labels.
    for label in possible_labels:
        for data, q_data in zip(
            one_site_class_encoded_mpos[label], one_site_quimb_encoded_mpos[label].tensors
        ):
            assert np.array_equal(data, q_data.data)


    #Check truncated multiple sites
    truncated_class_encoded_mpos = [
        class_encode_mps_to_mpo(
            mps_train[0], label, truncated_quimb_hairy_bitstrings
        )
        for label in possible_labels
    ]
    truncated_quimb_encoded_mpos = [
        data_to_QTN(mpo) for mpo in truncated_class_encoded_mpos
    ]
    # Check converting to quimb tensor does not alter the data.
    # Check for all labels.
    for label in possible_labels:
        for data, q_data in zip(
            truncated_class_encoded_mpos[label],
            truncated_quimb_encoded_mpos[label].tensors,
        ):
            assert np.array_equal(data, q_data.data)


    #Check truncated one site
    one_site_truncated_class_encoded_mpos = [
        class_encode_mps_to_mpo(
            mps_train[0], label, truncated_one_site_quimb_hairy_bitstrings
        )
        for label in possible_labels
    ]
    one_site_truncated_quimb_encoded_mpos = [
        data_to_QTN(mpo) for mpo in one_site_truncated_class_encoded_mpos
    ]
    # Check converting to quimb tensor does not alter the data.
    # Check for all labels.
    for label in possible_labels:
        for data, q_data in zip(
            one_site_truncated_class_encoded_mpos[label],
            one_site_truncated_quimb_encoded_mpos[label].tensors,
        ):
            assert np.array_equal(data, q_data.data)


def test_save_and_load_qtn_classifier():
    # Test non-squeezed classifier
    save_qtn_classifier(mpo_classifier, "pytest")
    loaded_classifier = load_qtn_classifier("pytest")

    assert np.array(
        [
            np.array_equal(x.data, y.data)
            for x, y in zip(mpo_classifier.tensors, loaded_classifier.tensors)
        ]
    ).all()

    # Test squeezed classifier loading (not truncated)
    squeezed_classifier = mpo_classifier.squeeze()
    save_qtn_classifier(squeezed_classifier, "pytest_squeezed")
    # loading automatically pads classifier
    loaded_squeezed_classifier = load_qtn_classifier("pytest_squeezed")

    assert np.array(
        [
            np.array_equal(x.data, y.data)
            for x, y in zip(mpo_classifier.tensors, loaded_squeezed_classifier.tensors)
        ]
    ).all()

    # Test squeezed classifier loading (truncated)
    truncated_mpo_classifier = create_mpo_classifier(
        mps_train,truncated_quimb_hairy_bitstrings, seed=420
    ).squeeze()

    save_qtn_classifier(truncated_mpo_classifier, "pytest_squeezed_truncated")
    loaded_squeezed_truncated_classifier = load_qtn_classifier(
        "pytest_squeezed_truncated"
    )

    truncated_mpo_classifier = create_mpo_classifier(mps_train, truncated_quimb_hairy_bitstrings, seed=420)

    assert np.array(
        [
            np.array_equal(x.data, y.data)
            for x, y in zip(
                truncated_mpo_classifier.tensors,
                loaded_squeezed_truncated_classifier.tensors,
            )
        ]
    ).all()

def test_pad_qtn_classifier():
    padded_classifier = pad_qtn_classifier(mpo_classifier)
    D_max = np.max([np.max(tensor.shape, axis = -1) for tensor in mpo_classifier.tensors])

    #Check shape
    for k, (site_a, site_b) in enumerate(zip(mpo_classifier.tensors, padded_classifier.tensors)):
        d1, s1, i1, j1 = site_a.shape
        d2, s2, i2, j2 = site_b.shape

        if k == 0:
            assert(d1 == d2)
            assert(s1 == s2)
            assert(i2 == 1)
            assert(j2 == D_max)
        elif k == (padded_classifier.num_tensors - 1):
            assert(d1 == d2)
            assert(s1 == s2)
            assert(i2 == D_max)
            assert(j2 == 1)
        else:
            assert(d1 == d2)
            assert(s1 == s2)
            assert(i2 == D_max)
            assert(j2 == D_max)

    #Check data is encoded
    for site_a, site_b in zip(mpo_classifier.tensors, padded_classifier.tensors):
        d1, s1, i1, j1 = site_a.shape
        d2, s2, i2, j2 = site_b.shape

        #check data is there
        assert(np.array_equal(site_b.data[:d1,:s1,:i1,:j1], site_a.data))
        #check rest of data is just zeros
        assert(np.array_equal(site_b.data[d1:,s1:,i1:,j1:], np.zeros((d2-d1,s2-s1,i2-i1,j2-j1))))



if __name__ == "__main__":
    # test_bitstring_data_to_QTN()
    #test_save_and_load_qtn_classifier()
    test_pad_qtn_classifier()
