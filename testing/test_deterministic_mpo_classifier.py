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
from deterministic_mpo_classifier import (
    mpo_encoding,
    class_encode_mps_to_mpo,
    add_sublist,
    adding_batches,
    prepare_batched_classifier,
    unitary_qtn
)

from xmps.ncon import ncon as nc


def ncon(*args, **kwargs):
    return nc(
        *args, check_indices=False, **kwargs
    )  # make default ncon not check indices


n_train = 100
n_test = 100
D_total = 10
batch_num = 10

x_train, y_train, x_test, y_test = load_data(n_train, n_test, equal_numbers=True)

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
truncated_mpo_train = mpo_encoding(mps_train, y_train, truncated_quimb_hairy_bitstrings)

fMPOs = [fMPO([site.data for site in mpo.tensors]) for mpo in truncated_mpo_train]


def test_class_encode_mps_to_mpo():
    # Encode a single image with all different classes.
    mpo_train = [
        class_encode_mps_to_mpo(mps_train[0], label, quimb_hairy_bitstrings)
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

    # Truncated
    truncated_mpo_train = [
        class_encode_mps_to_mpo(mps_train[0], label, truncated_quimb_hairy_bitstrings)
        for label in possible_labels
    ]

    for label in possible_labels:
        for i, (mpo_site, mps_site, bs_site) in enumerate(
            zip(
                truncated_mpo_train[label],
                mps_train[0],
                truncated_quimb_hairy_bitstrings[label],
            )
        ):
            if i < (n_sites - n_hairysites):
                # same as projecting onto |00> state
                assert np.array_equal(mpo_site[:, 0, :, :], mps_site.data)
            else:
                # Projecting onto label bitstring state for that site
                proj = np.einsum("dsij, s...", mpo_site, bs_site.data)
                assert np.array_equal(proj.squeeze(), mps_site.data.squeeze())

    # One Site- untruncated
    untruncated_one_site_mpo_train = [
        class_encode_mps_to_mpo(mps_train[0], label, one_site_quimb_hairy_bitstrings)
        for label in possible_labels
    ]

    for label in possible_labels:
        for i, (mpo_site, mps_site, bs_site) in enumerate(
            zip(
                untruncated_one_site_mpo_train[label],
                mps_train[0],
                one_site_quimb_hairy_bitstrings[label],
            )
        ):
            if i < (n_sites - 1):
                # same as projecting onto |00> state
                assert np.array_equal(mpo_site[:, 0, :, :], mps_site.data)
            else:
                # Projecting onto label bitstring state for that site
                proj = np.einsum("dsij, s...", mpo_site, bs_site.data)
                assert np.array_equal(proj.squeeze(), mps_site.data.squeeze())

    # One Site- truncated
    truncated_one_site_mpo_train = [
        class_encode_mps_to_mpo(
            mps_train[0], label, truncated_one_site_quimb_hairy_bitstrings
        )
        for label in possible_labels
    ]

    for label in possible_labels:
        for i, (mpo_site, mps_site, bs_site) in enumerate(
            zip(
                truncated_one_site_mpo_train[label],
                mps_train[0],
                truncated_one_site_quimb_hairy_bitstrings[label],
            )
        ):
            if i < (n_sites - 1):
                # same as projecting onto |00> state
                assert np.array_equal(mpo_site[:, 0, :, :], mps_site.data)
            else:
                # Projecting onto label bitstring state for that site
                proj = np.einsum("dsij, s...", mpo_site, bs_site.data)
                assert np.array_equal(proj.squeeze(), mps_site.data.squeeze())


def test_mpo_encoding():
    # Check conversion from data to qmpo doesn't affect data
    mpo_train_data = [
        class_encode_mps_to_mpo(mps_train[i], y_train[i], quimb_hairy_bitstrings)
        for i in range(len(mps_train))
    ]

    for TN_data, TN in zip(mpo_train_data, mpo_train):
        for tensor_data, tensor in zip(TN_data, TN.tensors):
            assert np.array_equal(tensor_data, tensor.data)


def test_add_sublist():
    # Only works for truncated images
    # Cannot test if added images without truncation is just block diagonal.
    # Since SVD procedure is always done.

    # Test whether bond rank is correct
    D_max = 10
    sub_list_result = add_sublist((D_max, False), fMPOs)
    assert sub_list_result.D == 10

    # Ensure shape is unchanged
    for k, (added_site, site) in enumerate(zip(sub_list_result.data, fMPOs[0].data)):
        d1, s1, i1, j1 = added_site.shape
        d2, s2, i2, j2 = site.shape

        assert d1 == d1
        assert s1 == s2
        assert i1 <= D_max
        assert j1 <= D_max

        if k == 0:
            assert i1 == 1
        if k == (sub_list_result.L - 1):
            assert j1 == 1


def test_adding_batches():

    # Test for no truncation, that resulting MPO has D=n_train
    batch_added_mpo = adding_batches(fMPOs, None, batch_num)[0]
    assert batch_added_mpo.D == len(fMPOs)

    # Test for truncation, that resulting MPO has D=D_max
    batch_added_mpo = adding_batches(fMPOs, 5, batch_num)[0]
    assert batch_added_mpo.D == 5

    # Check truncation of number of images within adding batches.
    nearest_number_expo = np.log(20) // np.log(batch_num)
    nearest_number = int(batch_num ** nearest_number_expo)

    batch_added_mpo = adding_batches(fMPOs[:20], None, batch_num)[0]
    assert batch_added_mpo.D == nearest_number * fMPOs[0].D


def test_prepare_batched_classifier():

    # Ensure shape is same as mpo images, bar virtual bonds
    # multiple sites
    D_max = 10
    arr_x_train, arr_y_train = arrange_data(x_train, y_train, arrangement="one class")
    arr_mps_train = mps_encoding(arr_x_train, D_max)
    multiple_site_prepared_classifier = prepare_batched_classifier(
        arr_mps_train, arr_y_train, D_max, batch_num, one_site=False
    )

    for k, (prep_site, site) in enumerate(
        zip(multiple_site_prepared_classifier.data, fMPOs[0].data)
    ):
        d1, s1, i1, j1 = prep_site.shape
        d2, s2, i2, j2 = site.shape

        assert d1 == d1
        assert s1 == s2
        assert i1 <= D_max
        assert j1 <= D_max

        if k == 0:
            assert i1 == 1
        if k == (fMPOs[0].L - 1):
            assert j1 == 1

    # one site
    one_site_prepared_classifier = prepare_batched_classifier(
        arr_mps_train, arr_y_train, D_max, batch_num, one_site=True
    )
    one_site_mpo_train = mpo_encoding(
        mps_train, y_train, truncated_one_site_quimb_hairy_bitstrings
    )
    fMPOs_one_site = [
        fMPO([site.data for site in mpo.tensors]) for mpo in one_site_mpo_train
    ]

    for k, (prep_site, site) in enumerate(
        zip(one_site_prepared_classifier.data, fMPOs_one_site[0].data)
    ):
        d1, s1, i1, j1 = prep_site.shape
        d2, s2, i2, j2 = site.shape

        assert d1 == d1
        assert s1 == s2
        assert i1 <= D_max
        assert j1 <= D_max

        if k == 0:
            assert i1 == 1
        if k == (fMPOs_one_site[0].L - 1):
            assert j1 == 1

    classifier_data = multiple_site_prepared_classifier.compress(
        D=D_max, orthogonalise=False
    )
    squeezed_multiple_site_mpo_classifier = data_to_QTN(classifier_data.data).squeeze()

    classifier_data = one_site_prepared_classifier.compress_one_site(
        D=D_max, orthogonalise=False
    )
    squeezed_one_site_mpo_classifier = data_to_QTN(classifier_data.data).squeeze()

    squeezed_truncated_quimb_hairy_bitstrings = [
        i for i in truncated_quimb_hairy_bitstrings
    ]
    squeezed_truncated_one_site_quimb_hairy_bitstrings = [
        i for i in truncated_one_site_quimb_hairy_bitstrings
    ]

    arr_mps_train = mps_encoding(arr_x_train, D_max)
    squeezed_mps_train = [i for i in arr_mps_train]

    multi_site_predictions = classifier_predictions(
        squeezed_multiple_site_mpo_classifier,
        squeezed_mps_train,
        squeezed_truncated_quimb_hairy_bitstrings,
    )
    one_site_predictions = classifier_predictions(
        squeezed_one_site_mpo_classifier,
        squeezed_mps_train,
        squeezed_truncated_one_site_quimb_hairy_bitstrings,
    )

    multi_site_result = evaluate_classifier_top_k_accuracy(
        multi_site_predictions, arr_y_train, 1
    )
    one_site_result = evaluate_classifier_top_k_accuracy(
        one_site_predictions, arr_y_train, 1
    )

    # Check performance is above random.
    assert multi_site_result > 0.1
    assert one_site_result > 0.1
    # Check multisite performance is better than onesite (for adding)
    assert multi_site_result >= one_site_result

    # Check orthogonal is worse than non-orthogonal
    classifier_data = multiple_site_prepared_classifier.compress(
        D=D_max, orthogonalise=True
    )
    ortho_squeezed_multiple_site_mpo_classifier = data_to_QTN(
        classifier_data.data
    ).squeeze()

    classifier_data = one_site_prepared_classifier.compress_one_site(
        D=D_max, orthogonalise=True
    )
    ortho_squeezed_one_site_mpo_classifier = data_to_QTN(classifier_data.data).squeeze()

    ortho_multi_site_predictions = classifier_predictions(
        ortho_squeezed_multiple_site_mpo_classifier,
        squeezed_mps_train,
        squeezed_truncated_quimb_hairy_bitstrings,
    )
    ortho_one_site_predictions = classifier_predictions(
        ortho_squeezed_one_site_mpo_classifier,
        squeezed_mps_train,
        squeezed_truncated_one_site_quimb_hairy_bitstrings,
    )

    ortho_multi_site_result = evaluate_classifier_top_k_accuracy(
        multi_site_predictions, arr_y_train, 1
    )
    ortho_one_site_result = evaluate_classifier_top_k_accuracy(
        one_site_predictions, arr_y_train, 1
    )

    assert multi_site_result >= ortho_multi_site_result
    assert one_site_result >= ortho_one_site_result

def test_unitary_qtn():
    #Test is only for prepared batch classifiers for now..
    mps_train = mps_encoding(x_train, 32)

    fmpo_classifier = prepare_batched_classifier(
            mps_train, y_train, 32, 10, one_site=False
        ).compress_one_site(D = None, orthogonalise = True)
    qtn_classifier = data_to_QTN(fmpo_classifier.data)

    uclassifier = unitary_qtn(qtn_classifier)
    #test that everysite is unitary
    for tensor in uclassifier.tensors:
        site = tensor.data
        d,s,i,j = site.shape
        site = site.transpose(0,2,1,3).reshape(d * i, s * j)


        assert np.isclose((site.conj().T @ site ),np.eye(s*j)).all()
        assert np.isclose((site @ site.conj().T),np.eye(d*i)).all()



if __name__ == "__main__":
    test_unitary_qtn()

    # Since one site and multisite are generated differently,
    # They will never be equivalent in losses. Therefore use the
    # compress function to convert multisite to one site.
    # additionally one site to multisite.
    # These 2 should be the same. Only works with truncated though.
    # This can go in the compress testing

    # Truncated. Multisite.
    # Truncated. Multisite to one site.

    # Truncated. One site.
    # Truncated. One site to multisite.
