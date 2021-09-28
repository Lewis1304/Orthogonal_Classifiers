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

def test_compress():
    pass

def test_compress_one_site():
    pass

def test_add():
    pass

def test_apply_mpo_from_bottom():
    pass

"""
def test_add_mpos():

    # Check mpos are added in the right place (check for 3, for speed)
    added_mpos = mpo_train[0]
    for mpo in mpo_train[1:3]:
        added_mpos = add_mpos(added_mpos, mpo)

    # Check data of added MPOs (checks shape too).
    for i, (site0, site1, site2, site3) in enumerate(
        zip(
            mpo_train[0].tensors,
            mpo_train[1].tensors,
            mpo_train[2].tensors,
            added_mpos.tensors,
        )
    ):

        if i == 0:
            assert np.array_equal(site0.data, site3.data[:, :, :, : site0.shape[3]])
            assert np.array_equal(
                site1.data,
                site3.data[:, :, :, site0.shape[3] : site0.shape[3] + site1.shape[3]],
            )
            assert np.array_equal(
                site2.data,
                site3.data[
                    :,
                    :,
                    :,
                    site0.shape[3]
                    + site1.shape[3] : site0.shape[3]
                    + site1.shape[3]
                    + site2.shape[3],
                ],
            )

        elif i == (mpo_train[0].num_tensors - 1):
            assert np.array_equal(site0.data, site3.data[:, :, : site0.shape[2], :])
            assert np.array_equal(
                site1.data,
                site3.data[:, :, site0.shape[2] : site0.shape[2] + site1.shape[2], :],
            )
            assert np.array_equal(
                site2.data,
                site3.data[
                    :,
                    :,
                    site0.shape[2]
                    + site1.shape[2] : site0.shape[2]
                    + site1.shape[2]
                    + site2.shape[2],
                    :,
                ],
            )

        else:
            assert np.array_equal(
                site0.data, site3.data[:, :, : site0.shape[2], : site0.shape[3]]
            )
            assert np.array_equal(
                site1.data,
                site3.data[
                    :,
                    :,
                    site0.shape[2] : site0.shape[2] + site1.shape[2],
                    site0.shape[3] : site0.shape[3] + site1.shape[3],
                ],
            )
            assert np.array_equal(
                site2.data,
                site3.data[
                    :,
                    :,
                    site0.shape[2]
                    + site1.shape[2] : site0.shape[2]
                    + site1.shape[2]
                    + site2.shape[2],
                    site0.shape[3]
                    + site1.shape[3] : site0.shape[3]
                    + site1.shape[3]
                    + site2.shape[3],
                ],
            )
"""

"""
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
"""
"""
def test_compress_one_site():

    one_site_mpo_train = mpo_encoding(
        mps_train, y_train, truncated_one_site_quimb_hairy_bitstrings
    )
    one_site_mpo_classifier = create_mpo_classifier(one_site_mpo_train, seed=420)

    one_site_compressed_mpo = compress_QTN(
        one_site_mpo_classifier, D=None, orthogonalise=False
    )

    # Check norm is still 1
    assert np.isclose((one_site_compressed_mpo.H @ one_site_compressed_mpo), 1)

    # Check overlap between initial classifier
    # and compressed mpo with D=None is 1.
    assert np.isclose((one_site_mpo_classifier.H @ one_site_compressed_mpo).norm(), 1)

    orthogonal_one_site_mpo = compress_QTN(
        one_site_mpo_classifier, D=None, orthogonalise=True
    )
    # Check Canonical form and orthogonal
    for k, site in enumerate(orthogonal_one_site_mpo.tensors):
        d, s, i, j = site.data.shape
        U = site.data.transpose(0, 2, 1, 3).reshape(d * i, s * j)
        Uh = U.conj().T
        if k < one_site_compressed_mpo.num_tensors - 1:
            assert np.isclose(Uh @ U, np.eye(s * j)).all()
        else:
            assert np.isclose(U @ Uh, np.eye(d * i)).all()
"""
