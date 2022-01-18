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
from fMPO_reduced import fMPO

from xmps.ncon import ncon as nc


def ncon(*args, **kwargs):
    return nc(
        *args, check_indices=False, **kwargs
    )  # make default ncon not check indices


n_train = 100
n_test = 100
D_total = 10

x_train, y_train, x_test, y_test = load_data(n_train, n_test, equal_numbers=True)

possible_labels = list(set(y_train))
n_sites = int(np.ceil(math.log(x_train.shape[-1], 2)))
n_pixels = len(x_train[0])

hairy_bitstrings_data_untruncated_data = create_hairy_bitstrings_data(
    possible_labels, n_sites
)
one_site_bitstrings_data_untruncated_data = create_hairy_bitstrings_data(
    possible_labels, n_sites
)

truncated_quimb_hairy_bitstrings = bitstring_data_to_QTN(
    hairy_bitstrings_data_untruncated_data, n_sites, truncated=True
)

truncated_one_site_quimb_hairy_bitstrings = bitstring_data_to_QTN(
    one_site_bitstrings_data_untruncated_data, n_sites, truncated=True
)

mps_train = mps_encoding(x_train, D_total)

multiple_site_mpo_classifier = create_mpo_classifier(
    mps_train, truncated_quimb_hairy_bitstrings, seed=420
)
multiple_site_fMPO = fMPO([i.data for i in multiple_site_mpo_classifier.tensors])

one_site_mpo_classifier = create_mpo_classifier(
    mps_train, truncated_one_site_quimb_hairy_bitstrings, seed=420
)
one_site_fMPO = fMPO([i.data for i in one_site_mpo_classifier.tensors])

mpo_train = mpo_encoding(mps_train, y_train, truncated_quimb_hairy_bitstrings)
fMPOs = [fMPO([site.data for site in mpo.tensors]) for mpo in mpo_train]

# TODO: Implement orthogonalisation test
def test_compress_one_site():

    # Check norm is still 1
    # One site
    one_site_fMPO = fMPO([i.data for i in one_site_mpo_classifier.tensors])
    compressed_one_site_mpo = one_site_fMPO.compress_one_site(
        D=None, orthogonalise=False
    )
    compressed_qtn_mpo = data_to_QTN(compressed_one_site_mpo.data)
    assert np.isclose(
        abs(compressed_qtn_mpo.squeeze().H @ compressed_qtn_mpo.squeeze()), 1
    )

    # Check overlap between initial classifier
    # and compressed mpo with D=None is 1.
    one_site_fMPO = fMPO([i.data for i in one_site_mpo_classifier.tensors])
    compressed_one_site_mpo = one_site_fMPO.compress_one_site(
        D=None, orthogonalise=False
    )
    compressed_qtn_mpo = data_to_QTN(compressed_one_site_mpo.data)

    one_site_fMPO = fMPO([i.data for i in one_site_mpo_classifier.tensors])
    qtn_mpo = data_to_QTN(one_site_fMPO.data)
    assert np.isclose(abs(qtn_mpo.squeeze().H @ compressed_qtn_mpo.squeeze()), 1)

    # Check each site is unitary. i.e. in canonical form:
    # U @ U.H = I
    compressed_one_site_mpo = one_site_fMPO.compress_one_site(
        D=None, orthogonalise=False, sweep_back = True
    )

    for n, site in enumerate(compressed_one_site_mpo.data):
        d, s, i, j = site.shape
        # reshape from (d, s, i, j) --> (i, s*i*d).
        # As SVD on the return sweep was done like that.
        # U = site.data.transpose(0, 3, 1, 2).reshape(d * j, s * i)
        U = site.transpose(2, 1, 3, 0).reshape(i, s * j * d)
        Uh = U.conj().T
        assert np.array_equal(np.round(U @ Uh, 5), np.eye(i))

    # Check compressed has right shape for range of different Ds
    for max_D in range(1, 5):
        one_site_fMPO = fMPO([i.data for i in one_site_mpo_classifier.tensors])
        compressed_one_site_mpo = one_site_fMPO.compress_one_site(
            D=max_D, orthogonalise=False
        )

        for i, (site0, site1) in enumerate(zip(compressed_one_site_mpo, one_site_fMPO)):

            d0, s0, i0, j0 = site0.shape
            d1, s1, i1, j1 = site1.shape

            assert d0 == d1
            assert s0 == s1

            assert i0 <= max_D
            assert j0 <= max_D

    # Check using compress_one_site on multisite mpo outputs one site mpo
    multiple_site_fMPO = fMPO([i.data for i in multiple_site_mpo_classifier.tensors])
    compressed_multiple_site_mpo = multiple_site_fMPO.compress_one_site(
        D=None, orthogonalise=False
    )

    for k, (site1, site2) in enumerate(
        zip(compressed_multiple_site_mpo.data, compressed_one_site_mpo.data)
    ):
        d1, s1, i1, j1 = site1.shape
        d2, s2, i2, j2 = site2.shape

        assert d1 == d2
        assert s1 == s2

    # Check overlap is for multi site and converted one site.
    qtn_mpo = data_to_QTN(compressed_multiple_site_mpo.data)
    overlap1 = mps_train[0].squeeze() @ (
        qtn_mpo.squeeze() @ truncated_one_site_quimb_hairy_bitstrings[0].squeeze()
    )
    overlap2 = mps_train[0].squeeze() @ (
        multiple_site_mpo_classifier.squeeze()
        @ truncated_quimb_hairy_bitstrings[0].squeeze()
    )
    assert np.isclose(abs(overlap1), abs(overlap2))


def test_add():

    # Check mpos are added in the right place (check for 3, for speed)
    added_mpos = fMPOs[0]

    for mpo in fMPOs[1:3]:
        added_mpos = added_mpos.add(mpo)

    # Check data of added MPOs (checks shape too).
    for i, (site0, site1, site2, site3) in enumerate(
        zip(
            fMPOs[0].data,
            fMPOs[1].data,
            fMPOs[2].data,
            added_mpos.data,
        )
    ):

        if i == 0:
            assert np.array_equal(site0, site3[:, :, :, : site0.shape[3]])
            assert np.array_equal(
                site1,
                site3[:, :, :, site0.shape[3] : site0.shape[3] + site1.shape[3]],
            )
            assert np.array_equal(
                site2,
                site3[
                    :,
                    :,
                    :,
                    site0.shape[3]
                    + site1.shape[3] : site0.shape[3]
                    + site1.shape[3]
                    + site2.shape[3],
                ],
            )

        elif i == (fMPOs[0].L - 1):
            assert np.array_equal(site0, site3[:, :, : site0.shape[2], :])
            assert np.array_equal(
                site1,
                site3[:, :, site0.shape[2] : site0.shape[2] + site1.shape[2], :],
            )
            assert np.array_equal(
                site2,
                site3[
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
                site0, site3[:, :, : site0.shape[2], : site0.shape[3]]
            )
            assert np.array_equal(
                site1,
                site3[
                    :,
                    :,
                    site0.shape[2] : site0.shape[2] + site1.shape[2],
                    site0.shape[3] : site0.shape[3] + site1.shape[3],
                ],
            )
            assert np.array_equal(
                site2,
                site3[
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


if __name__ == "__main__":
    test_compress_one_site()
