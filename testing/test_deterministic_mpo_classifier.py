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

def test_class_encode_mps_to_mpo():
    pass

def test_mpo_encoding():
    pass

def test_add_sublist():
    pass

def test_adding_batches():
    pass

def test_prepare_batched_classifier():
    pass








"""
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
"""
"""
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
"""



#Since one site and multisite are generated differently,
#They will never be equivalent in losses. Therefore use the
#compress function to convert multisite to one site.
#additionally one site to multisite.
#These 2 should be the same. Only works with truncated though.
#This can go in the compress testing

#Truncated. Multisite.
#Truncated. Multisite to one site.

#Truncated. One site.
#Truncated. One site to multisite.
