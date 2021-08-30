import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np

import quimb
import quimb.tensor as qtn
from quimb.tensor.tensor_core import rand_uuid

from oset import oset

from xmps.fMPS import fMPS
from fMPO_reduced import fMPO


"""
Data tools
"""


def load_data(n_train, n_test=1):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = (x_train.reshape(len(x_train), -1) / 255)[:n_train], (
        x_test.reshape(len(x_test), -1) / 255
    )[:n_test]
    y_train, y_test = y_train[:n_train], y_test[:n_test]
    return x_train, y_train, x_test, y_test


"""
Bitstring tools
"""


def create_bitstrings(possible_labels, n_hairysites):
    return [bin(label)[2:].zfill(n_hairysites * 2) for label in possible_labels]


def bitstring_to_product_state_data(bitstring_data):
    return fMPS().from_product_state(bitstring_data).data


def bitstring_data_to_QTN(data, n_hairysites, n_sites, truncated=False):
    # Doesn't work for truncated_data
    prod_state_data = bitstring_to_product_state_data(data)
    if truncated:
        prod_state_data = [
            [
                site[:1] if i < (n_sites - n_hairysites) else site
                for i, site in enumerate(l)
            ]
            for l in prod_state_data
        ]

    q_product_states = []
    for prod_state in prod_state_data:
        qtn_data = []
        previous_ind = rand_uuid()
        for j, site in enumerate(prod_state):
            next_ind = rand_uuid()
            tensor = qtn.Tensor(
                site, inds=(f"s{j}", previous_ind, next_ind), tags=oset([f"{j}"])
            )
            previous_ind = next_ind
            qtn_data.append(tensor)
        q_product_states.append(qtn.TensorNetwork(qtn_data))
    return q_product_states


"""
MPS encoding tools
"""


def image_to_mps(f_image, D):
    num_pixels = f_image.shape[0]
    L = int(np.ceil(np.log2(num_pixels)))
    padded_image = np.pad(f_image, (0, 2 ** L - num_pixels)).reshape(*[2] * L)
    return fMPS().left_from_state(padded_image).left_canonicalise(D=D)


def fMPS_to_QTN(fmps):
    qtn_data = []
    previous_ind = rand_uuid()
    for j, site in enumerate(fmps.data):
        next_ind = rand_uuid()
        tensor = qtn.Tensor(
            site, inds=(f"k{j}", previous_ind, next_ind), tags=oset([f"{j}"])
        )
        previous_ind = next_ind
        qtn_data.append(tensor)
    return qtn.TensorNetwork(qtn_data)


"""
MPO encoding tools
"""


def data_to_QTN(data):
    qtn_data = []
    previous_ind = rand_uuid()
    for j, site in enumerate(data):
        next_ind = rand_uuid()
        tensor = qtn.Tensor(
            site, inds=(f"d{j}", f"s{j}", previous_ind, next_ind), tags=oset([f"{j}"])
        )
        previous_ind = next_ind
        qtn_data.append(tensor)
    return qtn.TensorNetwork(qtn_data)


"""
Creating Initialised Classifier tools
"""


def add_mpos(a, b):
    # Add quimb MPOs together. Assumes physical indicies are
    # of same dimension

    # Check tensors are of same length
    assert a.num_tensors == b.num_tensors

    a_data = [site.data for site in a.tensors]
    b_data = [site.data for site in b.tensors]

    new_data = [
        1j
        * np.zeros(
            (
                a_site.shape[0],
                a_site.shape[1],
                a_site.shape[2] + b_site.shape[2],
                a_site.shape[3] + b_site.shape[3],
            )
        )
        for a_site, b_site in zip(a_data, b_data)
    ]

    for i, (a_site, b_site) in enumerate(zip(a_data, b_data)):
        if i == 0:
            new_data[i] = np.concatenate([a_site, b_site], 3)
        elif i == a.num_tensors - 1:
            new_data[i] = np.concatenate([a_site, b_site], 2)
        else:
            new_data[i][:, :, : a_site.shape[2], : a_site.shape[3]] = a_site
            new_data[i][:, :, a_site.shape[2] :, a_site.shape[3] :] = b_site
    return data_to_QTN(new_data)


def adding_sublist(sublist):
    # Add all MPOs in a list
    added_batch = sublist[0]
    for mpo in sublist[1:]:
        added_batch = add_mpos(added_batch, mpo)
    return added_batch


def QTN_to_fMPO(QTN):
    qtn_data = [site.data for site in QTN.tensors]
    return fMPO(qtn_data)


def fMPO_to_QTN(fmpo):
    fmpo_data = fmpo.data
    return data_to_QTN(fmpo_data)


if __name__ == "__main__":
    pass
