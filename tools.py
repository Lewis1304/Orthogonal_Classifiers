import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import math

import quimb
import quimb.tensor as qtn
from quimb.tensor.tensor_core import rand_uuid

from oset import oset
from scipy.linalg import null_space

from xmps.fMPS import fMPS
from fMPO_reduced import fMPO


"""
Data tools
"""


def load_data(n_train, n_test=10, shuffle=False, equal_numbers=False):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if shuffle:
        r_train = np.arange(len(x_train))
        r_test = np.arange(len(x_test))
        np.random.shuffle(r_train)
        np.random.shuffle(r_test)

        x_train = x_train[r_train]
        y_train = y_train[r_train]

        x_test = x_test[r_test]
        y_test = y_test[r_test]

    if equal_numbers:
        n_train_per_class = n_train // len(list(set(y_train)))
        #n_test_per_class = n_test // len(list(set(y_train)))

        grouped_x_train = [
            x_train[y_train == label][:n_train_per_class]
            for label in list(set(y_train))
        ]
        #grouped_x_test = [
        #    x_test[y_test == label][:n_test_per_class] for label in list(set(y_test))
        #]

        train_data = np.array([images for images in zip(*grouped_x_train)])
        train_labels = [list(set(y_train)) for _ in train_data]
        #test_data = np.array([images for images in zip(*grouped_x_test)])
        #test_labels = [list(set(y_test)) for _ in range(len(test_data))]

        x_train = np.array([item for sublist in train_data for item in sublist])
        y_train = np.array([item for sublist in train_labels for item in sublist])
        #x_test = np.array([item for sublist in test_data for item in sublist])
        #y_test = np.array([item for sublist in test_labels for item in sublist])

    x_train, x_test = (x_train.reshape(len(x_train), -1) / 255)[:n_train], (
        x_test.reshape(len(x_test), -1) / 255
    )[:n_test]
    y_train, y_test = y_train[:n_train], y_test[:n_test]

    return x_train, y_train, x_test, y_test


def arrange_data(data, labels, **kwargs):

    if list(kwargs.values())[0] == "random":
        r_train = np.arange(len(data))
        np.random.shuffle(r_train)
        return data[r_train], labels[r_train]

    elif list(kwargs.values())[0] == "one of each":
        return data, labels

    elif list(kwargs.values())[0] == "one class":
        possible_labels = list(set(labels))
        data = [
            [data[i] for i in range(k, len(data), len(possible_labels))]
            for k in possible_labels
        ]
        data = np.array([image for label in data for image in label])

        labels = [
            [labels[i] for i in range(k, len(labels), len(possible_labels))]
            for k in possible_labels
        ]
        labels = np.array([image for label in labels for image in label])
        return data, labels
    else:
        raise Exception("Arrangement type not understood")

def shuffle_arranged_data(data, labels):
    #Assumes data is correctly arranged
    #And equal amounts of data

    num_class = len(labels) // len(list(set(labels)))
    shuffled_data = []
    shuffled_labels = []

    for i in range(len(data) // num_class):

        #Data of all the same class
        sub_data = data[i*num_class:(i+1)*num_class]
        sub_labels = labels[i*num_class:(i+1)*num_class]

        shuff = np.arange(len(sub_labels))
        np.random.shuffle(shuff)

        sub_data = np.array(sub_data)[shuff]
        sub_labels = np.array(sub_labels)[shuff]

        shuffled_data.append(sub_data)
        shuffled_labels.append(sub_labels)

    shuffled_data = np.array([item for sublist in shuffled_data for item in sublist])
    shuffled_labels = np.array([item for sublist in shuffled_labels for item in sublist])

    return shuffled_data, shuffled_labels





"""
Bitstring tools
"""


def create_bitstrings(possible_labels, n_hairysites = 1):
    return [bin(label)[2:].zfill(n_hairysites * 2) for label in possible_labels]


def bitstring_to_product_state_data(bitstring_data):
    return fMPS().from_product_state(bitstring_data).data


def bitstring_data_to_QTN(data, n_sites, truncated=False):
    # Doesn't work for truncated_data
    prod_state_data = bitstring_to_product_state_data(data)
    if truncated:

        # Check to see whether state is one-site hairy
        site = prod_state_data[0][-1]
        # If true: state is one-site hairy
        if math.log(site.shape[0], 4) > 1:
            n_hairysites = 1

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


def padded_bitstring_data_to_QTN(data, uclassifier):
    prod_state_data = [bitstring_to_product_state_data(i) for i in data]

    prod_state_data = [[
        [
            site1[:site2.shape[1]]
            for site1, site2 in zip(l, uclassifier.tensors)
        ]
        for l in padding
    ]
    for padding in prod_state_data]

    q_product_states = []
    for padding in prod_state_data:
        paddings = []
        for prod_state in padding:
            qtn_data = []
            previous_ind = rand_uuid()
            for j, site in enumerate(prod_state):
                next_ind = rand_uuid()
                tensor = qtn.Tensor(
                    site, inds=(f"s{j}", previous_ind, next_ind), tags=oset([f"{j}"])
                )
                previous_ind = next_ind
                qtn_data.append(tensor)
            paddings.append(qtn.TensorNetwork(qtn_data))
        q_product_states.append(paddings)
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
            site, inds=(f"k{j}", f"s{j}", previous_ind, next_ind), tags=oset([f"{j}"])
        )
        previous_ind = next_ind
        qtn_data.append(tensor)
    return qtn.TensorNetwork(qtn_data)


"""
Classifier tools
"""


def save_qtn_classifier(QTN, dir):
    if not os.path.exists("Classifiers/" + dir):
        os.makedirs("Classifiers/" + dir)
    for i, site in enumerate(QTN.tensors):
        np.save("Classifiers/" + dir + f"/site_{i}", site.data)


def load_qtn_classifier(dir):
    files = os.listdir("Classifiers/" + dir)
    num_files = len(files)

    data = []

    for i in range(num_files):

        site = np.load("Classifiers/" + dir + f"/site_{i}.npy", allow_pickle=True)
        if i == 0:
            if len(site.shape) == 2:
                site = np.expand_dims(np.expand_dims(site, 1), 1)
            if len(site.shape) == 3:
                site = np.expand_dims(site, 2)
        elif i == num_files - 1:
            if len(site.shape) == 3:
                site = np.expand_dims(site, -1)
        else:
            if len(site.shape) != 4:
                site = np.expand_dims(site, 1)
        data.append(site)

    return data_to_QTN(data)


def pad_qtn_classifier(QTN):
    D_max = np.max([np.max(tensor.shape, axis=-1) for tensor in QTN.tensors])
    qtn_data = [site.data for site in QTN.tensors]

    data_padded = []
    for k, site in enumerate(qtn_data):
        d, s, i, j = site.shape

        if k == 0:
            site_padded = np.pad(site, ((0, 0), (0, 0), (0, 0), (0, D_max - j)))
        elif k == (len(qtn_data) - 1):
            site_padded = np.pad(site, ((0, 0), (0, 0), (0, D_max - i), (0, 0)))
        else:
            site_padded = np.pad(site, ((0, 0), (0, 0), (0, D_max - i), (0, D_max - j)))

        data_padded.append(site_padded)

    return data_to_QTN(data_padded)


if __name__ == "__main__":
    pass
