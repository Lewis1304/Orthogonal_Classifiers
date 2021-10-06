import numpy as np
from functools import reduce
from xmps.fMPS import fMPS
from tqdm import tqdm
from fMPO_reduced import fMPO
from tools import load_data, data_to_QTN, arrange_data
from variational_mpo_classifiers import evaluate_classifier_top_k_accuracy, mps_encoding, create_hairy_bitstrings_data, bitstring_data_to_QTN, squeezed_classifier_predictions
from math import log as mlog

"""
MPO Encoding
"""

def class_encode_mps_to_mpo(mps, label, q_hairy_bitstrings):
    n_sites = mps.num_tensors
    mps_data = [tensor.data for tensor in mps.tensors]
    # class_encoded_mps.shape = #pixels, dim(d), d(s), i, j
    class_encoded_mps = [
        np.array(
            [mps_data[site] * i for i in q_hairy_bitstrings[label].tensors[site].data]
        ).transpose(1, 0, 2, 3)
        for site in range(n_sites)
    ]
    return class_encoded_mps


def mpo_encoding(mps_train, y_train, q_hairy_bitstrings):
    n_samples = len(mps_train)
    mpo_train = [
        data_to_QTN(
            class_encode_mps_to_mpo(
                mps_train[i], y_train[i], q_hairy_bitstrings
            )
        )
        for i in range(n_samples)
    ]
    return mpo_train


"""
Adding Images
"""

def add_sublist(*args):
    """
    :param args: tuple of B_D and MPOs to be added together
    """

    B_D = args[0][0]
    ortho = args[0][1]
    sub_list_mpos = args[1]
    N = len(sub_list_mpos)

    c = sub_list_mpos[0]

    for i in range(1,N):
        c = c.add(sub_list_mpos[i])
    if c.data[-2].shape[1] == 1:
        return c.compress_one_site(B_D, orthogonalise=ortho)
    return c.compress(B_D, orthogonalise=ortho)

def adding_batches(list_to_add,D,batch_num=2,truncate=True, orthogonalise = False):
    # if batches are not of equal size, the remainder is added
    # at the end- this is a MAJOR problem with equal weightings!

    if len(list_to_add) % batch_num != 0:
        if not truncate:
            raise Exception('Batches are not of equal size!')
        else:
            trun_expo = int(np.log(len(list_to_add)) / np.log(batch_num))
            list_to_add = list_to_add[:batch_num**trun_expo]
    result = []

    for i in range(int(len(list_to_add)/batch_num)+1):
        sub_list = list_to_add[batch_num*i:batch_num*i+batch_num]
        if len(sub_list) > 0:
            result.append(reduce(add_sublist,((D, orthogonalise),sub_list)))
    return result

"""
Prepare classifier
"""

def prepare_batched_classifier(train_data, train_labels, D_total, batch_num, one_site = False):

    possible_labels = list(set(train_labels))
    n_hairy_sites = int(np.ceil(mlog(len(possible_labels), 4)))
    n_sites = int(np.ceil(mlog(train_data.shape[-1], 2)))

    #Encoding images as MPOs. The structure of the MPOs might be different
    #To the variational MPO structure. This requires creating bitstrings
    #again as well
    mps_train = mps_encoding(train_data, D_total)
    hairy_bitstrings_data = create_hairy_bitstrings_data(
        possible_labels, n_hairy_sites, n_sites, one_site
    )
    q_hairy_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, n_hairy_sites, n_sites, truncated=True
    )
    train_mpos = mpo_encoding(mps_train, train_labels, q_hairy_bitstrings)

    #Converting qMPOs into fMPOs
    MPOs = [fMPO([site.data for site in mpo.tensors]) for mpo in train_mpos]

    #Adding fMPOs together
    while len(MPOs) > 1:
        MPOs = adding_batches(MPOs, D_total, batch_num)

    return MPOs[0]

"""
Misc.
"""

def batch_initialise_classifier():
    D_total = 32
    n_train = 1000
    one_site = False
    ortho_at_end = False
    batch_num = 10

    #Load Data- ensuring particular order to be batched added
    x_train, y_train, x_test, y_test = load_data(n_train, shuffle = False, equal_numbers = True)
    train_data, train_labels = arrange_data(x_train, y_train, arrangement = 'one class')

    #Add images together- forming classifier initialisation
    fMPO_classifier = prepare_batched_classifier(train_data, train_labels, D_total, batch_num, one_site = one_site)
    qtn_classifier = data_to_QTN(fMPO_classifier.data).squeeze()
    qtn_classifier_data = fMPO_classifier.compress_one_site(D=D_total, orthogonalise=ortho_at_end)
    qtn_classifier = data_to_QTN(qtn_classifier_data.data).squeeze()
    
    #Evaluating Classifier
    n_hairy_sites = 1
    n_sites = 10
    one_site = True
    possible_labels = list(set(train_labels))

    mps_train = mps_encoding(train_data, D_total)
    mps_train = [i.squeeze() for i in mps_train]
    hairy_bitstrings_data = create_hairy_bitstrings_data(
        possible_labels, n_hairy_sites, n_sites, one_site
    )
    q_hairy_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, n_hairy_sites, n_sites, one_site
    )
    q_hairy_bitstrings = [i.squeeze() for i in q_hairy_bitstrings]

    predictions = squeezed_classifier_predictions(qtn_classifier, mps_train, q_hairy_bitstrings)
    print(evaluate_classifier_top_k_accuracy(predictions, train_labels, 1))


if __name__ == '__main__':
    #sequential_mpo_classifier_experiment()
    batch_initialise_classifier()
