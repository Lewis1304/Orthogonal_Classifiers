import numpy as np
from functools import reduce
from xmps.fMPS import fMPS
from tqdm import tqdm
from fMPO_reduced import fMPO
from tools import load_data, data_to_QTN, arrange_data, shuffle_arranged_data
from variational_mpo_classifiers import (
    evaluate_classifier_top_k_accuracy,
    mps_encoding,
    create_hairy_bitstrings_data,
    bitstring_data_to_QTN,
    classifier_predictions,
)
from math import log as mlog
from scipy.linalg import null_space
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
            class_encode_mps_to_mpo(mps_train[i], y_train[i], q_hairy_bitstrings)
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

    for i in range(1, N):
        c = c.add(sub_list_mpos[i])
    if c.data[-2].shape[1] == 1:
        return c.compress_one_site(B_D, orthogonalise=ortho)
    return c.compress(B_D, orthogonalise=ortho)


def adding_batches(list_to_add, D, batch_num=2, truncate=True, orthogonalise=False):
    # if batches are not of equal size, the remainder is added
    # at the end- this is a MAJOR problem with equal weightings!

    if len(list_to_add) % batch_num != 0:
        if not truncate:
            raise Exception("Batches are not of equal size!")
        else:
            trun_expo = int(np.log(len(list_to_add)) / np.log(batch_num))
            list_to_add = list_to_add[: batch_num ** trun_expo]
    result = []

    for i in range(int(len(list_to_add) / batch_num) + 1):
        sub_list = list_to_add[batch_num * i : batch_num * (i + 1)]
        if len(sub_list) > 0:
            result.append(reduce(add_sublist, ((D, orthogonalise), sub_list)))
    return result


"""
Prepare classifier
"""


def prepare_batched_classifier(
    mps_train, labels, D_total, batch_num, one_site=False
):

    possible_labels = list(set(labels))
    n_hairy_sites = int(np.ceil(mlog(len(possible_labels), 4)))
    n_sites = mps_train[0].num_tensors

    #Bitstrings have to be non-truncated

    hairy_bitstrings_data = create_hairy_bitstrings_data(
        possible_labels, n_hairy_sites, n_sites, one_site
    )
    q_hairy_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, n_hairy_sites, n_sites, truncated=True
    )
    train_mpos = mpo_encoding(mps_train, labels, q_hairy_bitstrings)

    # Converting qMPOs into fMPOs
    MPOs = [fMPO([site.data for site in mpo.tensors]) for mpo in train_mpos]

    # Adding fMPOs together
    while len(MPOs) > 1:
        MPOs = adding_batches(MPOs, D_total, batch_num)

    return MPOs[0]

def prepare_sum_states(
    mps_train, labels, D_total, batch_num, one_site=False
):

    possible_labels = list(set(labels))
    n_hairy_sites = int(np.ceil(mlog(len(possible_labels), 4)))
    n_sites = mps_train[0].num_tensors

    #Bitstrings have to be non-truncated

    hairy_bitstrings_data = create_hairy_bitstrings_data(
        possible_labels, n_hairy_sites, n_sites, one_site
    )
    q_hairy_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, n_hairy_sites, n_sites, truncated=True
    )
    train_mpos = mpo_encoding(mps_train, labels, q_hairy_bitstrings)

    # Converting qMPOs into fMPOs
    MPOs = [fMPO([site.data for site in mpo.tensors]) for mpo in train_mpos]

    # Adding fMPOs together
    while len(MPOs) > batch_num:
        MPOs = adding_batches(MPOs, D_total, batch_num)

    return MPOs



"""
Ensemble classifiers
"""


def prepare_ensemble(*args, **kwargs):
    """
    param: args : Arguments for data
    param: kwargs : Keyword arguments for hyperparameters
    """
    #assumes training data is loaded same way for evaluating.
    # TODO: Change prepare_batched_classifier ot accept mps_images
    n_classifiers = args[0]
    mps_train = args[1]
    labels = args[2]
    D_total = kwargs['D_total']
    batch_num = kwargs['batch_num']

    classifiers = []
    for i in tqdm(range(n_classifiers)):
        mps_train, labels = shuffle_arranged_data(mps_train, labels)

        fmpo_classifier = prepare_batched_classifier(
            mps_train, labels, D_total, batch_num, one_site=False
        )
        classifier_data = fmpo_classifier.compress_one_site(
            D=D_total, orthogonalise=False
        )
        qmpo_classifier = data_to_QTN(classifier_data.data).squeeze()

        classifiers.append(qmpo_classifier)

    return classifiers



"""
Misc.
"""


def batch_initialise_classifier():
    D_total = 32
    n_train = 1000
    one_site = False
    ortho_at_end = False
    batch_num = 10

    # Load Data- ensuring particular order to be batched added
    x_train, y_train, x_test, y_test = load_data(
        n_train, shuffle=False, equal_numbers=True
    )
    train_data, train_labels = arrange_data(x_train, y_train, arrangement="one class")

    # Add images together- forming classifier initialisation
    fMPO_classifier = prepare_batched_classifier(
        train_data, train_labels, D_total, batch_num, one_site=one_site
    )
    qtn_classifier = data_to_QTN(fMPO_classifier.data).squeeze()
    qtn_classifier_data = fMPO_classifier.compress_one_site(
        D=D_total, orthogonalise=ortho_at_end
    )
    qtn_classifier = data_to_QTN(qtn_classifier_data.data).squeeze()

    # Evaluating Classifier
    n_hairy_sites = 1
    n_sites = 10
    one_site = True
    possible_labels = list(set(train_labels))

    mps_train = mps_encoding(train_data, D_total)
    hairy_bitstrings_data = create_hairy_bitstrings_data(
        possible_labels, n_hairy_sites, n_sites, one_site
    )
    q_hairy_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, n_hairy_sites, n_sites, one_site
    )

    predictions = classifier_predictions(qtn_classifier, mps_train, q_hairy_bitstrings)
    print(evaluate_classifier_top_k_accuracy(predictions, train_labels, 1))

def unitary_qtn(qtn):
    #Only works for powers of bond dimensions which are (due to reshaping of tensors)
    D_max = max([tensor.shape[-1] for tensor in qtn.tensors])
    if not mlog(D_max,2).is_integer():
        raise Exception('Classifier has to have bond order of power 2!')
    def unitary_extension(Q):

        def direct_sum(A, B):
            '''direct sum of two matrices'''
            (a1, a2), (b1, b2) = A.shape, B.shape
            O = np.zeros((a2, b1))
            return np.block([[A, O], [O.T, B]])

        '''extend an isometry to a unitary (doesn't check its an isometry)'''
        s = Q.shape
        flipped=False
        N1 = null_space(Q)
        N2 = null_space(Q.conj().T)


        if s[0]>s[1]:
            Q_ = np.concatenate([Q, N2], 1)
        elif s[0]<s[1]:
            Q_ = np.concatenate([Q.conj().T, N1], 1).conj().T
        else:
            Q_ = Q
        return Q_


    data = []
    for tensor in qtn.tensors:
        site = tensor.data
        d, s, i, j = site.shape

        site = site.transpose(0, 2, 1, 3).reshape(d * i, s * j)
        if not np.isclose(site @ site.conj().T, np.eye(d*i)).all() or not np.isclose(site.conj().T @ site, np.eye(s*j)).all():

            usite = unitary_extension(site)
            #print(usite.conj().T @ usite)

            #assert np.isclose(usite.conj().T @ usite, np.eye(usite.shape[0])).all()
            #assert np.isclose(usite @ usite.conj().T, np.eye(usite.shape[0])).all()
            usite = usite.reshape(d, i, -1, j).transpose(0, 2, 1, 3)
            data.append(usite)
        else:
            data.append(site.reshape(d, i, -1, j).transpose(0, 2, 1, 3))

    uclassifier = data_to_QTN(data)
    return uclassifier

if __name__ == "__main__":
    pass
    # sequential_mpo_classifier_experiment()
    #batch_initialise_classifier()
