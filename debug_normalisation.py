from tools import load_data, arrange_data, data_to_QTN
from variational_mpo_classifiers import create_hairy_bitstrings_data, bitstring_data_to_QTN, mps_encoding, evaluate_classifier_top_k_accuracy
from deterministic_mpo_classifier import mpo_encoding, prepare_batched_classifier, adding_batches
from fMPO_reduced import fMPO

from tqdm import tqdm
import numpy as np
import math
from functools import reduce


def label_last_site_to_centre(qtn):
    data = [site.data for site in qtn.tensors]
    centre_index = len(data) // 2
    data[-1], data[centre_index] = data[centre_index], data[-1]
    return data

def centred_bitstring_to_qtn(bitstrings_data):
    from quimb.tensor.tensor_core import rand_uuid
    import quimb.tensor as qtn
    from oset import oset

    q_product_states = []
    for prod_state in bitstrings_data:
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

def prepare_centred_batched_classifier(mps_train, labels, q_hairy_bitstrings, D_total, batch_nums):

    train_mpos = mpo_encoding(mps_train, labels, q_hairy_bitstrings)

    # Converting qMPOs into fMPOs
    MPOs = [fMPO([site.data for site in mpo.tensors]) for mpo in train_mpos]

    # Adding fMPOs together
    i = 0
    while len(MPOs) > 10:
        batch_num = batch_nums[i]
        MPOs = adding_centre_batches(MPOs, D_total, batch_num)
        i += 1
    return MPOs

def adding_centre_batches(list_to_add, D, batch_num=2, orthogonalise=False):
    # if batches are not of equal size, the remainder is added
    # at the end- this is a MAJOR problem with equal weightings!

    result = []
    for i in range(int(len(list_to_add) / batch_num) + 1):
        sub_list = list_to_add[batch_num * i : batch_num * (i + 1)]
        if len(sub_list) > 0:
            result.append(reduce(add_centre_sublist, ((D, orthogonalise), sub_list)))
    return result

def add_centre_sublist(*args):
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

    return c.compress_centre_one_site(B_D, orthogonalise=ortho)

def initialise_experiment(n_samples,D,initialise_classifier=False,initialise_classifier_settings=(10, False),centre_site = False):
    D_encode, D_batch, D_final = D
    # Load & Organise Data
    x_train, y_train, x_test, y_test = load_data(
        n_samples, shuffle=False, equal_numbers=True
    )

    #print('Loaded Data!')
    x_train, y_train = arrange_data(x_train, y_train, arrangement="one class")
    #print('Arranged Data!')

    # All possible class labels
    possible_labels = list(set(y_train))

    # Number of total sites (mps encoding)
    n_sites = int(np.ceil(math.log(x_train.shape[-1], 2)))

    # Create hairy bitstrings
    hairy_bitstrings_data = create_hairy_bitstrings_data(
        possible_labels, n_sites
    )
    q_hairy_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, n_sites, truncated = True
    )

    if centre_site:
        hairy_bitstrings_data = [label_last_site_to_centre(b) for b in q_hairy_bitstrings]
        q_hairy_bitstrings = centred_bitstring_to_qtn(hairy_bitstrings_data)

    # MPS encode data
    mps_train = mps_encoding(x_train, D_encode)

    batch_num, ortho_at_end = initialise_classifier_settings
    batch_final = batch_num.pop(-1)

    #Add together all images in the same class.
    #Produces a sum state for each class
    if centre_site:
        sum_states = prepare_centred_batched_classifier(
            mps_train, y_train, q_hairy_bitstrings, D_batch, batch_num
        )
    else:
        sum_states = prepare_batched_classifier(
            mps_train, y_train, q_hairy_bitstrings, D_batch, batch_num, True
        )

    qsum_states = [data_to_QTN(s.data) for s in sum_states]

    #Add together sum states from all classes
    if centre_site:
        classifier_data = adding_centre_batches(sum_states, D_final, batch_final, orthogonalise = ortho_at_end)[0]
    else:
        classifier_data = adding_batches(sum_states, D_final, batch_final, orthogonalise = ortho_at_end)[0]

    #classifier_data = classifier_data.compress_to_centre_one_site(D=32, orthogonalise = False)
    #classifier_data = classifier_data.compress_one_site(D=32, orthogonalise = False)

    mpo_classifier = data_to_QTN(classifier_data.data)
    print(mpo_classifier)
    return (mps_train, y_train), (mpo_classifier, qsum_states), q_hairy_bitstrings

if __name__ == "__main__":

    num_samples = 1000
    D = (32, 32, 32)
    batch_num = [10,10,10]
    ortho_at_end = False

    data, classifier, bitstrings = initialise_experiment(
                num_samples,
                D,
                initialise_classifier=True,
                centre_site = False,
                initialise_classifier_settings=(batch_num, ortho_at_end),
            )
    """
    CENTRE SITE IS FIRST SITE AT THE MOMENT
    """

    mps_images, labels = data
    classifier, sum_states = classifier

    predictions = np.array([abs((mps_image.H @ classifier).squeeze().data) for mps_image in tqdm(mps_images)])
    #predictions = [[abs(mps_image.H.squeeze() @ (classifier @ b).squeeze()) for b in bitstrings] for mps_image in tqdm(mps_images)]

    print(evaluate_classifier_top_k_accuracy(predictions, labels, 1))

    #sum_state_predictions = [[abs(mps_image.H.squeeze() @ (s.squeeze() @ b.squeeze())) for s,b in zip(sum_states, bitstrings)] for mps_image in tqdm(mps_images)]
    #print(evaluate_classifier_top_k_accuracy(sum_state_predictions, labels, 1))
