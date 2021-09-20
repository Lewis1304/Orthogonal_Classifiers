import numpy as np
from functools import reduce
from xmps.fMPS import fMPS
from tqdm import tqdm
from fMPO_reduced import fMPO
from tools import load_data, data_to_QTN
from variational_mpo_classifiers import evaluate_classifier_top_k_accuracy, mps_encoding, create_hairy_bitstrings_data, bitstring_data_to_QTN, squeezed_classifier_predictions, mpo_encoding
from math import log as mlog
"""
Load Data
"""

def prepare_data(n_samples, x_train, y_train, x_test, y_test):

    n_samples_per_class = n_samples // len(list(set(y_train)))
    grouped_x_train = [x_train[y_train == label][:n_samples_per_class] for label in list(set(y_train))]
    grouped_x_test = [x_test[y_test == label][:n_samples_per_class] for label in list(set(y_test))]

    train_data = np.array([images for images in zip(*grouped_x_train)])
    train_labels = [list(set(y_train)) for _ in train_data]
    test_data = np.array([images for images in zip(*grouped_x_test)])
    test_labels = [list(set(y_test)) for _ in range(len(test_data))]

    train_data = np.array([item for sublist in train_data for item in sublist])
    train_labels = np.array([item for sublist in train_labels for item in sublist])
    test_data = np.array([item for sublist in test_data for item in sublist])
    test_labels = np.array([item for sublist in test_labels for item in sublist])

    return train_data,train_labels,test_data,test_labels

"""
Encode Data
"""

def fmps_encoding(images, D=2):

    def image_to_mps(f_image, D):
        num_pixels = f_image.shape[0]
        L = int(np.ceil(np.log2(num_pixels)))
        padded_image = np.pad(f_image, (0, 2**L - num_pixels)).reshape(*[2] * L)
        return fMPS().left_from_state(padded_image).left_canonicalise(D=D)

    #Embed as MPSs
    images_mps = [image_to_mps(image, D) for image in images]

    return images_mps

def spread_ditstring(n_hairy_sites, possible_labels, n_features, truncate = True, one_site = False):

    if one_site:
        num_qubits = int(np.log2(len(possible_labels))) + 1
        hairy_sites = np.expand_dims(
            [i for i in np.eye(2 ** num_qubits)][: len(possible_labels)], 1
        )

        other_sites = np.array(
            [
                [np.eye(2 ** num_qubits)[0] for pixel in range(n_features - 1)]
                for _ in possible_labels
            ]
        )

    else:
        """
        Iterates over bitstrings in chunks. i.e i = bitstring[chunksize*i:chunksize*(i+1)]
        different basis vector for each chunk. i.e. 00 = [1,0,0,0], 01 = [0,1,0,0]
        """
        bitstrings = [bin(label)[2:].zfill(n_hairy_sites*2) for label in possible_labels]

        hairy_sites = np.array([[[1,0,0,0]*(1-int(bitstring[i:i + 2][0]))*(1-int(bitstring[i:i + 2][1])) +
              [0,1,0,0]*(1-int(bitstring[i:i + 2][0]))*(int(bitstring[i:i + 2][1])) +
              [0,0,1,0]*(int(bitstring[i:i + 2][0]))*(1-int(bitstring[i:i + 2][1])) +
              [0,0,0,1]*(int(bitstring[i:i + 2][0]))*(int(bitstring[i:i + 2][1]))
              for i in range(0, len(bitstring), 2)]
              for bitstring in bitstrings])

        other_sites = np.array([[[1,0,0,0] for pixel in range(n_features - n_hairy_sites)] for label in possible_labels])

    spread_ditstrings = np.append(other_sites,hairy_sites,axis=1)


    if truncate:
        spread_ditstrings = [[ spread_ditstrings[label][pixel][:1] if pixel < (n_features-n_hairy_sites) else spread_ditstrings[label][pixel][:] for pixel in range(n_features)] for label in possible_labels]
        return spread_ditstrings

    return spread_ditstrings

def mpo_data_to_mpo(possible_labels,mpo_data):
    return {label: [fMPO(mpo_prod_image) for mpo_prod_image in mpo_data[label]] for label in possible_labels}

def mpo_encoded_data(train_data, train_labels, D, one_site = True):
    possible_labels = list(set(train_labels))
    n_train = int(len(train_data)/len(possible_labels))
    n_features = int(np.log2(len(train_data[0]))) + 1

    if not one_site:
        n_hairy_sites = int((int(np.log2(len(possible_labels)))+1)/2)
    else:
        n_hairy_sites = 1

    train_mps = [fmps_encoding(train_data[train_labels == label], D=D) for label in possible_labels]

    b = spread_ditstring(n_hairy_sites, possible_labels, n_features, truncate=False, one_site = one_site)

    mpo_data_train = [[[np.array([train_mps[label][image][pixel] * i for i in j]).transpose(1,0,2,3)
            for pixel,j in enumerate(b[label])]
            for image in range(n_train)]
            for label in possible_labels]

    mpo_data_train_trunc = [[[ mpo_data_train[label][image][pixel][:,:1,:,:] if pixel < (n_features-n_hairy_sites) else mpo_data_train[label][image][pixel][:,:,:,:] for pixel in range(n_features)]  for image in range(n_train)] for label in possible_labels]

    MPOs_train = mpo_data_to_mpo(possible_labels, mpo_data_train_trunc)
    return MPOs_train

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

def adding_batches(list,D,batch_num=2,truncate=True, orthogonalise = False):
    # if batches are not of equal size, the remainder is added
    # at the end- this is a MAJOR problem with equal weightings!

    if len(list) % batch_num != 0:
        if not truncate:
            raise Exception('Batches are not of equal size!')
        else:
            trun_expo = int(np.log(len(list)) / np.log(batch_num))
            list = list[:batch_num**trun_expo]
    result = []

    for i in range(int(len(list)/batch_num)+1):
        sub_list = list[batch_num*i:batch_num*i+batch_num]
        if len(sub_list) > 0:
            result.append(reduce(add_sublist,((D, orthogonalise),sub_list)))
    return result

"""
Evaluate classifier
"""

def deterministic_classifier_predictions(classifier, mps_test, bitstrings):
    def mpo_overlap(classifier, bitstring, test_datum):
        return np.abs(classifier.apply_mpo_from_bottom(bitstring).overlap(test_datum))

    return np.array([[mpo_overlap(classifier,i,j) for i in bitstrings] for j in (mps_test)])

"""
Prepare classifier
"""

def prepare_batched_classifier(train_data, train_labels, D_total, batch_num, one_site = False, ortho_at_end = False):

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
        MPOs = adding_batches(MPOs, D_total, batch_num, orthogonalise=False)

    return MPOs[0]

"""
Experiment
"""

def sequential_mpo_classifier_experiment():
    #Results are different if shape one-site = True or False in batch adding procedure?
    #Results are not different when compressing at the end.

    n_samples = 1000
    possible_labels = list(range(10))
    #D_total = 32
    one_site = False
    batch_num = 10
    n_rounds = 10

    if one_site:
        n_hairy_sites = 1
    else:
        n_hairy_sites = int((int(np.log2(len(possible_labels)))+1)/2)

    results_non_ortho = []
    results_ortho = []
    for _ in tqdm(range(n_rounds)):
        for D_total in tqdm(range(2,42,2)):

            #Load Data
            x_train, y_train, x_test, y_test = load_data(60000, shuffle = True)
            train_data, train_labels, test_data, test_labels = prepare_data(n_samples, x_train, y_train, x_test, y_test)

            for ortho_at_end in [False, True]:
                #Add/train classifier
                classifier = prepare_batched_classifier(train_data, train_labels, D_total, batch_num, one_site = one_site, ortho_at_end = ortho_at_end)

                #Evaluate Classifier
                n_features = int(np.log2(len(x_train[0]))) + 1
                list_spread_ditstrings = [fMPS().from_product_state(i) for i in spread_ditstring(1, possible_labels, n_features, one_site = True)]
                predicitions = deterministic_classifier_predictions(classifier, fmps_encoding(train_data, D=D_total), list_spread_ditstrings)
                result = evaluate_classifier_top_k_accuracy(predicitions, train_labels, 1)

                if not ortho_at_end:
                    results_non_ortho.append(result)
                else:
                    results_ortho.append(result)

    np.save('results_non_ortho', results_non_ortho)
    np.save('results_ortho', results_ortho)


def batch_initialise_classifier():
    D_total = 32
    n_train = 1000
    one_site = False
    ortho_at_end = False
    batch_num = 10

    #Load Data- ensuring particular order to be batched added
    train_data, train_labels, test_data, test_labels = load_data(n_train, shuffle = False, equal_numbers = True)
    possible_labels = list(set(train_labels))

    train_data = [[train_data[i] for i in range(k, len(train_data), len(possible_labels))] for k in possible_labels]
    train_data = np.array([image for label in train_data for image in label])

    train_labels = [[train_labels[i] for i in range(k, len(train_labels), len(possible_labels))] for k in possible_labels]
    train_labels = np.array([image for label in train_labels for image in label])

    #Add images together- forming classifier initialisation
    fMPO_classifier = prepare_batched_classifier(train_data, train_labels, D_total, batch_num, one_site = one_site)
    qtn_classifier_data = fMPO_classifier.compress_one_site(D=D_total, orthogonalise=ortho_at_end)
    qtn_classifier = data_to_QTN(qtn_classifier_data.data).squeeze()

    #Evaluating Classifier
    n_hairy_sites = 1
    n_sites = 10
    one_site = True
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
