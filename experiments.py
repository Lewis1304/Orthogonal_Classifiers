from variational_mpo_classifiers import *
from deterministic_mpo_classifier import *
#from stacking import *

import os
from tqdm import tqdm
from xmps.svd_robust import svd

"""
Prepare Experiment
"""


def initialise_experiment(n_samples, D, arrangement="one class", initialise_classifier=False, prep_sum_states = False, centre_site= False, initialise_classifier_settings=(10, False)):
    """
    int: n_samples: Number of data samples (total)
    int: D_total: Bond dimension of classifier and data
    string: arrangement: Order of training images- this matters for batch added initialisation
    bool: initialise_classifier: Whether classifier is initialised using batch adding procedure
    tuple: initialise_classifier_settings: (
                                            batch_num: how many images per batch,
                                            ortho_at_end: Whether polar decomp is performed at the end or not
                                            )
    """
    D_encode, D_batch, D_final = D

    # Load & Organise Data
    x_train, y_train, x_test, y_test = load_data(
        n_samples, shuffle=False, equal_numbers=True
    )

    #print('Loaded Data!')
    x_train, y_train = arrange_data(x_train, y_train, arrangement=arrangement)
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
        hairy_bitstrings_data, n_sites
    )

    if centre_site:
        hairy_bitstrings_data = [label_last_site_to_centre(b) for b in q_hairy_bitstrings]
        q_hairy_bitstrings = centred_bitstring_to_qtn(hairy_bitstrings_data)

    # MPS encode data
    mps_train = mps_encoding(x_train, D_encode)

    #print('Encoded Data!')

    # Initial Classifier
    if initialise_classifier:

        batch_nums, ortho_at_end = initialise_classifier_settings

        #Prepare sum states for each class. Then adds sum states together.
        if prep_sum_states:
            assert(arrangement == "one class")
            batch_final = batch_nums.pop(-1)

            if centre_site:
                sum_states = prepare_centred_batched_classifier(
                    mps_train, y_train, q_hairy_bitstrings, D_batch, batch_nums
                )
            else:
                sum_states = prepare_batched_classifier(
                    mps_train, y_train, q_hairy_bitstrings, D_batch, batch_nums, prep_sum_states
                )

            qsum_states = [data_to_QTN(s.data) for s in sum_states]

            if centre_site:
                classifier_data = adding_centre_batches(sum_states, D_final, batch_final, orthogonalise = ortho_at_end)[0]
            else:
                classifier_data = adding_batches(sum_states, D_final, batch_final, orthogonalise = ortho_at_end)[0]

            mpo_classifier = data_to_QTN(classifier_data.data)

            return (mps_train, y_train), (mpo_classifier, qsum_states), q_hairy_bitstrings

        else:

            fmpo_classifier = prepare_batched_classifier(
                mps_train, y_train, D_batch, batch_nums
            )

            classifier_data = fmpo_classifier.compress_one_site(
                D=D_final, orthogonalise=ortho_at_end
            )
            mpo_classifier = data_to_QTN(classifier_data.data)#.squeeze()

    else:
        # MPO encode data (already encoded as mps)
        # Has shape: # classes, mpo.shape
        old_classifier_data = prepare_batched_classifier(
            mps_train[:10], list(range(10)), 32, [10]
        ).compress_one_site(D=32, orthogonalise=False)
        old_classifier = data_to_QTN(old_classifier_data.data)#.squeeze()
        mpo_classifier = create_mpo_classifier_from_initialised_classifier(old_classifier).squeeze()

    return (mps_train, y_train), mpo_classifier, q_hairy_bitstrings

"""
Get sum states experiments
"""

def D_total_experiment():
    print('ACCURACY VS D_TOTAL EXPERIMENT')

    D_totals = range(2,37,2)
    #num_samples = 5421*10
    num_samples = 1000
    #batch_nums = [3, 13, 139, 10]
    batch_nums = [10, 10, 10]
    ortho_at_end = False


    for D_total in tqdm(D_totals):
        D_encode = D_total
        D_batch = D_total
        D_final = D_total

        D = (D_encode, D_batch, D_final)
        data, classifiers, bitstrings = initialise_experiment(
                    num_samples,
                    D,
                    arrangement='one class',
                    initialise_classifier=True,
                    prep_sum_states = True,
                    centre_site = True,
                    initialise_classifier_settings=([3, 13, 139, 10], ortho_at_end),
                )
        mps_images, labels = data
        _, sum_states = classifiers

        path = "Classifiers/mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_{D_total}/"
        os.makedirs(path, exist_ok=True)
        #[save_qtn_classifier(s , "mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_{D_total}/" + f"digit_{i}") for i, s in enumerate(sum_states)]

def D_encode_experiment():
    print('ACCURACY VS D_ENCODE EXPERIMENT')

    D_encodes = range(2,33,2)
    D_batch = 32
    D_final = 32
    num_samples = 5421*10
    #num_samples = 1000
    batch_nums = [3, 13, 139, 10]
    #batch_nums = [10, 10, 10]
    ortho_at_end = False

    for D_encode in tqdm(D_encodes):

        D = (D_encode, D_batch, D_final)
        data, classifier, bitstrings = initialise_experiment(
                    num_samples,
                    D,
                    arrangement='one class',
                    initialise_classifier=True,
                    prep_sum_states = True,
                    centre_site = True,
                    initialise_classifier_settings=([3, 13, 139, 10], ortho_at_end),
                    )
        mps_images, labels = data
        _, sum_states = classifier

        path = "Classifiers/mnist_mixed_sum_states/D_encode/" + f"sum_states_D_encode_{D_encode}/"
        os.makedirs(path, exist_ok=True)
        [save_qtn_classifier(s , "mnist_mixed_sum_states/D_encode/" + f"sum_states_D_encode_{D_encode}/" + f"digit_{i}") for i, s in enumerate(sum_states)]

def D_batch_experiment():
    print('ACCURACY VS D_BATCH EXPERIMENT')

    D_batches = range(2,33,2)
    D_encode = 32
    num_samples = 5421*10
    #num_samples = 1000
    batch_nums = [3, 13, 139]
    #batch_nums = [10, 10, 10]
    ortho_at_end = False

    x_train, y_train, x_test, y_test = load_data(
        num_samples, shuffle=False, equal_numbers=True
    )
    x_train, y_train = arrange_data(x_train, y_train, arrangement='one class')
    mps_train = mps_encoding(x_train, D_encode)
    q_hairy_bitstrings = create_experiment_bitstrings(x_train, y_train)


    for D_batch in tqdm(D_batches):

        list_of_classifiers = prepare_centred_batched_classifier(
            mps_train, y_train, q_hairy_bitstrings, D_batch, batch_nums
        )

        qsum_states = [data_to_QTN(s.data) for s in list_of_classifiers]

        path = "Classifiers/mnist_mixed_sum_states/D_batch/" + f"sum_states_D_batch_{D_batch}/"
        os.makedirs(path, exist_ok=True)
        [save_qtn_classifier(s , "mnist_mixed_sum_states/D_batch/" + f"sum_states_D_batch_{D_batch}/" + f"digit_{i}") for i, s in enumerate(qsum_states)]

def create_experiment_bitstrings(x_train, y_train):

    possible_labels = list(set(y_train))

    # Number of total sites (mps encoding)
    n_sites = int(np.ceil(math.log(x_train.shape[-1], 2)))

    # Create hairy bitstrings
    hairy_bitstrings_data = create_hairy_bitstrings_data(
        possible_labels, n_sites
    )
    q_hairy_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, n_sites
    )

    hairy_bitstrings_data = [label_last_site_to_centre(b) for b in q_hairy_bitstrings]
    q_hairy_bitstrings = centred_bitstring_to_qtn(hairy_bitstrings_data)

    return q_hairy_bitstrings

"""
Get predictions
"""

def get_D_total_predictions():
    print('OBTAINING D_TOTAL PREDICTIONS')

    #n_train_samples = 5421*10
    n_train_samples = 1000
    n_test_samples = 10000
    #n_test_samples = 1000

    x_train, y_train, x_test, y_test = load_data(
        n_train_samples,n_test_samples, shuffle=False, equal_numbers=True
    )
    x_train, y_train = arrange_data(x_train, y_train, arrangement='one class')

    # MPS encode data
    D_encode = 32
    mps_train = mps_encoding(x_train, D_encode)
    mps_test = mps_encoding(x_test, D_encode)


    non_ortho_training_predictions = []
    ortho_training_predictions = []
    non_ortho_test_predictions = []
    ortho_test_predictions = []

    D_totals = range(2, 37, 2)
    for D_total in tqdm(D_totals):

        path = "mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_{D_total}/"

        sum_states = [load_qtn_classifier(path + f"digit_{i}") for i in range(10)]
        sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in sum_states]

        #non_ortho_classifier_data = adding_centre_batches(sum_states_data, D_total, 10, orthogonalise = False)[0]
        #non_ortho_mpo_classifier = data_to_QTN(non_ortho_classifier_data.data).squeeze()

        ortho_classifier_data = adding_centre_batches(sum_states_data, D_total, 10, orthogonalise = True)[0]
        ortho_mpo_classifier = data_to_QTN(ortho_classifier_data.data).squeeze()

        #print('Training predicitions: ')
        #non_ortho_training_prediction = [np.abs((mps_image.H.squeeze() @ non_ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_train)]
        #ortho_training_prediction = [np.abs((mps_image.H.squeeze() @ ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_train)]

        #print('Test predicitions: ')
        #non_ortho_test_prediction = [np.abs((mps_image.H.squeeze() @ non_ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_test)]
        ortho_test_prediction = [np.abs((mps_image.H.squeeze() @ ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_test)]
        #non_ortho_training_predictions.append(non_ortho_training_prediction)
        #ortho_training_predictions.append(ortho_training_prediction)
        #non_ortho_test_predictions.append(non_ortho_test_prediction)
        ortho_test_predictions.append(ortho_test_prediction)

        #print('D_total non-ortho test acc:', evaluate_classifier_top_k_accuracy(non_ortho_test_prediction, y_test, 1))
        print('D_total ortho test acc:', evaluate_classifier_top_k_accuracy(ortho_test_prediction, y_test, 1))

        #np.save('Classifiers/mnist_mixed_sum_states/D_total/' + "non_ortho_d_total_vs_training_predictions", non_ortho_training_predictions)
        #np.save('Classifiers/mnist_mixed_sum_states/D_total/' + "ortho_d_total_vs_training_predictions", ortho_training_predictions)
        #np.save('Classifiers/mnist_mixed_sum_states/D_total/' + "non_ortho_d_total_vs_test_predictions", non_ortho_test_predictions)
        np.save('Classifiers/mnist_mixed_sum_states/D_total/' + "ortho_d_final_vs_test_predictions", ortho_test_predictions)

def get_D_final_predictions():
    print('OBTAINING D_FINAL PREDICTIONS')

    #n_train_samples = 5421*10
    #n_train_samples = 60000
    n_train_samples = 1000
    n_test_samples = 10000
    #n_test_samples = 1000

    x_train, y_train, x_test, y_test = load_data(
        n_train_samples,n_test_samples, shuffle=False, equal_numbers=True
    )
    x_train, y_train = arrange_data(x_train, y_train, arrangement='one class')

    # MPS encode data
    D_encode = 32
    mps_train = mps_encoding(x_train, D_encode)
    mps_test = mps_encoding(x_test, D_encode)


    non_ortho_training_predictions = []
    ortho_training_predictions = []
    non_ortho_test_predictions = []
    ortho_test_predictions = []

    path = "mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_{32}/"

    sum_states = [load_qtn_classifier(path + f"digit_{i}") for i in range(10)]
    sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in sum_states]

    D_finals = range(2, 37, 2)
    for D_final in tqdm(D_finals):

        non_ortho_classifier_data = adding_centre_batches(sum_states_data, D_final, 10, orthogonalise = False)[0]
        non_ortho_mpo_classifier = data_to_QTN(non_ortho_classifier_data.data).squeeze()

        ortho_classifier_data = adding_centre_batches(sum_states_data, D_final, 10, orthogonalise = True)[0]
        ortho_mpo_classifier = data_to_QTN(ortho_classifier_data.data).squeeze()

        #print('Training predicitions: ')
        #non_ortho_training_prediction = [np.abs((mps_image.H.squeeze() @ non_ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_train)]
        #ortho_training_prediction = [np.abs((mps_image.H.squeeze() @ ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_train)]

        #print('Test predicitions: ')
        non_ortho_test_prediction = [np.abs((mps_image.H.squeeze() @ non_ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_test)]
        ortho_test_prediction = [np.abs((mps_image.H.squeeze() @ ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_test)]


        #non_ortho_training_predictions.append(non_ortho_training_prediction)
        #ortho_training_predictions.append(ortho_training_prediction)
        non_ortho_test_predictions.append(non_ortho_test_prediction)
        ortho_test_predictions.append(ortho_test_prediction)

        #print('D_total non-ortho test acc:', evaluate_classifier_top_k_accuracy(non_ortho_test_prediction, y_test, 1))
        #print('D_total ortho test acc:', evaluate_classifier_top_k_accuracy(ortho_test_prediction, y_test, 1))
        #assert()

        #np.savez_compressed('Classifiers/fashion_mnist_mixed_sum_states/D_total/' + "non_ortho_d_final_vs_training_predictions_compressed", non_ortho_training_predictions)
        #np.savez_compressed('Classifiers/fashion_mnist_mixed_sum_states/D_total/' + "ortho_d_final_vs_training_predictions_compressed", ortho_training_predictions)
        np.save('Classifiers/mnist_mixed_sum_states/D_total/' + "non_ortho_d_final_vs_test_predictions", non_ortho_test_predictions)
        np.save('Classifiers/mnist_mixed_sum_states/D_total/' + "ortho_d_final_vs_test_predictions", ortho_test_predictions)

def get_D_encode_predictions():

    n_train_samples = 1000
    n_test_samples = 10000

    x_train, y_train, x_test, y_test = load_data(
        n_train_samples,n_test_samples, shuffle=False, equal_numbers=True
    )

    # MPS encode data
    D_encode = 32
    mps_train = mps_encoding(x_train, D_encode)
    mps_test = mps_encoding(x_test, D_encode)

    non_ortho_test_predictions = []
    ortho_test_predictions = []

    D_encodes = range(2, 33, 2)
    for D_final in [10,20,32]:
        non_ortho_test_predictions = []
        ortho_test_predictions = []

        for D_encode in tqdm(D_encodes):

            path = "mnist_mixed_sum_states/D_encode/" + f"sum_states_D_encode_{D_encode}/"
            sum_states = [load_qtn_classifier(path + f"digit_{i}") for i in range(10)]
            sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in sum_states]


            non_ortho_classifier_data = adding_centre_batches(sum_states_data, D_final, 10, orthogonalise = False)[0]
            non_ortho_mpo_classifier = data_to_QTN(non_ortho_classifier_data.data).squeeze()
            ortho_classifier_data = adding_centre_batches(sum_states_data, D_final, 10, orthogonalise = True)[0]
            ortho_mpo_classifier = data_to_QTN(ortho_classifier_data.data).squeeze()


            #print('Test predicitions: ')
            non_ortho_test_prediction = [np.abs((mps_image.H.squeeze() @ non_ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_test)]
            ortho_test_prediction = [np.abs((mps_image.H.squeeze() @ ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_test)]


            non_ortho_test_predictions.append(non_ortho_test_prediction)
            ortho_test_predictions.append(ortho_test_prediction)

            #print('D_encode non-ortho test acc:', evaluate_classifier_top_k_accuracy(non_ortho_test_prediction, y_test, 1))
            #print('D_encode ortho test acc:', evaluate_classifier_top_k_accuracy(ortho_test_prediction, y_test, 1))
            #assert()

            #np.save('Classifiers/mnist_mixed_sum_states/D_encode/' + f"D_final_{D_final}_non_ortho_d_total_vs_test_predictions", non_ortho_test_predictions)
            #np.save('Classifiers/mnist_mixed_sum_states/D_encode/' + f"D_final_{D_final}_ortho_d_total_vs_test_predictions", ortho_test_predictions)

def get_D_batch_predictions():

    n_train_samples = 1000
    n_test_samples = 10000

    x_train, y_train, x_test, y_test = load_data(
        n_train_samples,n_test_samples, shuffle=False, equal_numbers=True
    )

    # MPS encode data
    D_encode = 32
    mps_test = mps_encoding(x_test, D_encode)

    non_ortho_test_predictions = []
    ortho_test_predictions = []

    D_batches = range(2, 33, 2)
    for D_final in [10,20,32]:
        non_ortho_test_predictions = []
        ortho_test_predictions = []

        for D_batch in tqdm(D_batches):

            path = "mnist_mixed_sum_states/D_batch/" + f"sum_states_D_batch_{D_batch}/"
            sum_states = [load_qtn_classifier(path + f"digit_{i}") for i in range(10)]
            sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in sum_states]


            non_ortho_classifier_data = adding_centre_batches(sum_states_data, D_final, 10, orthogonalise = False)[0]
            non_ortho_mpo_classifier = data_to_QTN(non_ortho_classifier_data.data).squeeze()
            ortho_classifier_data = adding_centre_batches(sum_states_data, D_final, 10, orthogonalise = True)[0]
            ortho_mpo_classifier = data_to_QTN(ortho_classifier_data.data).squeeze()


            #print('Test predicitions: ')
            non_ortho_test_prediction = [np.abs((mps_image.H.squeeze() @ non_ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_test)]
            ortho_test_prediction = [np.abs((mps_image.H.squeeze() @ ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_test)]


            non_ortho_test_predictions.append(non_ortho_test_prediction)
            ortho_test_predictions.append(ortho_test_prediction)

            #print('D_total non-ortho test acc:', evaluate_classifier_top_k_accuracy(non_ortho_test_prediction, y_test, 1))
            #print('D_total ortho test acc:', evaluate_classifier_top_k_accuracy(ortho_test_prediction, y_test, 1))
            #assert()

            #np.save('Classifiers/mnist_mixed_sum_states/D_batch/' + f"D_final_{D_final}_non_ortho_d_total_vs_test_predictions", non_ortho_test_predictions)
            #np.save('Classifiers/mnist_mixed_sum_states/D_batch/' + f"D_final_{D_final}_ortho_d_total_vs_test_predictions", ortho_test_predictions)

"""
Centre site functions
"""

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

"""
Stacking
"""
def mps_stacking(dataset, n_copies):

    def generate_copy_state(QTN, n_copies):
        initial_QTN = QTN
        for _ in range(n_copies):
            QTN = QTN | initial_QTN
        return relabel_QTN(QTN)

    def relabel_QTN(QTN):
        qtn_data = []
        previous_ind = rand_uuid()
        for j, site in enumerate(QTN.tensors):
            next_ind = rand_uuid()
            tensor = qtn.Tensor(
                site.data, inds=(f"k{j}", previous_ind, next_ind), tags=oset([f"{j}"])
            )
            previous_ind = next_ind
            qtn_data.append(tensor)
        return qtn.TensorNetwork(qtn_data)

    #Upload Data
    training_label_qubits = np.load('Classifiers/' + dataset + '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle = True)['arr_0'][15]#.astype(np.float32)
    y_train = np.load('Classifiers/' + dataset + '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_labels.npy')#.astype(np.int8)
    training_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in training_label_qubits])

    #training_label_qubits = np.array(reduce(list.__add__, [list(training_label_qubits[i*6000 : i * 6000 + 10]) for i in range(10)]))
    #y_train = np.array(reduce(list.__add__, [list(y_train[i*6000 : i * 6000 + 10]) for i in range(10)]))

    outer_ket_states = training_label_qubits
    for k in range(n_copies):
        outer_ket_states = np.array([np.kron(i, j) for i,j in zip(outer_ket_states, training_label_qubits)])

    #Convert predictions to MPS
    print('Encoding predictions...')
    D_encode = 32
    #training_mps_predictions = mps_encoding(training_label_qubits, 4)
    training_mps_predictions = mps_encoding(outer_ket_states, D_encode)

    #Add n_copies of same prediction state
    #training_copied_predictions = [generate_copy_state(pred,n_copies) for pred in training_mps_predictions]
    training_copied_predictions = training_mps_predictions

    #Create bitstrings to add onto copied states
    possible_labels = list(set(y_train))
    n_sites = training_copied_predictions[0].num_tensors

    hairy_bitstrings_data = create_hairy_bitstrings_data(
        possible_labels, n_sites
    )
    q_hairy_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, n_sites
    )
    hairy_bitstrings_data = [label_last_site_to_centre(b) for b in q_hairy_bitstrings]
    q_hairy_bitstrings = centred_bitstring_to_qtn(hairy_bitstrings_data)


    """
    train_mpos = mpo_encoding(training_copied_predictions, y_train, q_hairy_bitstrings)
    train_fmpos = np.array([fMPO([site.data for site in mpo.tensors]) for mpo in train_mpos])

    print('Batch adding predictions...')
    sum_states = []
    for l in tqdm(possible_labels):
        sub_list_mpo = train_fmpos[y_train == l][0]
        for i in tqdm(train_fmpos[y_train == l][1:]):
            sub_list_mpo = sub_list_mpo.add(i).compress_centre_one_site(4, orthogonalise=False)
        sum_states.append(sub_list_mpo)

    fMPO_classifier = sum_states[0]
    for s in sum_states[1:]:
        fMPO_classifier = fMPO_classifier.add(s)

    fMPO_stacking_unitary = fMPO_classifier.compress_centre_one_site(4, orthogonalise=True)
    stacking_unitary = data_to_QTN(fMPO_stacking_unitary.data)
    """

    #Parameters for batch adding label predictions
    D_batch = 32
    batch_nums = [3, 13, 139]
    #batch_nums = [2, 3, 5, 2, 5, 2, 5, 2, 10]
    #batch_nums = [100]

    #Batch adding copied predictions to create sum states
    print('Batch adding predictions...')
    sum_states = prepare_centred_batched_classifier(training_copied_predictions, y_train, q_hairy_bitstrings, D_batch, batch_nums)

    #Batch adding sum states to create stacking unitary
    D_final = 32
    batch_final = 10
    ortho_at_end = True
    classifier_data = adding_centre_batches(sum_states, D_final, batch_final, orthogonalise = ortho_at_end)[0]
    stacking_unitary = data_to_QTN(classifier_data.data)#.squeeze()

    #Evaluate mps stacking unitary
    test_label_qubits = np.load('Classifiers/' + dataset + '_mixed_sum_states/D_total/ortho_d_final_vs_test_predictions.npy')[15]
    y_test = np.load('Classifiers/' + dataset + '_mixed_sum_states/D_total/ortho_d_final_vs_test_predictions_labels.npy')
    test_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in test_label_qubits])

    #test_label_qubits = test_label_qubits[:100]
    #y_test = y_test[:100]

    #Generate test copy states
    print('Encoding test predictions...')
    outer_ket_states = test_label_qubits
    for k in range(n_copies):
        outer_ket_states = np.array([np.kron(i, j) for i,j in zip(outer_ket_states, test_label_qubits)])

    #mps_test = mps_encoding(test_label_qubits, 4)
    mps_test = mps_encoding(outer_ket_states, outer_ket_states.shape[1])

    #test_copied_predictions = [generate_copy_state(pred,n_copies) for pred in mps_test]
    test_copied_predictions = mps_test

    #Perform overlaps
    print('Performing overlaps...')
    stacked_predictions = np.array([np.abs((mps_image.H.squeeze() @ stacking_unitary.squeeze()).data) for mps_image in tqdm(test_copied_predictions)])


    #Compute Accuracy
    print()
    print('Test accuracy before:', evaluate_classifier_top_k_accuracy(test_label_qubits, y_test, 1))
    print('Test accuracy U:', evaluate_classifier_top_k_accuracy(stacked_predictions, y_test, 1))
    print()



if __name__ == "__main__":
    for i in range(2,3):
        mps_stacking('mnist',i)
    assert()
    #obtain_D_encode_preds()
    #single_image_sv, sum_state_sv = mps_image_singular_values()
    num_samples = 1000
    #batch_nums = [5, 2, 5, 2, 10]
    batch_nums = [10, 10, 10]
    #num_samples = 5421*10
    #batch_nums = [3, 13, 139, 10]
    #num_samples = 6000*10
    #batch_nums = [2, 3, 5, 2, 5, 2, 5, 2, 10]
    ortho_at_end = False
    D_total = 32
    #print('COLLECTING D_TOTAL SUM STATES')
    D_encode = D_total
    D_batch = D_total
    D_final = D_total
    D = (D_encode, D_batch, D_final)

    data, classifiers, bitstrings = initialise_experiment(
                num_samples,
                D,
                arrangement='one class',
                initialise_classifier=True,
                prep_sum_states = True,
                centre_site = True,
                initialise_classifier_settings=([10,10,10], ortho_at_end),
            )
    mps_images, labels = data
    classifier, sum_states = classifiers
    compute_confusion_matrix(bitstrings)
    assert()
    #path = "Classifiers/mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_{D_total}/"
    #os.makedirs(path, exist_ok=True)
    #[save_qtn_classifier(s , "mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_{D_total}/" + f"digit_{i}") for i, s in enumerate(sum_states)]
    #assert()
    #d_batch_vs_acc(bitstrings)
    #assert()
    #generate_classifier_images(sum_states)
    #sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in sum_states]
    #sum_states = [sum_state.compress_one_site(D = D_final, orthogonalise = True) for sum_state in sum_states_data]
    #sum_states = [data_to_QTN(sum_state.data) for sum_state in sum_states]

    #compute_confusion_matrix(bitstrings)
    #assert()
    """
    d_final_vs_acc(bitstrings)
    assert()
    #for i, sum_state in enumerate(sum_states):
    #    save_qtn_classifier(sum_state, f'fashion_mnist_sum_states/sum_state_digit_{i}')
    #assert()
    """

    sum_states = [load_qtn_classifier("mnist_mixed_sum_states/" + f"sum_states_D_total_{32}/" + f"digit_{i}") for i in range(10)]
    sum_states_data = [fMPO([i.data for i in s.tensors]) for s in sum_states]

    classifier_data = adding_centre_batches(sum_states_data, 32, 10, orthogonalise = False)[0]
    classifier = data_to_QTN(classifier_data.data)

    #label_preds = (mps_images[0].H.squeeze() @ sum_states[1].squeeze()).data
    #print(label_preds)
    #print(np.sum([abs(i) for i in label_preds]))
    #assert()

    #sum_state_predictions = [[(mps_image.H.squeeze() @ s.squeeze()).norm() for s in sum_states] for mps_image in tqdm(mps_images)]
    #sum_state_predictions = [[(mps_image.H.squeeze() @ (s.squeeze() @ b.squeeze())) for s,b in zip(sum_states, bitstrings)] for mps_image in tqdm(mps_images)]
    #sum_state_predictions = [[abs(mps_image.H.squeeze() @ s.squeeze()) for s in sum_states] for mps_image in tqdm(mps_images)]
    #print(evaluate_classifier_top_k_accuracy(sum_state_predictions, labels, 1))
    #assert()

    predictions = np.array([abs((mps_image.H @ classifier).squeeze().data) for mps_image in tqdm(mps_images)])
    print(evaluate_classifier_top_k_accuracy(predictions, labels, 1))

    assert()
