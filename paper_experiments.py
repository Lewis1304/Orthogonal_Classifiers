from variational_mpo_classifiers import *
from deterministic_mpo_classifier import *
from stacking import *
from experiments import initialise_experiment, label_last_site_to_centre, centred_bitstring_to_qtn, prepare_centred_batched_classifier, adding_centre_batches, add_centre_sublist
#prepare_batched_classifier, prepare_ensemble, unitary_qtn, prepare_sum_states, adding_batches, prepare_linear_classifier, linear_classifier_predictions
import os
from tqdm import tqdm
from xmps.svd_robust import svd

def accuracy_vs_D_total():
    print('ACCURACY VS D_TOTAL EXPERIMENT')

    D_totals = range(2,37,2)
    num_samples = 5421*10
    batch_nums = [3, 13, 139, 10]
    ortho_at_end = False


    for D_total in tqdm(D_totals):
        D_encode = D_total
        D_batch = D_total
        D_final = D_total
        D = (D_total, D_total, D_total)

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
        [save_qtn_classifier(s , "mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_{D_total}/" + f"digit_{i}") for i, s in enumerate(sum_states)]

def accuracy_vs_D_encode():
    print('ACCURACY VS D_ENCODE EXPERIMENT')

    D_encodes = range(2,33,2)
    D_batch = 32
    D_final = 32
    num_samples = 5421*10
    batch_nums = [3, 13, 139, 10]
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
        _, list_of_classifiers = classifier

        path = "Classifiers/mnist_mixed_sum_states/" + f"sum_states_D_encode_{D_encode}/"
        os.makedirs(path, exist_ok=True)
        [save_qtn_classifier(s , "mnist_mixed_sum_states/" + f"sum_states_D_encode_{D_encode}/" + f"digit_{i}") for i, s in enumerate(list_of_classifiers)]

def accuracy_vs_D_batch():
    print('ACCURACY VS D_BATCH EXPERIMENT')

    D_batches = range(2,33,2)
    D_encode = 32
    num_samples = 5421*10
    batch_nums = [3, 13, 139, 10]
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

        path = "Classifiers/mnist_mixed_sum_states/" + f"sum_states_D_batch_{D_batch}/"
        os.makedirs(path, exist_ok=True)
        [save_qtn_classifier(s , "mnist_mixed_sum_states/" + f"sum_states_D_batch_{D_batch}/" + f"digit_{i}") for i, s in enumerate(qsum_states)]

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


if __name__ == '__main__':
    accuracy_vs_D_total()
