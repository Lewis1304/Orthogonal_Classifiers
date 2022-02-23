from variational_mpo_classifiers import *
from deterministic_mpo_classifier import *
from stacking import *
from experiments import initialise_experiment, label_last_site_to_centre, centred_bitstring_to_qtn, prepare_centred_batched_classifier, adding_centre_batches, add_centre_sublist
#prepare_batched_classifier, prepare_ensemble, unitary_qtn, prepare_sum_states, adding_batches, prepare_linear_classifier, linear_classifier_predictions
import os
from tqdm import tqdm
from xmps.svd_robust import svd

"""
Get sum states
"""

def D_total_experiment():
    print('ACCURACY VS D_TOTAL EXPERIMENT')

    D_totals = range(2,37,2)
    num_samples = 5421*10
    #num_samples = 1000
    batch_nums = [3, 13, 139, 10]
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
        [save_qtn_classifier(s , "mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_{D_total}/" + f"digit_{i}") for i, s in enumerate(sum_states)]

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

    n_train_samples = 5421*10
    #n_train_samples = 1000
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

        non_ortho_classifier_data = adding_centre_batches(sum_states_data, D_total, 10, orthogonalise = False)[0]
        non_ortho_mpo_classifier = data_to_QTN(non_ortho_classifier_data.data).squeeze()

        ortho_classifier_data = adding_centre_batches(sum_states_data, D_total, 10, orthogonalise = True)[0]
        ortho_mpo_classifier = data_to_QTN(ortho_classifier_data.data).squeeze()

        #print('Training predicitions: ')
        non_ortho_training_prediction = [np.abs((mps_image.H.squeeze() @ non_ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_train)]
        ortho_training_prediction = [np.abs((mps_image.H.squeeze() @ ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_train)]

        #print('Test predicitions: ')
        non_ortho_test_prediction = [np.abs((mps_image.H.squeeze() @ non_ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_test)]
        ortho_test_prediction = [np.abs((mps_image.H.squeeze() @ ortho_mpo_classifier.squeeze()).data) for mps_image in tqdm(mps_test)]


        non_ortho_training_predictions.append(non_ortho_training_prediction)
        ortho_training_predictions.append(ortho_training_prediction)
        non_ortho_test_predictions.append(non_ortho_test_prediction)
        ortho_test_predictions.append(ortho_test_prediction)

        #print('D_total non-ortho test acc:', evaluate_classifier_top_k_accuracy(non_ortho_test_prediction, y_test, 1))
        #print('D_total ortho test acc:', evaluate_classifier_top_k_accuracy(ortho_test_prediction, y_test, 1))
        #assert()

        np.save('Classifiers/mnist_mixed_sum_states/D_total/' + "non_ortho_d_total_vs_training_predictions", non_ortho_training_predictions)
        np.save('Classifiers/mnist_mixed_sum_states/D_total/' + "ortho_d_total_vs_training_predictions", ortho_training_predictions)
        np.save('Classifiers/mnist_mixed_sum_states/D_total/' + "non_ortho_d_total_vs_test_predictions", non_ortho_test_predictions)
        np.save('Classifiers/mnist_mixed_sum_states/D_total/' + "ortho_d_total_vs_test_predictions", ortho_test_predictions)

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

            np.save('Classifiers/mnist_mixed_sum_states/D_encode/' + f"D_final_{D_final}_non_ortho_d_total_vs_test_predictions", non_ortho_test_predictions)
            np.save('Classifiers/mnist_mixed_sum_states/D_encode/' + f"D_final_{D_final}_ortho_d_total_vs_test_predictions", ortho_test_predictions)

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

            np.save('Classifiers/mnist_mixed_sum_states/D_batch/' + f"D_final_{D_final}_non_ortho_d_total_vs_test_predictions", non_ortho_test_predictions)
            np.save('Classifiers/mnist_mixed_sum_states/D_batch/' + f"D_final_{D_final}_ortho_d_total_vs_test_predictions", ortho_test_predictions)

"""
Get Accuracies
"""





if __name__ == '__main__':
    get_D_encode_predictions()
