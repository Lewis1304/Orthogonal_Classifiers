from variational_mpo_classifiers import *
from deterministic_mpo_classifier import *
from stacking import *
#prepare_batched_classifier, prepare_ensemble, unitary_qtn, prepare_sum_states, adding_batches, prepare_linear_classifier, linear_classifier_predictions
import os
from tqdm import tqdm
from xmps.svd_robust import svd

"""
Prepare Experiment
"""


def initialise_experiment(
    n_samples,
    D,
    arrangement="one class",
    initialise_classifier=False,
    prep_sum_states = False,
    centre_site= False,
    initialise_classifier_settings=(10, False)
):
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
Experiments
"""


def all_classes_experiment(mpo_classifier, mps_train, q_hairy_bitstrings, y_train, predict_func, loss_func, title):
    print(title)

    classifier_opt = mpo_classifier
    #classifier_opt = pad_qtn_classifier(mpo_classifier)
    # classifier_opt = create_mpo_classifier_from_initialised_classifier(classifier_opt, seed = 420)

    def optimiser(classifier):
        optmzr = TNOptimizer(
            classifier,  # our initial input, the tensors of which to optimize
            loss_fn=lambda c: loss_func(c, mps_train, q_hairy_bitstrings, y_train),
            norm_fn=normalize_tn,
            autodiff_backend="autograd",  # {'jax', 'tensorflow', 'autograd'}
            optimizer="nadam",  # supplied to scipy.minimize
        )
        return optmzr

    save_qtn_classifier(classifier_opt, title + '/mpo_classifier_epoch_0')

    print(classifier_opt)
    print(q_hairy_bitstrings[0])
    best_accuracy = 0

    for i in range(1,1000):
        optmzr = optimiser(classifier_opt)
        classifier_opt = optmzr.optimize(1)

        #if i % 10 == 0:
        save_qtn_classifier(classifier_opt, title + f'/mpo_classifier_epoch_{i}')

        """
        predictions = predict_func(classifier_opt, mps_train, q_hairy_bitstrings)
        predicitions_store.append(predictions)
        accuracy = evaluate_classifier_top_k_accuracy(predictions, y_train, 1)
        accuracies.append(accuracy)

        losses.append(optmzr.loss)

        plot_results((accuracies, losses, predicitions_store), title)
        """
        """
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_qtn_classifier(classifier_opt, title)
        """
    #return accuracies, losses

def svd_classifier(dir, mps_images, labels):
    def compress_QTN(projected_q_mpo, D, orthogonalise):
        # ASSUMES q_mpo IS ALREADY PROJECTED ONTO |0> STATE FOR FIRST (n_sites - n_hairysites) SITES
        # compress procedure leaves mpo in mixed canonical form
        # center site is at left most hairest site.

        if projected_q_mpo.tensors[-1].shape[1] > 4:
            return fMPO_to_QTN(
                QTN_to_fMPO(projected_q_mpo).compress_one_site(
                    D=D, orthogonalise=orthogonalise
                )
            )
        return fMPO_to_QTN(
            QTN_to_fMPO(projected_q_mpo).compress(D=D, orthogonalise=orthogonalise)
        )

    def QTN_to_fMPO(QTN):
        qtn_data = [site.data for site in QTN.tensors]
        return fMPO(qtn_data)

    def fMPO_to_QTN(fmpo):
        fmpo_data = fmpo.data
        return data_to_QTN(fmpo_data)

    classifier_og = load_qtn_classifier(dir)
    # print('Original Classifier:', classifier_og)

    # shift all legs to the right. Does not effect performance.
    classifier_og = fMPO_to_QTN(
        QTN_to_fMPO(classifier_og).compress_one_site(D=None, orthogonalise=False)
    )

    hairy_bitstrings_data = create_hairy_bitstrings_data(
        list(set(labels)), 1, classifier_og.num_tensors, one_site=True
    )
    one_site_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, 1, classifier_og.num_tensors, truncated=True
    )

    predictions_og = classifier_predictions(
        classifier_og, mps_images, one_site_bitstrings
    )
    og_acc = evaluate_classifier_top_k_accuracy(predictions_og, labels, 1)
    print("Original Classifier Accuracy:", og_acc)

    # print('Original Classifier Loss:', stoundenmire_loss(classifier_og, mps_images, bitstrings, labels))

    """
    Shifted, but not orthogonalised
    """
    # classifier_shifted = compress_QTN(classifier_og, None, False)
    # classifier_shifted = fMPO_to_QTN(
    #    QTN_to_fMPO(classifier_og).compress_one_site(
    #        D=None, orthogonalise=False
    #    )
    # )
    # print(classifier_shifted)

    # predictions_shifted = classifier_predictions(classifier_shifted, mps_images, one_site_bitstrings)
    # shifted_acc = evaluate_classifier_top_k_accuracy(predictions_shifted, labels, 1)
    # print('Shifted Classifier Accuracy:', shifted_acc)
    # print('Shifted Classifier Loss:', stoundenmire_loss(classifier_shifted, mps_images, bitstrings, labels))

    """
    Shifted, and orthogonalised
    """
    classifier_ortho = compress_QTN(classifier_og, None, True)
    # print(classifier_ortho)

    predictions_ortho = classifier_predictions(
        classifier_ortho, mps_images, one_site_bitstrings
    )
    ortho_acc = evaluate_classifier_top_k_accuracy(predictions_ortho, labels, 1)
    print("Orthogonalised Classifier Accuracy:", ortho_acc)

    # print('Orthogonalised Classifier Loss:', stoundenmire_loss(classifier_ortho, mps_images, bitstrings, labels))

    return og_acc, ortho_acc

def deterministic_mpo_classifier_experiment(n_samples,batch_num):
    D_encode, D_batch, D_final = (32, None, 50)

    """
    # Load & Organise Data
    """
    x_train, y_train, x_test, y_test = load_data(
        n_samples, shuffle=False, equal_numbers=True
    )
    x_train, y_train = arrange_data(x_train, y_train, arrangement="one class")

    """
    # Create Bitstrings
    """
    # All possible class labels
    possible_labels = list(set(y_train))
    # Number of "label" sites
    n_hairysites = int(np.ceil(math.log(len(possible_labels), 4)))
    # Number of total sites (mps encoding)
    n_sites = int(np.ceil(math.log(x_train.shape[-1], 2)))

    # Create hairy bitstrings
    hairy_bitstrings_data = create_hairy_bitstrings_data(
        possible_labels, n_hairysites, n_sites, True
    )
    q_hairy_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, n_hairysites, n_sites, truncated=True
    )

    """
    # MPS encode data
    """
    mps_train = mps_encoding(x_train, D_encode)

    """
    # Initialise Classifier
    """
    accuracies = []
    from fMPO_reduced import fMPO

    for D_batch in tqdm(range(10, 110, 10)):

        fmpo_classifier = prepare_batched_classifier(
            mps_train, y_train, D_batch, batch_num, one_site=False
        )

        data = fMPO(fmpo_classifier.data)
        classifier_data = data.compress_one_site(
            D=D_final, orthogonalise=False
        )
        mpo_classifier = data_to_QTN(classifier_data.data).squeeze()

        """
        # Evaluate Classifier
        """
        predictions = classifier_predictions(mpo_classifier, mps_train, q_hairy_bitstrings)
        accuracy = evaluate_classifier_top_k_accuracy(predictions, y_train, 1)
        accuracies.append(accuracy)

        np.save(f'accuracies_D_final_{D_final}', accuracies)

def ensemble_experiment(n_classifiers, mps_images, labels, D_total, batch_num):
    ensemble = prepare_ensemble(n_classifiers, mps_images, labels, D_total = D_total, batch_num = batch_num)

    predictions = ensemble_predictions(ensemble, mps_images, bitstrings)
    hard_result = evaluate_hard_ensemble_top_k_accuracy(predictions, labels, 1)
    soft_result = evaluate_soft_ensemble_top_k_accuracy(predictions, labels, 1)

    print('Hard result:', hard_result)
    print('Soft result:', soft_result)

def d_encode_vs_acc():
    print('SIMULATING TEST ACCURACY VS D_ENCODE')
    #Biggest equal size is n_train = 5329 * 10 with batch_num = 73
    #Can use n_train = 4913 with batch_num = 17
    num_samples = 5421*10
    #batch_nums = [139, 39, 10]
    batch_nums = [3, 13, 139, 10]
    #num_samples = 100
    #batch_num = 10

    ortho_at_end = False
    D_batch = 32

    #x_train, y_train, x_test, y_test = load_data(
    #    100,10000, shuffle=False, equal_numbers=True
    #)

    #D_test = 32
    #mps_test = mps_encoding(x_test, D_test)

    accuracies = []
    for D_encode in tqdm(range(24, 33, 2)):

        D = (D_encode, D_batch, 32)
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

        #for D_final in tqdm([10, 32, 100]):
        #    sum_states = list_of_classifiers

        #    classifier_data = adding_batches(sum_states, D_final, 10, orthogonalise = ortho_at_end)[0]
        #    classifier = data_to_QTN(classifier_data.data).squeeze()

        #    predictions = classifier_predictions(classifier, mps_test, bitstrings)
        #    accuracy = evaluate_classifier_top_k_accuracy(predictions, y_test, 1)

        #    accuracies.append(accuracy)
        #    np.save('results/correct_norm/mnist_non_ortho_d_encode_vs_acc_d_final_10_32_100_30_32_3', accuracies)

    assert()

def d_batch_vs_acc(q_hairy_bitstrings):
    print('SIMULATING TEST ACCURACY VS D_BATCH')
    num_samples = 5421*10
    batch_nums = [3, 13, 139, 10]
    final_batch_num = batch_nums.pop(-1)

    ortho_at_end = False
    D_encode = 32
    #D_test = 32

    x_train, y_train, x_test, y_test = load_data(
        num_samples, shuffle=False, equal_numbers=True
    )

    mps_train = mps_encoding(x_train, D_encode)
    #mps_test = mps_encoding(x_test, D_test)

    accuracies = []
    for D_batch in tqdm(range(2, 33, 2)):

        list_of_classifiers = prepare_centred_batched_classifier(
            mps_train, y_train, q_hairy_bitstrings, D_batch, batch_nums
        )

        qsum_states = [data_to_QTN(s.data) for s in list_of_classifiers]

        path = "Classifiers/mnist_mixed_sum_states/" + f"sum_states_D_batch_{D_batch}/"
        os.makedirs(path, exist_ok=True)
        [save_qtn_classifier(s , "mnist_mixed_sum_states/" + f"sum_states_D_batch_{D_batch}/" + f"digit_{i}") for i, s in enumerate(qsum_states)]

        """

        for D_final in tqdm([10, 32, 100]):
            #print(f'D_final: {D_final}')
            sum_states = list_of_classifiers

            fmpo_classifier = adding_batches(sum_states, D_final, 10, orthogonalise = ortho_at_end)[0]
            classifier = data_to_QTN(fmpo_classifier.data).squeeze()

            predictions = classifier_predictions(classifier.squeeze(), mps_test, bitstrings)
            accuracy = evaluate_classifier_top_k_accuracy(predictions, y_test, 1)

            accuracies.append(accuracy)
            np.save('results/correct_norm/mnist_non_ortho_d_batch_vs_acc_d_final_10_32_100', accuracies)
        """

    assert()

def d_final_vs_acc(bitstrings):

    n_train_samples = 5421*10
    n_test_samples = 10000

    x_train, y_train, x_test, y_test = load_data(
        n_train_samples,n_test_samples, shuffle=False, equal_numbers=True
    )

    #print('Loaded Data!')
    x_train, y_train = arrange_data(x_train, y_train, arrangement='one class')
    #print('Arranged Data!')


    # MPS encode data
    D_encode = 32
    mps_train = mps_encoding(x_train, D_encode)
    mps_test = mps_encoding(x_test, D_encode)

    """
    ###############
    qtn_classifier = load_qtn_classifier('fashion_mnist_big_classifier_D_total_32_orthogonal').squeeze()
    train_predictions = np.array([abs((mps_image.H @ qtn_classifier).squeeze().data) for mps_image in tqdm(mps_train)])
    test_predictions = np.array([abs((mps_image.H @ qtn_classifier).squeeze().data) for mps_image in tqdm(mps_test)])

    np.save('Classifiers/fashion_mnist/big_dataset_training_labels.npy', y_train)
    np.save('Classifiers/fashion_mnist/big_dataset_test_labels.npy', y_test)

    np.save('Classifiers/fashion_mnist/initial_training_predictions_ortho_mpo_classifier.npy', train_predictions)
    np.save('Classifiers/fashion_mnist/initial_test_predictions_ortho_mpo_classifier.npy', test_predictions)
    assert()
    ##############
    non_ortho_classifier = load_qtn_classifier('fashion_mnist_big_classifier_D_total_32')
    data = [site.data for site in non_ortho_classifier.tensors]
    """
    sum_states = [load_qtn_classifier(f'mnist_sum_states/sum_state_digit_{i}') for i in range(10)]
    sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in sum_states]

    non_ortho_training_accuracies = []
    ortho_training_accuracies = []
    non_ortho_test_accuracies = []
    ortho_test_accuracies = []

    #D_finals = [2, 10, 20, 32, 50, 100, 150, 200, 250, 300, 310, 320, 330, 350]
    D_finals = range(2, 34, 2)[::-1]
    for D_final in tqdm(D_finals):

        """
        non_ortho_classifier_data = adding_batches(sum_states_data, D_final, 10, orthogonalise = False)[0]
        non_ortho_mpo_classifier = data_to_QTN(non_ortho_classifier_data.data).squeeze()

        ortho_classifier_data = adding_batches(sum_states_data, D_final, 10, orthogonalise = True)[0]
        ortho_mpo_classifier = data_to_QTN(ortho_classifier_data.data).squeeze()
        """
        non_ortho_sum_states = [sum_state.compress_one_site(D = D_final, orthogonalise = False) for sum_state in sum_states_data]
        qtn_non_ortho_sum_states = [data_to_QTN(sum_state.data) for sum_state in non_ortho_sum_states]

        ortho_sum_states = [sum_state.compress_one_site(D = D_final, orthogonalise = True) for sum_state in sum_states_data]
        qtn_ortho_sum_states = [data_to_QTN(sum_state.data) for sum_state in ortho_sum_states]


        #non_ortho_truncated_classifier = fMPO(data).compress_one_site(D=D_final, orthogonalise = False)
        #ortho_truncated_classifier = fMPO(data).compress_one_site(D=D_final, orthogonalise = True)

        #non_ortho_qtn_classifier = data_to_QTN(non_ortho_truncated_classifier)
        #ortho_qtn_classifier = data_to_QTN(ortho_truncated_classifier)

        #print('Training predicitions: ')
        """
        non_ortho_training_predictions = classifier_predictions(non_ortho_mpo_classifier, mps_train, bitstrings)
        non_ortho_training_accuracy = evaluate_classifier_top_k_accuracy(non_ortho_training_predictions, y_train, 1)

        ortho_training_predictions = classifier_predictions(ortho_mpo_classifier, mps_train, bitstrings)
        ortho_training_accuracy = evaluate_classifier_top_k_accuracy(ortho_training_predictions, y_train, 1)
        """
        non_ortho_training_predictions = [[(mps_image.H.squeeze() @ s.squeeze()).norm() for s in qtn_non_ortho_sum_states] for mps_image in tqdm(mps_train)]
        non_ortho_training_accuracy = evaluate_classifier_top_k_accuracy(non_ortho_training_predictions, y_train, 1)

        ortho_training_predictions = [[(mps_image.H.squeeze() @ s.squeeze()).norm() for s in qtn_ortho_sum_states] for mps_image in tqdm(mps_train)]
        ortho_training_accuracy = evaluate_classifier_top_k_accuracy(ortho_training_predictions, y_train, 1)

        #print('Test predicitions: ')
        """
        non_ortho_test_predictions = classifier_predictions(non_ortho_mpo_classifier, mps_test, bitstrings)
        non_ortho_test_accuracy = evaluate_classifier_top_k_accuracy(non_ortho_test_predictions, y_test, 1)

        ortho_test_predictions = classifier_predictions(ortho_mpo_classifier, mps_test, bitstrings)
        ortho_test_accuracy = evaluate_classifier_top_k_accuracy(ortho_test_predictions, y_test, 1)
        """
        non_ortho_test_predictions = [[(mps_image.H.squeeze() @ s.squeeze()).norm() for s in qtn_non_ortho_sum_states] for mps_image in tqdm(mps_test)]
        non_ortho_test_accuracy = evaluate_classifier_top_k_accuracy(non_ortho_test_predictions, y_test, 1)

        ortho_test_predictions = [[(mps_image.H.squeeze() @ s.squeeze()).norm() for s in qtn_ortho_sum_states] for mps_image in tqdm(mps_test)]
        ortho_test_accuracy = evaluate_classifier_top_k_accuracy(ortho_test_predictions, y_test, 1)


        non_ortho_training_accuracies.append(non_ortho_training_accuracy)
        ortho_training_accuracies.append(ortho_training_accuracy)
        non_ortho_test_accuracies.append(non_ortho_test_accuracy)
        ortho_test_accuracies.append(ortho_test_accuracy)


        np.save('Classifiers/mnist_sum_states/sum_state_non_ortho_d_final_vs_training_accuracy', non_ortho_training_accuracies)
        np.save('Classifiers/mnist_sum_states/sum_state_ortho_d_final_vs_training_accuracy', ortho_training_accuracies)
        np.save('Classifiers/mnist_sum_states/sum_state_non_ortho_d_final_vs_test_accuracy', non_ortho_test_accuracies)
        np.save('Classifiers/mnist_sum_states/sum_state_ortho_d_final_vs_test_accuracy', ortho_test_accuracies)
    assert()

def obtain_deterministic_accuracies(bitstrings):
    n_train_samples = 5329*10
    n_test_samples = 10000
    #n_samples = 100

    x_train, y_train, x_test, y_test = load_data(
        n_train_samples,n_test_samples, shuffle=False, equal_numbers=True
    )

    #print('Loaded Data!')
    x_train, y_train = arrange_data(x_train, y_train, arrangement='one class')
    #print('Arranged Data!')


    # MPS encode data
    D_encode = 32
    mps_train = mps_encoding(x_train, D_encode)
    mps_test = mps_encoding(x_test, D_encode)


    non_ortho_training_accuracies = []
    ortho_training_accuracies = []
    non_ortho_test_accuracies = []
    ortho_test_accuracies = []
    for i in tqdm(range(34,50, 2)):
        print(f'Bond order: {i}')
        non_ortho_classifier = load_qtn_classifier(f'fashion_mnist_big_classifier_D_total_{i}')
        ortho_classifier = load_qtn_classifier(f'fashion_mnist_big_classifier_D_total_{i}_orthogonal')

        #print('Training predicitions: ')
        non_ortho_training_predictions = classifier_predictions(non_ortho_classifier.squeeze(), mps_train, bitstrings)
        non_ortho_training_accuracy = evaluate_classifier_top_k_accuracy(non_ortho_training_predictions, y_train, 1)

        ortho_training_predictions = classifier_predictions(ortho_classifier.squeeze(), mps_train, bitstrings)
        ortho_training_accuracy = evaluate_classifier_top_k_accuracy(ortho_training_predictions, y_train, 1)

        #print('Test predicitions: ')
        non_ortho_test_predictions = classifier_predictions(non_ortho_classifier.squeeze(), mps_test, bitstrings)
        non_ortho_test_accuracy = evaluate_classifier_top_k_accuracy(non_ortho_test_predictions, y_test, 1)

        ortho_test_predictions = classifier_predictions(ortho_classifier.squeeze(), mps_test, bitstrings)
        ortho_test_accuracy = evaluate_classifier_top_k_accuracy(ortho_test_predictions, y_test, 1)

        non_ortho_training_accuracies.append(non_ortho_training_accuracy)
        ortho_training_accuracies.append(ortho_training_accuracy)
        non_ortho_test_accuracies.append(non_ortho_test_accuracy)
        ortho_test_accuracies.append(ortho_test_accuracy)

        np.save('Classifiers/fashion_mnist/non_ortho_training_accuracies_2_32_4', non_ortho_training_accuracies)
        np.save('Classifiers/fashion_mnist/ortho_training_accuracies_2_32_4', ortho_training_accuracies)
        np.save('Classifiers/fashion_mnist/non_ortho_test_accuracies_2_32_4', non_ortho_test_accuracies)
        np.save('Classifiers/fashion_mnist/ortho_test_accuracies_2_32_4', ortho_test_accuracies)
    assert()

def collect_variational_classifier_results(title, mps_test, y_test):
    if title == 'cross_entropy_random':
        x = range(0, 1000, 10)
    elif title == 'abs_stoudenmire_loss_random':
        x = range(89)
    elif title == 'cross_entropy_loss_det_init_non_ortho':
        x = range(0, 941, 10)
    elif title == 'abs_stoudenmire_loss_det_init_non_ortho':
        x = range(77)


    accuracies = []
    for i in tqdm(x[::-1]):
        classifier = load_qtn_classifier(title + f'/mpo_classifier_epoch_{i}')#.squeeze()
        fmpo_classifier = fMPO([site.data for site in classifier.tensors])
        ortho_classifier = data_to_QTN((fmpo_classifier.compress_one_site(D = 32, orthogonalise = True)).data).squeeze()

        predictions = np.array([abs((mps_image.H @ ortho_classifier).squeeze().data) for mps_image in tqdm(mps_test)])
        accuracies.append(evaluate_classifier_top_k_accuracy(predictions, y_test, 1))
        print(accuracies)
        assert()
    np.save('ortho_' + title + '_test_accuracies', accuracies)

def mps_image_singular_values():
    from xmps.fMPS import fMPS
    from functools import reduce

    def get_singular_values(qtn):
        #L = qtn.num_tensors
        #middle_site = qtn.tensors[L//2 -1]
        contracted = (qtn ^ all).fuse({'t0': ('k0', 'k1', 'k2', 'k3', 'k4'), 't1': ('k5', 'k6', 'k7', 'k8', 'k9')}).squeeze()
        U, S, V = svd(contracted.data)
        return S

    def get_singular_values_fmps(ftn):
        L = ftn.L
        middle_site = ftn.data[L//2 -1]

        site_data = middle_site
        d, i, j = site_data.shape
        reshaped_data = site_data.reshape(d*i, j)

        U, S, Vd = svd(reshaped_data)
        return S

    def get_sum_states(mps_images, labels):

        def add_mpss(a, b):
            return a.add(b)

        mps_images_data = [[site.data for site in i.tensors] for i in mps_images]
        fMPSs = [fMPS(i) for i in mps_images_data]

        #shape: num_classes, num_digit_in_class
        sorted_fMPSs = [[fMPSs[i] for i in range(len(fMPSs)) if labels[i] == l] for l in list(set(labels))]

        added_mpss = [reduce(add_mpss, digits) for digits in sorted_fMPSs]

        return [i.left_canonicalise(D = 32) for i in added_mpss]

    def get_batched_sum_states(mps_images, labels, batch_num, D):

        def adding_mps_batches(list_to_add, D, batch_num, truncate=True):
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
                    result.append(reduce(add_mps_sublist, (D, sub_list)))
            return result

        def add_mps_sublist(*args):
            """
            :param args: tuple of B_D and MPOs to be added together
            """

            B_D = args[0]
            sub_list_mpos = args[1]
            N = len(sub_list_mpos)

            c = sub_list_mpos[0]

            for i in range(1, N):
                c = c.add(sub_list_mpos[i])
            return c.left_canonicalise(B_D)

        mps_images_data = [[site.data for site in i.tensors] for i in mps_images]
        fMPSs = [fMPS(i) for i in mps_images_data]

        #shape: num_classes, num_digit_in_class
        sorted_fMPSs = [[fMPSs[i] for i in range(len(fMPSs)) if labels[i] == l] for l in list(set(labels))]

        flat_fMPSs = [item for sublist in sorted_fMPSs for item in sublist]

        while len(flat_fMPSs) > 10:
            flat_fMPSs = adding_mps_batches(flat_fMPSs, D, batch_num)

        return flat_fMPSs
        """
        sum_states = prepare_batched_classifier(
            mps_train, y_train, D_batch, batch_num, prep_sum_states
            )

        sum_states = [data_to_QTN(s.compress_one_site(D=D_batch, orthogonalise = ortho_at_end).data).reindex({'s9':'t9'}) for s in sum_states]
        """

    def evalulate_sum_states(sum_states, test_data, test_labels):
        test_fmps = [fMPS([site.data for site in qtn_image.tensors]) for qtn_image in test_data]
        predictions = [[abs(state.overlap(test)) for state in sum_states] for test in tqdm(test_fmps)]
        print(evaluate_classifier_top_k_accuracy(predictions, test_labels, 1))

    num_samples = 1000#5329*10
    D = 32
    batch_num = 10#73

    x_train, y_train, x_test, y_test = load_data(
        num_samples, shuffle=False, equal_numbers=True
    )

    #fmnist = tf.keras.datasets.fashion_mnist
    #(x_train, y_train), (x_test, y_test) = fmnist.load_data()
    #test_image = x_train[0].reshape(-1)#.reshape(28,28)[M:28-M,M:28-M].reshape(-1)
    #num_pixels = test_image.shape[0]
    #L = int(np.ceil(np.log2(num_pixels)))
    #padded_image = np.pad(test_image, (0, 2 ** L - num_pixels)).reshape(2 ** (L//2) , 2 ** (L//2))
    #U, S, V = svd(padded_image)
    #plt.plot(S)
    #plt.show()

    #assert()
    #plt.imshow(test_image)
    #plt.show()
    #assert()
    #test_mps = mps_encoding(x_train[:1], D)
    #test_singular_values = get_singular_values(*test_mps)
    #plt.plot(test_singular_values)
    #plt.show()

    mps_images = mps_encoding(x_train, D)
    labels = y_train

    #sum_states = get_sum_states(mps_images, labels)
    batched_sum_states = get_batched_sum_states(mps_images, labels, batch_num, D)

    single_image_sv = get_singular_values(mps_images[0])
    batched_image_sv = get_singular_values(fMPS_to_QTN(batched_sum_states[0]))


    plt.plot(batched_image_sv, label = 'sum state')
    plt.plot(single_image_sv, label = 'single image')
    plt.ylabel('Normalised Values')
    plt.xlabel('Singular Value i')
    plt.legend()
    plt.savefig('zeros_digit_sing_vals.pdf')
    plt.show()

    assert()

    #evalulate_sum_states(batched_sum_states, mps_images, labels)
    #assert()
    single_image_singular_values = [[get_singular_values(qtn) for qtn in np.array(mps_images)[labels == l]] for l in list(set(labels))]


    #image_singular_values = [np.mean([get_singular_values(qtn) for qtn in np.array(mps_images)[labels == l]], axis = 0) for l in list(set(labels))]
    #sum_states_singular_values = [get_singular_values_fmps(ss) for ss in sum_states]
    batched_sum_states_singular_values = [get_singular_values_fmps(ss) for ss in batched_sum_states]

    plt.plot(batched_sum_states_singular_values[0], label = 'sum state')
    plt.plot(single_image_singular_values[0][0], label = 'single image')
    plt.title('Digit 0 Singular Values (Center Site)')
    plt.ylabel('Unnormalised Values')
    plt.xlabel('Singular Value i')
    plt.legend()
    plt.savefig('zeros_digit_sing_vals.pdf')
    plt.show()
    assert()
    #return single_image_singular_values[0][0], batched_sum_states_singular_values[0]

    """
    plt.show()
    #np.save('mean_img_singular_vals', np.mean(image_singular_values, axis = 0))
    #np.save('mean_sum_state_singular_vals', np.mean(sum_states_singular_values, axis = 0))
    #np.save('mean_batched_sum_state_singular_vals', np.mean(batched_sum_states_singular_values, axis = 0))
    assert()
    """

def generate_classifier_images(sum_states):

    def bitstring(k, L):
        string = bin(k)[2:].zfill(L)
        return [[1, 0] * (1 - int(i)) + [0, 1] * int(i) for i in string]

    def overlap(mps, prod_state):
        return mps.overlap(fMPS().from_product_state(prod_state))

    def state_to_image(state):
        image = (
            np.array([overlap(state, bitstring(i, 10)) for i in range(784)])
            .reshape(-1, 28)
            .real
        )
        return image

    def generate_image_from_state(state):
        from collections import Counter
        from qiskit.quantum_info import Statevector
        nshots = 10000

        flat_state = np.abs(state).reshape(-1)
        normalised_state = flat_state / np.sum(flat_state)
        """
        for _ in range(10):
            samples = Counter(np.random.choice(np.arange(len(normalised_state)), nshots, p = normalised_state))
            generated_image = np.array([samples[i]/nshots for i in np.arange(len(normalised_state))]).reshape(-1, len(state))
            plt.imshow(generated_image)
            plt.show()
        """
        samples = Counter(np.random.choice(np.arange(len(normalised_state)), nshots, p = normalised_state))
        generated_image = np.array([samples[i]/nshots for i in np.arange(len(normalised_state))]).reshape(-1, len(state))
        return generated_image


    #sum_states = [classifier.squeeze() @ b.squeeze() for b in bitstrings]
    sum_states = [fMPS([site.data for site in s.tensors]) for s in sum_states]
    #fsum_states = [fMPS().left_from_state(i.data) for i in sum_states]

    #sum_state_images = [state_to_image(s) for s in tqdm(fsum_states)]
    sum_state_images = [state_to_image(sum_states[5])]

    """
    generated_image = generate_image_from_state(sum_state_images)
    return generated_image
    """

    for i in sum_state_images:
        plt.imshow(i)
        plt.show()

    assert()

def compute_confusion_matrix(bitstrings):

    mpo_sum_states = [load_qtn_classifier(f'mnist_sum_states/sum_state_digit_{i}') for i in range(10)]
    sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in mpo_sum_states]
    mps_sum_states = [s @ b for s, b in zip(mpo_sum_states, bitstrings)]

    classifier = data_to_QTN(adding_batches(sum_states_data, 320, 10, orthogonalise = True)[0].data)
    mnist_results = []
    for i in mps_sum_states:
        state_i = (classifier @ i).squeeze()
        state_i /= state_i.norm()
        mnist_results.append(abs(state_i.data[:10]))



    mpo_sum_states = [load_qtn_classifier(f'fashion_mnist_sum_states/sum_state_digit_{i}') for i in range(10)]
    sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in mpo_sum_states]
    mps_sum_states = [s @ b for s, b in zip(mpo_sum_states, bitstrings)]

    classifier = data_to_QTN(adding_batches(sum_states_data, 320, 10, orthogonalise = False)[0].data)

    fashion_mnist_results = []
    for i in mps_sum_states:
        state_i = (classifier @ i).squeeze()
        state_i /= state_i.norm()
        fashion_mnist_results.append(abs(state_i.data[:10]))

    #a = np.load('mnist_sum_state_predictions.npy')
    #b = np.load('fashion_mnist_sum_state_predictions.npy')
    plt.figure()

    a = np.load('orthogonal_mnist_sum_state_predictions.npy')
    b = np.load('orthogonal_fashion_mnist_sum_state_predictions.npy')
    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,2)
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    #axarr[0].imshow(np.array(mnist_results).T, cmap = 'Greys')
    axarr[0].imshow(a, cmap = 'Greys')
    #axarr[1].imshow(np.array(fashion_mnist_results).T, cmap = 'Greys')
    axarr[1].imshow(b, cmap = 'Greys')
    axarr[0].set_title('MNIST')
    axarr[1].set_title('FASHION MNIST')
    axarr[0].set_xticks(range(10))
    axarr[0].set_yticks(range(10))
    axarr[0].set_xlabel('Sum state i prediction')
    axarr[0].set_ylabel('Prediction Index')
    axarr[1].set_xticks(range(10))
    axarr[1].set_yticks(range(10))
    axarr[1].set_xlabel('Sum state i prediction')
    axarr[1].set_ylabel('Prediction Index')
    #plt.savefig('orthogonal_predictions_sum_state_confusion_matrix.pdf')

    #np.save('orthogonal_mnist_sum_state_predictions', np.array(mnist_results).T)
    #np.save('orthogonal_fashion_mnist_sum_state_predictions', np.array(fashion_mnist_results).T)
    plt.show()
    assert()

    """
    mpo_sum_states = [load_qtn_classifier(f'mnist_sum_states/sum_state_digit_{i}') for i in range(10)]
    sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in mpo_sum_states]
    mps_sum_states = [s @ b for s, b in zip(mpo_sum_states, bitstrings)]

    classifier = data_to_QTN(adding_batches(sum_states_data, 320, 10, orthogonalise = True)[0].data)
    mnist_results = []
    for i in mps_sum_states:
        state_i = classifier @ i
        state_i /= state_i.norm()
        for j in mps_sum_states:
            state_j = classifier @ j
            state_j /= state_j.norm()
            mnist_results.append(abs(state_i.squeeze() @ state_j.squeeze()))



    mpo_sum_states = [load_qtn_classifier(f'fashion_mnist_sum_states/sum_state_digit_{i}') for i in range(10)]
    sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in mpo_sum_states]
    mps_sum_states = [s @ b for s, b in zip(mpo_sum_states, bitstrings)]

    classifier = data_to_QTN(adding_batches(sum_states_data, 320, 10, orthogonalise = False)[0].data)

    fashion_mnist_results = []
    for i in mps_sum_states:
        state_i = classifier @ i
        state_i /= state_i.norm()
        for j in mps_sum_states:
            state_j = classifier @ j
            state_j /= state_j.norm()
            fashion_mnist_results.append(abs(state_i.squeeze() @ state_j.squeeze()))


    plt.figure()
    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,2)
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0].imshow(np.array(mnist_results).reshape(10, -1), cmap = 'Greys')
    axarr[1].imshow(np.array(fashion_mnist_results).reshape(10, -1), cmap = 'Greys')
    axarr[0].set_title('MNIST')
    axarr[1].set_title('FASHION MNIST')
    axarr[0].set_xticks(range(10))
    axarr[0].set_yticks(range(10))
    axarr[1].set_xticks(range(10))
    axarr[1].set_yticks(range(10))
    axarr[0].set_xlabel('Sum state i')
    axarr[0].set_ylabel('Sum state j')
    axarr[1].set_xlabel('Sum state i')
    axarr[1].set_ylabel('Sum state j')
    plt.savefig('orthogonal_sum_state_confusion_matrix.pdf')
    np.save('orthogonal_mnist_sum_state_overlaps', np.array(mnist_results).T)
    np.save('orthogonal_fashion_mnist_sum_state_overlaps', np.array(fashion_mnist_results).T)
    plt.show()
    assert()
    """

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


if __name__ == "__main__":
    #single_image_sv, sum_state_sv = mps_image_singular_values()
    d_encode_vs_acc()
    assert()
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
    #for D_total in tqdm(range(16,37,2)):
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
                initialise_classifier_settings=(batch_nums, ortho_at_end),
            )
    mps_images, labels = data
    classifier, sum_states = classifiers


    #path = "Classifiers/mnist_mixed_sum_states/" + f"sum_states_D_total_{D_total}/"
    #os.makedirs(path, exist_ok=True)
    #[save_qtn_classifier(s , "mnist_mixed_sum_states/" + f"sum_states_D_total_{D_total}/" + f"digit_{i}") for i, s in enumerate(sum_states)]
    #assert()
    d_batch_vs_acc(bitstrings)
    assert()
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
