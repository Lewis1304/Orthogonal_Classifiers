from variational_mpo_classifiers import *
from deterministic_mpo_classifier import *
from stacking import *
#prepare_batched_classifier, prepare_ensemble, unitary_qtn, prepare_sum_states, adding_batches, prepare_linear_classifier, linear_classifier_predictions
import os
from tqdm import tqdm

"""
Prepare Experiment
"""


def initialise_experiment(
    n_samples,
    D,
    arrangement="one class",
    truncated=False,
    one_site=False,
    initialise_classifier=False,
    initialise_classifier_settings=(10, False, False),
    prep_sum_states = False
):
    """
    int: n_samples: Number of data samples (total)
    int: D_total: Bond dimension of classifier and data
    string: arrangement: Order of training images- this matters for batch added initialisation
    bool: truncated: Whether sites with upwards indices projected onto |0> are sliced. Speeds up training.
    bool: one_site: Whether classifier/images have one site with label legs or more.
    bool: initialise_classifier: Whether classifier is initialised using batch adding procedure
    tuple: initialise_classifier_settings: (
                                            batch_num: how many images per batch,
                                            one_site_adding: images encoded and compressed as one site or not,
                                            ortho_at_end: Whether polar decomp is performed at the end or not
                                            )
    """
    D_encode, D_batch, D_final = D

    #print(f'D_encode: {D_encode}')
    #print(f'D_batch: {D_batch}')
    #print(f'D_final: {D_final}')
    # Load & Organise Data
    x_train, y_train, x_test, y_test = load_data(
        n_samples, shuffle=False, equal_numbers=True
    )

    #print('Loaded Data!')
    x_train, y_train = arrange_data(x_train, y_train, arrangement=arrangement)
    #print('Arranged Data!')

    # All possible class labels
    possible_labels = list(set(y_train))
    # Number of "label" sites
    n_hairysites = int(np.ceil(math.log(len(possible_labels), 4)))
    # Number of total sites (mps encoding)
    n_sites = int(np.ceil(math.log(x_train.shape[-1], 2)))

    # Create hairy bitstrings
    hairy_bitstrings_data = create_hairy_bitstrings_data(
        possible_labels, n_hairysites, n_sites, one_site
    )
    q_hairy_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, n_hairysites, n_sites, truncated=truncated
    )
    # MPS encode data
    mps_train = mps_encoding(x_train, D_encode)

    #print('Encoded Data!')


    # Initial Classifier
    if initialise_classifier:

        batch_num, one_site_adding, ortho_at_end = initialise_classifier_settings

        if prep_sum_states:

            sum_states = prepare_sum_states(mps_train, y_train, D_batch, batch_num, one_site=one_site_adding)
            mpo_classifier = sum_states

            sum_states = [data_to_QTN(s.compress_one_site(D=D_batch, orthogonalise = ortho_at_end).data).reindex({'s9':'t9'}) for s in sum_states]
            #classifier_data = adding_batches(sum_states, D_batch, 10)[0]
            #mpo_classifier = classifier_data
            classifier_data = adding_batches(mpo_classifier, D_final, 10)[0].compress_one_site(
                D=D_final, orthogonalise=ortho_at_end
            )
            mpo_classifier = data_to_QTN(classifier_data.data)#.squeeze()

            return (mps_train, y_train), (mpo_classifier, sum_states), q_hairy_bitstrings


        #elif not math.log(len(x_train), batch_num).is_integer():

        #    sum_states = prepare_sum_states(mps_train, y_train, D_batch, batch_num, one_site=one_site_adding)
        #    mpo_classifier = sum_states

        #    sum_states = [data_to_QTN(s.compress_one_site(D=D_batch, orthogonalise = ortho_at_end).data).reindex({'s9':'t9'}) for s in sum_states]
            #classifier_data = adding_batches(sum_states, D_batch, 10)[0]
            #mpo_classifier = classifier_data
        #    classifier_data = adding_batches(mpo_classifier, D_final, 10)[0].compress_one_site(
        #        D=D_final, orthogonalise=ortho_at_end
        #    )
        #    mpo_classifier = data_to_QTN(classifier_data.data)

        #    return (mps_train, y_train), mpo_classifier, q_hairy_bitstrings

        else:

            fmpo_classifier = prepare_batched_classifier(
                mps_train, y_train, D_batch, batch_num, one_site=one_site_adding
            )

            if one_site:
                # Works for n_sites != 1. End result is a classifier with n_site = 1.
                classifier_data = fmpo_classifier.compress_one_site(
                    D=D_final, orthogonalise=ortho_at_end
                )
                mpo_classifier = data_to_QTN(classifier_data.data)#.squeeze()
            else:
                classifier_data = fmpo_classifier.compress(
                    D=D_final, orthogonalise=ortho_at_end
                )
                mpo_classifier = data_to_QTN(classifier_data.data)#.squeeze()
    else:
        # MPO encode data (already encoded as mps)
        # Has shape: # classes, mpo.shape
        old_classifier_data = prepare_batched_classifier(
            mps_train[:10], list(range(10)), 32, 10, one_site=False
        ).compress_one_site(D=32, orthogonalise=False)
        old_classifier = data_to_QTN(old_classifier_data.data)#.squeeze()
        mpo_classifier = create_mpo_classifier_from_initialised_classifier(old_classifier).squeeze()
        #mpo_classifier = create_mpo_classifier(
        #    mps_train, q_hairy_bitstrings, seed=420, full_sized=True
        #).squeeze()

    return (mps_train, y_train), mpo_classifier, q_hairy_bitstrings

"""
Experiments
"""


def all_classes_experiment(
    mpo_classifier,
    mps_train,
    q_hairy_bitstrings,
    y_train,
    predict_func,
    loss_func,
    title):
    print(title)

    classifier_opt = mpo_classifier
    #classifier_opt = pad_qtn_classifier(mpo_classifier)
    # classifier_opt = create_mpo_classifier_from_initialised_classifier(classifier_opt, seed = 420)
    """
    initial_predictions = predict_func(classifier_opt, mps_train, q_hairy_bitstrings)

    predicitions_store = [initial_predictions]
    accuracies = [evaluate_classifier_top_k_accuracy(initial_predictions, y_train, 1)]
    # variances = [evaluate_prediction_variance(initial_predictions)]
    losses = [loss_func(classifier_opt, mps_train, q_hairy_bitstrings, y_train)]
    """

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

    #Biggest equal size is n_train = 5329 * 10 with batch_num = 73
    #Can use n_train = 4913 with batch_num = 17
    num_samples = 5329*10
    batch_num = 73
    #num_samples = 100
    #batch_num = 10

    one_site = True
    one_site_adding = False
    ortho_at_end = False
    D_batch = 32

    x_train, y_train, x_test, y_test = load_data(
        100,10000, shuffle=False, equal_numbers=True
    )
    D_test = 32
    mps_test = mps_encoding(x_test, D_test)

    accuracies = []
    for D_encode in tqdm(range(2, 33)):

        data, classifier_data, bitstrings = initialise_experiment(
                    num_samples,
                    (D_encode, D_batch, D_batch),
                    arrangement='one class',
                    truncated=True,
                    one_site=one_site,
                    initialise_classifier=True,
                    initialise_classifier_settings=(batch_num, one_site_adding, ortho_at_end),
                )
        mps_images, labels = data

        for D_final in tqdm([10, 20, 32]):

            fmpo_classifier = fMPO(classifier_data.data)
            classifier = data_to_QTN(fmpo_classifier.compress_one_site(D=D_final, orthogonalise=False))

            predictions = classifier_predictions(classifier.squeeze(), mps_test, bitstrings)
            accuracy = evaluate_classifier_top_k_accuracy(predictions, y_test, 1)

            accuracies.append(accuracy)
            np.save('d_encode_vs_acc_d_final_10_20_32', accuracies)

    assert()

def d_batch_vs_acc():

    #Biggest equal size is n_train = 5329 * 10 with batch_num = 73
    #Can use n_train = 4913 with batch_num = 17
    num_samples = 5329*10
    batch_num = 73
    #num_samples = 100
    #batch_num = 10

    one_site = True
    one_site_adding = False
    ortho_at_end = False
    D_encode = 32

    x_train, y_train, x_test, y_test = load_data(
        100,10000, shuffle=False, equal_numbers=True
    )
    D_test = 32
    mps_test = mps_encoding(x_test, D_test)

    accuracies = []
    for D_batch in tqdm(range(2, 33)):

        data, list_of_classifiers, bitstrings = initialise_experiment(
                    num_samples,
                    (D_encode, D_batch, None),
                    arrangement='one class',
                    truncated=True,
                    one_site=one_site,
                    initialise_classifier=True,
                    initialise_classifier_settings=(batch_num, one_site_adding, ortho_at_end),
                )
        mps_images, labels = data

        for D_final in tqdm([10, 20, 32]):

            sum_states = list_of_classifiers
            #Here is matters that we put D_final instead of D_batch. Since in this case
            #D_batch can be lower than D_final. I.e. D_batch >= D_final is ok. D_batch < D_final not ok.
            fmpo_classifier = adding_batches(sum_states, D_final, 10)[0]

            #2nd compress doesn't really do anything. Since classifier is compressed when all states
            #are added. Here for consistency sake.
            classifier = data_to_QTN(fmpo_classifier.compress_one_site(D=D_final, orthogonalise=False))

            predictions = classifier_predictions(classifier.squeeze(), mps_test, bitstrings)
            accuracy = evaluate_classifier_top_k_accuracy(predictions, y_test, 1)

            accuracies.append(accuracy)
            np.save('d_batch_vs_acc_d_final_10_20_32', accuracies)

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
    for i in tqdm(range(32,51)):
        print(f'Bond order: {i}')
        non_ortho_classifier = load_qtn_classifier(f'Big_Classifiers/non_ortho_mpo_classifier_{i}')
        ortho_classifier = load_qtn_classifier(f'Big_Classifiers/ortho_mpo_classifier_{i}')

        print('Training predicitions: ')
        non_ortho_training_predictions = classifier_predictions(non_ortho_classifier.squeeze(), mps_train, bitstrings)
        non_ortho_training_accuracy = evaluate_classifier_top_k_accuracy(non_ortho_training_predictions, y_train, 1)

        ortho_training_predictions = classifier_predictions(ortho_classifier.squeeze(), mps_train, bitstrings)
        ortho_training_accuracy = evaluate_classifier_top_k_accuracy(ortho_training_predictions, y_train, 1)

        print('Test predicitions: ')
        non_ortho_test_predictions = classifier_predictions(non_ortho_classifier.squeeze(), mps_test, bitstrings)
        non_ortho_test_accuracy = evaluate_classifier_top_k_accuracy(non_ortho_test_predictions, y_test, 1)

        ortho_test_predictions = classifier_predictions(ortho_classifier.squeeze(), mps_test, bitstrings)
        ortho_test_accuracy = evaluate_classifier_top_k_accuracy(ortho_test_predictions, y_test, 1)

        non_ortho_training_accuracies.append(non_ortho_training_accuracy)
        ortho_training_accuracies.append(ortho_training_accuracy)
        non_ortho_test_accuracies.append(non_ortho_test_accuracy)
        ortho_test_accuracies.append(ortho_test_accuracy)

        np.save('Classifiers/Big_Classifiers/non_ortho_training_accuracies_32_50', non_ortho_training_accuracies)
        np.save('Classifiers/Big_Classifiers/ortho_training_accuracies_32_50', ortho_training_accuracies)
        np.save('Classifiers/Big_Classifiers/non_ortho_test_accuracies_32_50', non_ortho_test_accuracies)
        np.save('Classifiers/Big_Classifiers/ortho_test_accuracies_32_50', ortho_test_accuracies)
    assert()

if __name__ == "__main__":

    #Biggest equal size is n_train = 5329 * 10 with batch_num = 73
    #Can use n_train = 4913 with batch_num = 17
    #num_samples = 5329*10
    #batch_num = 73
    num_samples = 1000
    batch_num = 10
    one_site = True
    one_site_adding = False
    ortho_at_end = True
    #prep_sum_states = False

    D_total = 32
    D_encode = D_total
    D_batch = D_total
    D_final = D_total
    D = (D_encode, D_batch, D_final)
    #deterministic_mpo_classifier_experiment(1000, 10)
    #assert()
    #for D in tqdm(range(33, 51)):
    #    print(f'Bond Dimension: {D}')
    data, classifier, bitstrings = initialise_experiment(
                num_samples,
                D,
                arrangement='one class',
                truncated=True,
                one_site=one_site,
                initialise_classifier=True,
                initialise_classifier_settings=(batch_num, one_site_adding, ortho_at_end),
            )
    mps_images, labels = data
    #classifier, sum_states = classifier
    #predictions = classifier_predictions(classifier.squeeze(), mps_images, bitstrings)
    #accuracy = evaluate_classifier_top_k_accuracy(predictions, labels, 1)
    #print('Initial Accuracy: ', accuracy)
    #quantum_stacking_with_pennylane(mps_images, labels, classifier, bitstrings, 0)
    #assert()
    #ortho_classifier = load_qtn_classifier('Big_Classifiers/ortho_mpo_classifier_32')
    #train_predictions(mps_images, labels, classifier.squeeze(), bitstrings)

    """
    x_train, y_train, x_test, y_test = load_data(
        100,100, shuffle=False, equal_numbers=True
    )
    D_test = 32
    x_test = [x_test[label == y_test][0] for label in range(10)]
    y_test = np.array(range(10))
    mps_test = mps_encoding(x_test, D_test)
    """

    #classical_stacking(mps_images, labels, classifier.squeeze(), bitstrings)
    n_copies = 3
    v_col = True
    U = efficent_deterministic_quantum_stacking(labels, bitstrings, n_copies, classifier, v_col=v_col)
    #U = deterministic_quantum_stacking(labels, bitstrings, n_copies, classifier, v_col=v_col)
    assert()
    U_param = parameterise_deterministic_U(U)
    #assert()

    quantum_stacking_with_copy_qubits(classifier, bitstrings, mps_images, labels, 1, U_param)
    assert()


    all_classes_experiment(
        classifier,
        mps_images,
        bitstrings,
        labels,
        classifier_predictions,
        cross_entropy_loss,
        "cross_entropy_random",
        #'abs_stoudenmire_loss'
    )
