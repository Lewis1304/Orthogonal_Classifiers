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
    initialise_classifier_settings=(10, False),
    prep_sum_states = False
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
    # MPS encode data
    mps_train = mps_encoding(x_train, D_encode)

    #print('Encoded Data!')

    # Initial Classifier
    if initialise_classifier:

        batch_num, ortho_at_end = initialise_classifier_settings

        if prep_sum_states:

            sum_states = prepare_batched_classifier(
                mps_train, y_train, D_batch, batch_num, prep_sum_states
            )

            mpo_classifier = sum_states

            sum_states = [data_to_QTN(s.compress_one_site(D=D_batch, orthogonalise = ortho_at_end).data).reindex({'s9':'t9'}) for s in sum_states]

            classifier_data = adding_batches(mpo_classifier, D_final, 10)[0].compress_one_site(
                D=D_final, orthogonalise=ortho_at_end
            )
            mpo_classifier = data_to_QTN(classifier_data.data)#.squeeze()

            return (mps_train, y_train), (mpo_classifier, sum_states), q_hairy_bitstrings

        else:

            fmpo_classifier = prepare_batched_classifier(
                mps_train, y_train, D_batch, batch_num
            )

            classifier_data = fmpo_classifier.compress_one_site(
                D=D_final, orthogonalise=ortho_at_end
            )
            mpo_classifier = data_to_QTN(classifier_data.data)#.squeeze()

    else:
        # MPO encode data (already encoded as mps)
        # Has shape: # classes, mpo.shape
        old_classifier_data = prepare_batched_classifier(
            mps_train[:10], list(range(10)), 32, 10
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

def d_final_vs_acc(mps_images, labels, bitstrings):

    #x_train, y_train, x_test, y_test = load_data(
    #    100,10000, shuffle=False, equal_numbers=True
    #)
    #D_test = 32
    #mps_test = mps_encoding(x_test, D_test)

    initial_classifier = load_qtn_classifier('Big_Classifiers/non_ortho_mpo_classifier_32')
    accuracies = []
    for D_final in tqdm(range(2, 33)):
        fmpo_classifier = fMPO([site.data for site in initial_classifier])
        truncated_fmpo_classifier = fmpo_classifier.compress_one_site(D=D_final, orthogonalise=False)
        qtn_classifier = data_to_QTN(truncated_fmpo_classifier)

        #predictions = classifier_predictions(qtn_classifier.squeeze(), mps_test, bitstrings)

        predictions = np.array([abs((qtn_classifier @ i).squeeze().data) for i in tqdm(mps_images)])
        accuracy = evaluate_classifier_top_k_accuracy(predictions, labels, 1)
        accuracies.append(accuracy)
        np.save('results/non_ortho_d_final_vs_training_acc', accuracies)

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
        L = qtn.num_tensors
        middle_site = qtn.tensors[L//2 -1]

        site_data = middle_site.data.squeeze()
        d, i, j = site_data.shape
        reshaped_data = site_data.reshape(d*i, j)

        U, S, Vd = svd(reshaped_data)
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

    def evalulate_sum_states(sum_states, test_data, test_labels):
        sum_states = [s for s in sum_states]
        test_fmps = [fMPS([site.data for site in qtn_image.tensors]) for qtn_image in test_data]
        predictions = [[abs(state.overlap(test)) for state in sum_states] for test in tqdm(test_fmps)]
        print(test_labels)
        print(predictions[:10])
        #assert()



        print(evaluate_classifier_top_k_accuracy(predictions, test_labels, 1))
        assert()

    num_samples = 1000#5329*10
    D = 32
    batch_num = 10#73

    x_train, y_train, x_test, y_test = load_data(
        num_samples, shuffle=False, equal_numbers=True
    )

    mps_images = mps_encoding(x_train, D)
    labels = y_train

    #sum_states = get_sum_states(mps_images, labels)
    batched_sum_states = get_batched_sum_states(mps_images, labels, batch_num, D)


    evalulate_sum_states(batched_sum_states, mps_images, labels)
    assert()


    image_singular_values = [np.mean([get_singular_values(qtn) for qtn in np.array(mps_images)[labels == l]], axis = 0) for l in list(set(labels))]
    sum_states_singular_values = [get_singular_values_fmps(ss) for ss in sum_states]
    batched_sum_states_singular_values = [get_singular_values_fmps(ss) for ss in batched_sum_states]

    np.save('mean_img_singular_vals', np.mean(image_singular_values, axis = 0))
    np.save('mean_sum_state_singular_vals', np.mean(sum_states_singular_values, axis = 0))
    np.save('mean_batched_sum_state_singular_vals', np.mean(batched_sum_states_singular_values, axis = 0))

    assert()




if __name__ == "__main__":

    #mps_image_singular_values()
    #Biggest equal size is n_train = 5329 * 10 with batch_num = 73
    #Can use n_train = 4913 with batch_num = 17
    #num_samples = 5329*10
    #batch_num = 73
    num_samples = 1000
    batch_num = 10
    ortho_at_end = False
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
                initialise_classifier=True,
                prep_sum_states = True,
                initialise_classifier_settings=(batch_num, ortho_at_end),
            )
    mps_images, labels = data
    classifier, sum_states = classifier

    predictions = np.array([abs((mps_image.H @ classifier).squeeze().data) for mps_image in tqdm(mps_images)])
    print(evaluate_classifier_top_k_accuracy(predictions, labels, 1))

    assert()
    d_final_vs_acc(mps_images, labels, bitstrings)
    assert()


    x_train, y_train, x_test, y_test = load_data(
        100,10000, shuffle=False, equal_numbers=True
    )
    D_test = 32
    #x_test = [x_test[label == y_test][0] for label in range(10)]
    #y_test = np.array(range(10))
    mps_test = mps_encoding(x_test, D_test)

    classifier = load_qtn_classifier('Big_Classifiers/non_ortho_mpo_classifier_32').squeeze()
    test_preds = np.array([abs((mps_image.H @ classifier).squeeze().data) for mps_image in tqdm(mps_test)])
    np.save('models/initial_test_predictions_non_ortho_mpo_classifier', test_preds)

    test_accuracy = evaluate_classifier_top_k_accuracy(test_preds, y_test, 1)
    print('Initial test accuracy: ', test_accuracy)

    import tensorflow as tf
    model = tf.keras.models.load_model('models/non_ortho_big_dataset_D_32')
    trained_test_predictions = model.predict(test_preds)
    np.save('models/trained_test_predictions_non_ortho_mpo_classifier', trained_test_predictions)

    test_accuracy = evaluate_classifier_top_k_accuracy(trained_test_predictions, y_test, 1)
    print('Trained test accuracy:', test_accuracy)
    assert()

    #classical_stacking(mps_images, labels, classifier.squeeze(), bitstrings)
    n_copies = 2
    v_col = True
    #U_preds = efficent_deterministic_quantum_stacking(labels, bitstrings, n_copies, classifier, v_col=v_col)
    #U = deterministic_quantum_stacking(labels, bitstrings, n_copies, classifier, v_col=v_col)
    #U_param = parameterise_deterministic_U(U)
    #quantum_stacking_with_pennylane(0, U_preds = None)#, U_param)
    assert()
    state_preparation_pennylane(U)
    #U = deterministic_quantum_stacking(labels, bitstrings, n_copies, classifier, v_col=v_col)
    assert()
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
