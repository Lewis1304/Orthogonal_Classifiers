from variational_mpo_classifiers import *
from deterministic_mpo_classifier import prepare_batched_classifier, prepare_ensemble, unitary_qtn, prepare_sum_states, adding_batches, prepare_linear_classifier, linear_classifier_predictions
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

        """
        sum_states = prepare_sum_states(mps_train, y_train, D_batch, batch_num, one_site=one_site_adding)
        print('Added Data!')
        mpo_classifier = sum_states

        #classifier_data = adding_batches(sum_states, D_batch, 10)[0]
        #mpo_classifier = classifier_data
        classifier_data_non_ortho = adding_batches(sum_states, D_final, 10)[0].compress_one_site(
            D=D_final, orthogonalise=False
        )

        classifier_data_ortho = adding_batches(sum_states, D_final, 10)[0].compress_one_site(
            D=D_final, orthogonalise=True
        )
        non_ortho_mpo_classifier = data_to_QTN(classifier_data_non_ortho.data)
        ortho_mpo_classifier = data_to_QTN(classifier_data_ortho.data)

        save_qtn_classifier(non_ortho_mpo_classifier, f'Big_Classifiers/non_ortho_mpo_classifier_{D_final}')
        save_qtn_classifier(ortho_mpo_classifier, f'Big_Classifiers/ortho_mpo_classifier_{D_final}')
        """
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
Experiment
"""


def all_classes_experiment(
    mpo_classifier,
    mps_train,
    q_hairy_bitstrings,
    y_train,
    predict_func,
    loss_func,
    title,
):
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

"""
Results
"""


def plot_results(results, title):

    accuracies, losses, predictions = results

    os.makedirs("results/" + title, exist_ok=True)

    np.save("results/" + title + "/accuracies", accuracies)
    np.save("results/" + title + "/losses", losses)
    # np.save('results/' + title + '_variances', variances)
    # np.save("results/" + title + "/predictions", predictions)

    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(losses, color=color, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color)  # we already handled the x-label with ax1
    ax2.plot(accuracies, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    fig.suptitle(title)
    plt.savefig("results/" + title + "/acc_loss_fig.pdf")
    plt.close(fig)


def plot_acc_before_ortho_and_after():

    different_classifiers = [
        # "one_site_false_ortho_at_end_false_weird_loss",
        "one_site_false_ortho_at_end_true_weird_loss",
        "one_site_true_ortho_at_end_false_weird_loss",
        "one_site_true_ortho_at_end_true_weird_loss",
        "random_one_site_false_ortho_at_end_false_weird_loss",
    ]
    #different_names = ["2 sites\northo", "1 site\nnon_ortho", "1 site\northo", "random"]
    # different_names = ["2 sites\nnon_ortho", "2 sites\northo", "1 site\nnon_ortho", "1 site\northo", "random"]
    different_names = ['Rand. init.\n- Not trained', 'Rand. init.\n- Trained', 'Batched init.\n- Not trained', 'Batched init.\n- Trained']
    # results_og = [0.826, 0.821, 0.827, 0.821, 0.786]
    # results_ortho = [0.802, 0.805, 0.805, 0.808, 0.795]

    #results_og = [0.864, 0.828, 0.869, 0.141]
    #results_ortho = [0.864, 0.804, 0.855, 0.124]

    results_og = [0.1, 0.827, 0.816, 0.869]
    results_ortho = [0.1, 0.805, 0.803, 0.855]
    # for c in different_classifiers:
    #    og, orth = svd_classifier(c, mps_images, labels)
    #    results_og.append(og)
    #    results_ortho.append(orth)

    fig, ax1 = plt.subplots()

    # ax1.axhline(0.95, linestyle = 'dashed', color = 'grey', label = 'Stoudenmire: D=10')
    # legend_1 = ax1.legend(loc = 'lower right')
    # legend_1.remove()
    ax1.grid(zorder=0.0, alpha=0.4)
    ax1.set_xlabel("Different Initialisations", labelpad=10)
    ax1.set_ylabel("Top 1- Training Accuracy")  # , color = 'C0')
    ax1.bar(
        np.arange(len(results_og)) - 0.2,
        np.round(results_og, 3),
        0.4,
        color="C0",
        label="Non-orthogonal",
        zorder=3,
    )
    ax1.bar(
        np.arange(len(results_ortho)) + 0.2,
        np.round(results_ortho, 3),
        0.4,
        color="C1",
        label="Orthogonal",
        zorder=3,
    )

    legend_1 = ax1.legend(loc="upper left")

    # ax1.tick_params(axis="y", labelcolor='C0')
    # ax1.set_xlim([1.75,10.25])
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%g'))

    # ax1.set_xticks(np.arange(2, 11, 1) )

    ax1.set_xticks(np.arange(0, len(results_og), 1))
    # ax1.set_xticklabels(different_names[:len(results_og)])

    ax1.set_xticklabels(different_names)
    # plt.savefig("different_initialisations_test.pdf")
    #ax1.set_yscale("log")
    ax1.set_yticks(np.arange(0.1, 1.1, 0.1))
    # ax1.ticklabel_format(useOffset=False)
    # ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    # ax1.yaxis.set_minor_formatter(NullFormatter())

    #plt.savefig("new_cost_function.pdf", bbox_inches="tight")
    plt.savefig('pp_fig.pdf')
    plt.show()


def plot_deterministic_mpo_classifier_results():
    random_arrangement = np.load(
        "results/one_site_vs_many_and_ortho_vs_non_ortho/one_site_false_ortho_at_end_false.npy"
    )
    one_class_arrangement = np.load(
        "results/one_site_vs_many_and_ortho_vs_non_ortho/one_site_false_ortho_at_end_true.npy"
    )

    x = list(range(2, 52, 2))

    plt.plot(x, random_arrangement, label="Non-orthogonal")
    plt.plot(x, one_class_arrangement, label="Orthogonal")
    #plt.plot(x, one_of_each_arrangement, label="one of each class batched")

    plt.xlim([2, 40])

    #plt.xlabel("$D_{total}$")
    plt.xlabel("Classifier bond dimension")
    plt.ylabel("Top-1 training accuracy")
    plt.legend()
    #plt.title("n_samples = 1000, Multiple label site, Non-orthogonal, batch_num = 10")
    #plt.savefig("results/different_arrangements/train_acc_vs_D_total.pdf")
    plt.savefig('pp_fig1.pdf')
    plt.show()


def plot_deterministic_initialisation_different_cost_function_results():
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    different_initialisations = [
        "one_site_false_ortho_at_end_true_weird_loss",
        "one_site_true_ortho_at_end_false_weird_loss",
        "one_site_true_ortho_at_end_true_weird_loss",
        "random_one_site_false_ortho_at_end_false_weird_loss",
    ]
    # loss_func_list = ['green_loss', 'abs_green_loss', 'mse_loss', 'abs_mse_loss', 'cross_entropy_loss', 'stoudenmire_loss', 'abs_stoudenmire_loss']
    different_names = ["2 sites\northo", "1 site\nnon_ortho", "1 site\northo", "random"]

    for initialisation in different_initialisations:
        results = []
        result = np.load("results/" + initialisation + "/accuracies.npy")
        av = moving_average(result, 2)
        plt.plot(range(1, len(av) + 1), av, label=initialisation)

    plt.xlim([0, 200])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    plt.xlabel("Epoch")
    plt.ylabel("Top-1 Accuracy")
    # plt.legend(prop={'size': 8})
    plt.tight_layout()
    plt.grid(alpha=0.4)
    plt.title("New Cost Function Training Results")
    plt.savefig("figures/" + "different_cost_function_training.pdf")
    plt.show()

    assert ()


def plot_padding_results():
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    different_initialisations = [
        "not_full_sized_random_one_site_false",
        "not_full_sized_random_one_site_true",
        "full_sized_random_one_site_false",
        "full_sized_random_one_site_true",
    ]

    for initialisation in different_initialisations:
        result = np.load("results/" + initialisation + "/accuracies.npy")
        av = moving_average(result, 2)
        plt.plot(range(1, len(av) + 1), av, label=initialisation)

    # plt.ylim([0.75,0.83])
    plt.xlim([0, 800])
    plt.xlabel("Epoch")
    plt.ylabel("Top-1 Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=0.4)
    plt.title("Padded vs Non-Padded")
    plt.savefig("figures/" + "padded_vs_non_padded.pdf")
    plt.show()

    assert ()

def tracing_over(classifier, bitstrings, mps_images, labels):

    """
    Unitaryfying the TN
    """
    uclassifier = unitary_qtn(classifier)
    bitstrings_data = create_padded_bitstrings_data(list(set(labels)), uclassifier)

    padded_bitstrings = padded_bitstring_data_to_QTN(
        bitstrings_data, uclassifier)

    print(uclassifier)
    print(padded_bitstrings[0][0])
    predictions = padded_classifier_predictions(uclassifier.squeeze(), mps_images, padded_bitstrings[:1])
    print(evaluate_classifier_top_k_accuracy(predictions, labels, 1))
    assert()

def quantum_stacking(classifier, bitstrings, mps_images, labels, loss_func, loss_name):
    import quimb as qu
    import quimb.tensor as qtn

    def ansatz_circuit(n, depth, gate2='CX', **kwargs):
        """Construct a circuit of single qubit and entangling layers.
        """
        def single_qubit_layer(circ, gate_round=None):
            """Apply a parametrizable layer of single qubit ``U3`` gates.
            """
            for i in range(circ.N):
                # initialize with random parameters
                #params = qu.randn(3, dist='uniform')
                params = np.zeros(3)
                circ.apply_gate(
                    'U3', *params, i,
                    gate_round=gate_round, parametrize=True)

        def two_qubit_layer(circ, gate2='CX', reverse=False, gate_round=None):
            """Apply a layer of constant entangling gates.
            """
            regs = range(0, circ.N - 1)
            if reverse:
                regs = reversed(regs)

            for i in regs:
                circ.apply_gate(
                    gate2, i, i + 1, gate_round=gate_round)

        circ = qtn.Circuit(n, **kwargs)

        for r in range(depth):
            # single qubit gate layer
            single_qubit_layer(circ, gate_round=r)

            # alternate between forward and backward CZ layers
            two_qubit_layer(
                circ, gate2=gate2, gate_round=r, reverse=r % 2 == 0)

        # add a final single qubit layer
        single_qubit_layer(circ, gate_round=r + 1)

        return circ

    def ansatz_circuit_2(n, depth, **kwargs):
        """Construct a circuit of SU4s
        """

        def single_qubit_layer(circ, gate_round=None):
            """Apply a parametrizable layer of single qubit ``U3`` gates.
            """
            for i in range(circ.N):
                # initialize with random parameters
                #params = np.zeros(3)
                params = qu.randn(3, dist='uniform')
                circ.apply_gate(
                    'U3', *params, i,
                    gate_round=gate_round, parametrize=True)


        def two_qubit_layer(circ, gate_round=None):
            """Apply a layer of constant entangling gates.
            """
            regs = range(0, circ.N - 1)
            params = qu.randn(15, dist='uniform')
            #params = np.zeros(15)

            for i in regs:
                circ.apply_gate(
                    'su4',*params, i, i + 1, gate_round=gate_round, parametrize = True, contract = False)

        circ = qtn.Circuit(n, **kwargs)

        for r in range(depth):
            # single qubit gate layer
            #single_qubit_layer(circ, gate_round=r)

            # alternate between forward and backward CZ layers
            two_qubit_layer(
                circ, gate_round=r)

        # add a final single qubit layer
        single_qubit_layer(circ, gate_round=r + 1)

        return circ

    def overlap(predicition, V, bitstring):
        return (predicition.squeeze().H @ (V @ bitstring.squeeze())).norm()#**2

    def optimiser(circ):
        optmzr = qtn.TNOptimizer(
                circ,  # our initial input, the tensors of which to optimize
                loss_fn = lambda V: loss(V, qtn_prediction_and_ancillae_qubits, qtn_bitstrings, labels),
                norm_fn=normalize_tn,
                autodiff_backend="autograd",  # {'jax', 'tensorflow', 'autograd'}
                optimizer="nadam",  # supplied to scipy.minimize
                )
        return optmzr

    #Reshape label site into qubit states
    n_label_qubits = 4
    n_ancillae =  n_label_qubits
    n_total = n_ancillae + n_label_qubits

    bitstrings_qubits = [bitstring.squeeze().tensors[-1].data.reshape(2,2,2,2) for bitstring in bitstrings]
    qtn_bitstrings = [qtn.Tensor(bitstring_qubits, inds = [f'b{n_total-4}', f'b{n_total-3}', f'b{n_total-2}', f'b{n_total-1}']) for bitstring_qubits in bitstrings_qubits]


    #Ancillae start in state |00...>
    ancillae_qubits = np.eye(2**n_ancillae)[0]
    #ancillae_qubits = np.ones(2**n_ancillae) / (2**(n_ancillae/2))
    #Tensor product ancillae with predicition qubits
    #Amount of ancillae equal to amount of predicition qubits
    qtn_prediction_and_ancillae_qubits = [qtn.Tensor(np.kron(ancillae_qubits, (mps_image.H @ classifier).squeeze().data).reshape(*[2]*n_total), inds = [f'k{i}' for i in range(n_total)]) for mps_image in mps_images]
    #Normalise predictions
    qtn_prediction_and_ancillae_qubits = [i/(i.H @ i)**0.5 for i in qtn_prediction_and_ancillae_qubits]


    circ = ansatz_circuit(n_total, n_total)
    V = circ.uni
    #V.draw(color=[f'ROUND_{i}' for i in range(n_total + 1)], show_inds=True, show_tags = False)
    #V.draw(color=['U3', 'SU4'], show_inds=True)

    #test = (V.contract(all) & qtn_bitstrings[0]) & qtn_prediction_and_ancillae_qubits[0]
    #print(test.contract(all).data)
    #test.draw(show_tags = False)
    #print(qtn_prediction_and_ancillae_qubits[0].H @ qtn_prediction_and_ancillae_qubits[0])
    #print(test.H @ test)
    print(f'Loss function: {loss_name}')

    predictions = [[overlap(p, V, b) for b in qtn_bitstrings] for p in qtn_prediction_and_ancillae_qubits]
    #print(predictions)
    accuracies = [evaluate_classifier_top_k_accuracy(predictions, labels, 1)]
    losses = [loss_func(V, qtn_prediction_and_ancillae_qubits, qtn_bitstrings, labels)]
    print(f'Accuracy before: {accuracies[0]}')
    assert()
    for _ in tqdm(range(1)):

        tnopt = qtn.TNOptimizer(
        V,                        # the tensor network we want to optimize
        loss_func,                     # the function we want to minimize
        loss_kwargs = {'mps_train': qtn_prediction_and_ancillae_qubits,
                        'q_hairy_bitstrings': qtn_bitstrings,
                        'y_train': labels},
        tags=['U3'],              # only optimize U3 tensors
        autodiff_backend='autograd',   # use 'autograd' for non-compiled optimization
        optimizer='nadam',     # the optimization algorithm
        )

        V = tnopt.optimize_basinhopping(n=500, nhop=10)

        losses.append(tnopt.loss)
        predictions = [[overlap(p, V, b) for b in qtn_bitstrings] for p in qtn_prediction_and_ancillae_qubits]
        accuracies.append(evaluate_classifier_top_k_accuracy(predictions, labels, 1))
        np.save(f'losses_{loss_name}_su4', losses)
        np.save(f'accuracies_{loss_name}_su4', accuracies)


    #predictions = [[overlap(p, V_opt, b) for b in qtn_bitstrings] for p in qtn_prediction_and_ancillae_qubits]
    #accuracies = [evaluate_classifier_top_k_accuracy(predictions, labels, 1)]
    #print(f'Accuracy after: {accuracies[0]}')

def quantum_stacking_2(classifier, bitstrings, mps_images, labels):
    import quimb as qu
    import quimb.tensor as qtn

    """
    Alternating U3 and CX Gates
    Depth inc. one layer of each
    """
    def ansatz_circuit(n, depth, gate2='CX', **kwargs):
        """Construct a circuit of single qubit and entangling layers.
        """
        def single_qubit_layer(circ, gate_round=None):
            """Apply a parametrizable layer of single qubit ``U3`` gates.
            """
            for i in range(circ.N):
                # initialize with random parameters
                #params = qu.randn(3, dist='uniform')
                params = np.zeros(3)
                circ.apply_gate(
                    'U3', *params, i,
                    gate_round=gate_round, parametrize=True)

        def two_qubit_layer(circ, gate2='CX', reverse=False, gate_round=None):
            """Apply a layer of constant entangling gates.
            """
            regs = range(0, circ.N - 1)
            if reverse:
                regs = reversed(regs)

            for i in regs:
                circ.apply_gate(
                    gate2, i, i + 1, gate_round=gate_round)

        circ = qtn.Circuit(n, **kwargs)

        for r in range(depth):
            # single qubit gate layer
            single_qubit_layer(circ, gate_round=r)

            # alternate between forward and backward CZ layers
            two_qubit_layer(
                circ, gate2=gate2, gate_round=r, reverse=r % 2 == 0)

        # add a final single qubit layer
        single_qubit_layer(circ, gate_round=r + 1)

        return circ

    """
    General SU4 gate layers. W/ U3 layer at end
    """
    def ansatz_circuit_2(n, depth, **kwargs):
        """Construct a circuit of SU4s
        """

        def single_qubit_layer(circ, gate_round=None):
            """Apply a parametrizable layer of single qubit ``U3`` gates.
            """
            for i in range(circ.N):
                # initialize with random parameters
                params = np.zeros(3)
                #params = qu.randn(3, dist='uniform')
                circ.apply_gate(
                    'U3', *params, i,
                    gate_round=gate_round, parametrize=True)


        def two_qubit_layer(circ, gate_round=None):
            """Apply a layer of constant entangling gates.
            """
            regs = range(0, circ.N - 1)
            #params = qu.randn(15, dist='uniform')
            params = np.zeros(15)

            for i in regs:
                circ.apply_gate(
                    'su4',*params, i, i + 1, gate_round=gate_round, parametrize = True, contract = False)

        circ = qtn.Circuit(n, **kwargs)

        for r in range(depth):
            two_qubit_layer(
                circ, gate_round=r)

        # add a final single qubit layer
        single_qubit_layer(circ, gate_round=r + 1)

        return circ

    """
    Overlap between:
    label qubits and copies (predicition)
    Circuit (V)
    Bitstring, with post-selection of copies on |000..>.
    If tracing over copy states, need to include norm()
    """
    def overlap(predicition, V, bitstring):
        return (predicition.squeeze().H @ (V @ bitstring.squeeze())).norm()#**2

    """
    Cross Entropy Loss function.
    Predictions are normalized
    """
    def loss_func(classifier, mps_train, q_hairy_bitstrings, y_train):
        #im_ov = [(p.squeeze().H @ classifier) for p in mps_train]
        #predictions = [[(pv @ b.squeeze()).norm() for b in q_hairy_bitstrings] for pv in im_ov]
        predictions = [[overlap(p, classifier, b) for b in q_hairy_bitstrings] for p in mps_train]
        n_predicitions = [i/anp.sqrt(np.dot(i,i)) for i in predictions]
        overlaps = [anp.log(p[i]) for p, i in zip(predictions, y_train)]
        return -np.sum(overlaps) / len(mps_train)

    """
    Parameters for experiment.
    n_copies = total number of copy states.
    """
    n_label_qubits = int(np.log2(bitstrings[0].squeeze().tensors[-1].data.shape)[0])
    n_copies =  2
    n_total = (n_copies * n_label_qubits) + n_label_qubits

    """
    Post select copy qubits in |000..> state.
    """
    #ancillae_qubits = np.eye(2**(n_copies * n_label_qubits))[0]
    #bitstrings_qubits = [np.kron(ancillae_qubits, bitstring.squeeze().tensors[-1].data).reshape(*[2]*n_total) for bitstring in bitstrings]
    #qtn_bitstrings = [qtn.Tensor(bitstring_qubits, inds = [f'b{i}' for i in range(n_total)]) for bitstring_qubits in bitstrings_qubits]

    """
    Trace over all copy qubits state
    """
    bitstrings_qubits = [bitstring.squeeze().tensors[-1].data.reshape(*[2]*n_label_qubits) for bitstring in bitstrings]
    qtn_bitstrings = [qtn.Tensor(bitstring_qubits, inds = [f'b{n_total-4}', f'b{n_total-3}', f'b{n_total-2}', f'b{n_total-1}']) for bitstring_qubits in bitstrings_qubits]

    """
    Generate prediction states
    """
    prediction_qubits = [(mps_image.H @ classifier).squeeze().data for mps_image in mps_images]
    """
    Normalise predictions
    """
    prediction_qubits = [i/np.sqrt(np.dot(i,i)) for i in prediction_qubits]


    initial_incorrect_predictions = prediction_qubits#[prediction_qubits[6], prediction_qubits[8], prediction_qubits[9]]
    incorrect_labels = labels#[labels[6], labels[8], labels[9]]

    np.save('initial_incorrect_predictions', initial_incorrect_predictions)
    np.save('incorrect_labels', incorrect_labels)


    """
    Construct input state.
    I.e. tensor product label qubit with copies
    """
    prediction_and_copy_qubits = prediction_qubits
    for k in range(n_copies):
        prediction_and_copy_qubits = [np.kron(i, j) for i,j in zip(prediction_and_copy_qubits, prediction_qubits)]
    qtn_prediction_and_copy_qubits = [qtn.Tensor(j.reshape(*[2]*n_total), inds = [f'k{i}' for i in range(n_total)]) for j in prediction_and_copy_qubits]

    """
    #Log depth. circuit
    """
    depth = int(np.ceil(np.log2(n_total)))
    """
    #Round to nearest power of 2. (means identity gives original result otherwise bitflips)
    """
    if depth % 2 != 0:
        depth += 1

    circ = ansatz_circuit_2(n_total, depth)
    V = circ.get_uni(transposed=True)
    #V.draw(color=[f'ROUND_{i}' for i in range(n_total + 1)], show_inds=True, show_tags = False)
    #V.draw(color=['U3', 'CX'], show_inds=True)

    """
    Collect predictions from circuit.
    Should be same as initial, since V is identity
    """
    predictions = [[overlap(p, V, b) for b in qtn_bitstrings] for p in qtn_prediction_and_copy_qubits]
    """
    Normalise predicitions (after measurement). Doesn't affect accuracy
    """
    predictions = [i/np.sqrt(np.dot(i,i)) for i in predictions]

    #print(np.round(predictions[6],5))
    #print(predictions)
    """
    Initial accuracy & loss
    """
    accuracies = [evaluate_classifier_top_k_accuracy(predictions, labels, 1)]
    losses = [loss_func(V, qtn_prediction_and_copy_qubits, qtn_bitstrings, labels)]
    print(f'Accuracy before: {accuracies[0]}')

    for _ in tqdm(range(1)):

        tnopt = qtn.TNOptimizer(
        V,                        # the tensor network we want to optimize
        loss_func,                     # the function we want to minimize
        loss_kwargs = {'mps_train': qtn_prediction_and_copy_qubits,
                        'q_hairy_bitstrings': qtn_bitstrings,
                        'y_train': labels},
        tags=['U3'],              # only optimize U3 tensors
        autodiff_backend='autograd',   # use 'autograd' for non-compiled optimization
        optimizer='nadam',     # the optimization algorithm
        )

        V = tnopt.optimize_basinhopping(n=200, nhop=10)

        losses.append(tnopt.losses)
        predictions = [[overlap(p, V, b) for b in qtn_bitstrings] for p in qtn_prediction_and_copy_qubits]
        predictions = [i/np.sqrt(np.dot(i,i)) for i in predictions]
        accuracies.append(evaluate_classifier_top_k_accuracy(predictions, labels, 1))

        np.save('variational_incorrect_predictions_test_3', predictions)
        np.save(f'quantum_stacking_2_losses_test_3', losses)
        np.save(f'quantum_stacking_2_accuracies_test_3', accuracies)


    #predictions = [[overlap(p, V_opt, b) for b in qtn_bitstrings] for p in qtn_prediction_and_ancillae_qubits]
    #accuracies = [evaluate_classifier_top_k_accuracy(predictions, labels, 1)]
    #print(f'Accuracy after: {accuracies[0]}')

def deterministic_quantum_stacking(y_train, bitstrings, n_copies):

    #Shape: n_train,2**label_qubits
    initial_label_qubits= np.load('initial_label_qubit_states_2.npy')
    #Shape: n_train, n_classes
    variational_label_qubits = np.load('final_label_qubit_states_2.npy')
    # No Need to project onto bitstring states, as evaluate picks highest
    # Which corresponds to bitstring state anyway
    #initial_preds = [[abs(i @ b.squeeze().tensors[-1].data) for b in bitstrings] for i in initial_label_qubits]

    print('Accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_train, 1))
    print('Accuracy After:', evaluate_classifier_top_k_accuracy(variational_label_qubits, y_train, 1))

    #print('Initial label qubits shape:', initial_label_qubits.shape)
    dim_l = initial_label_qubits.shape[1]
    outer_ket_states = initial_label_qubits

    #n_copies = 2
    dim_lc = dim_l ** (1 + n_copies)

    #.shape = n_train, dim_l**n_copies+1
    for k in range(n_copies):
        outer_ket_states = np.array([np.kron(i, j) for i,j in zip(outer_ket_states, initial_label_qubits)])
    """
    #.shape = n_train, dim_l**n_copies+1, dim_l**n_copies+1
    outer_states = np.array([np.outer(i.conj().T, i) for i in outer_ket_states])
    print('Outer label qubits shape:', outer_states.shape)

    #.shape = n_classes, dim_l**n_copies+1
    weighted_summed_states = np.zeros((len(set(y_train)), dim_lc, dim_lc), dtype = np.complex128)
    for i in tqdm(range(len(outer_states))):
        weighted_outer_states = []
        for fl in variational_label_qubits[i]:
            weighted_outer_states.append(outer_states[i] * fl)
        weighted_summed_states += weighted_outer_states
    #weighted_summed_states_inefficent = np.sum([[ outer_states[i] * fl for fl in variational_label_qubits[i]] for i in range(len(outer_states))], axis = 0)
    """
    weighted_summed_states = np.zeros((len(set(y_train)), dim_lc, dim_lc))
    for i in tqdm(range(1000)):
        weighted_outer_states = []

        for fl in variational_label_qubits[i]:
            ket = initial_label_qubits[i]

            for k in range(n_copies):
                ket = np.kron(ket, initial_label_qubits[i])
            outer = np.outer(ket, ket)
            weighted_outer_states.append(outer * fl)
        weighted_summed_states += weighted_outer_states

    from xmps.svd_robust import svd
    from scipy.linalg import polar
    #U and S.shape = dim_l**n_copies+1, dim_l**n_copies+1
    print('Performing SVDs!')
    USs = [svd(i)[:2] for i in tqdm(weighted_summed_states)]

    #V.shape = dim_l**n_copies+1 , dim_l
    V = np.array([i[0][:, :1] @ np.sqrt(np.diag(i[1])[:1, :1]) for i in USs]).squeeze().T#, axis = 0)

    print('Performing Polar Decomposition!')
    U = polar(V)[0]

    print('Performing Contractions!')
    preds_U = np.array([abs(i @ U) for i in outer_ket_states])
    preds_U = np.array([i / np.sqrt(i @ i) for i in preds_U])
    preds_V = np.array([abs(i @ V) for i in outer_ket_states])
    preds_V = np.array([i / np.sqrt(i @ i) for i in preds_V])

    print('Accuracy V:', evaluate_classifier_top_k_accuracy(preds_V, y_train, 1))
    print('Accuracy U:', evaluate_classifier_top_k_accuracy(preds_U, y_train, 1))
    """
    for i in range(10):
        plt.bar(range(10), preds_U[i], color = 'tab:blue', label = 'ortho')
        plt.bar(range(10), preds_V[i], fill = False, edgecolor = 'tab:orange', linewidth = 1.5, label = 'non_ortho')
        plt.legend()
        plt.show()
    """

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

def acc_vs_d_total_figure():
    non_ortho_training_accuracies = np.load('Classifiers/Big_Classifiers/non_ortho_training_accuracies.npy')
    ortho_training_accuracies = np.load('Classifiers/Big_Classifiers/ortho_training_accuracies.npy')
    non_ortho_test_accuracies = np.load('Classifiers/Big_Classifiers/non_ortho_test_accuracies.npy')
    ortho_test_accuracies = np.load('Classifiers/Big_Classifiers/ortho_test_accuracies.npy')

    non_ortho_training_accuracies2 = np.load('Classifiers/Big_Classifiers/non_ortho_training_accuracies_32_50.npy')[1:]
    ortho_training_accuracies2 = np.load('Classifiers/Big_Classifiers/ortho_training_accuracies_32_50.npy')[1:]
    non_ortho_test_accuracies2 = np.load('Classifiers/Big_Classifiers/non_ortho_test_accuracies_32_50.npy')[1:]
    ortho_test_accuracies2 = np.load('Classifiers/Big_Classifiers/ortho_test_accuracies_32_50.npy')[1:]

    non_ortho_training_accuracies = np.append(non_ortho_training_accuracies, non_ortho_training_accuracies2)
    ortho_training_accuracies = np.append(ortho_training_accuracies, ortho_training_accuracies2)
    non_ortho_test_accuracies = np.append(non_ortho_test_accuracies, non_ortho_test_accuracies2)
    ortho_test_accuracies = np.append(ortho_test_accuracies, ortho_test_accuracies2)


    x = range(2,len(non_ortho_training_accuracies)+2)
    plt.plot(non_ortho_training_accuracies, linestyle = 'dashed', color = 'tab:blue')
    plt.plot(ortho_training_accuracies, linestyle = 'dashed', color = 'tab:orange')
    plt.plot(non_ortho_test_accuracies, color = 'tab:blue')
    plt.plot(ortho_test_accuracies, color = 'tab:orange')
    plt.plot([],[],linestyle = 'dashed', color = 'grey', label = 'Training Accuracy')
    plt.plot([],[],linestyle = 'solid', color = 'grey', label = 'Test Accuracy')
    plt.plot([],[],linewidth = 0, marker = '.', markersize = 12, color = 'tab:blue', label = 'Non-orthogonal')
    plt.plot([],[],linewidth = 0, marker = '.', markersize = 12, color = 'tab:orange', label = 'Orthogonal')
    plt.xlabel('$D_{total}$')
    plt.ylabel('$Accuracy$')
    plt.xscale('log')
    plt.grid(alpha = 0.6)
    plt.legend()
    #plt.savefig('accuracy_vs_D_total.pdf')
    plt.show()
    assert()

def acc_vs_d_encode_d_batch():
    d_batch_accuracies = np.load('d_batch_vs_acc_d_final_10_20_32.npy').reshape(-1,3).T
    d_encode_accuracies = np.load('d_encode_vs_acc_d_final_10_20_32.npy').reshape(-1,3).T

    x = range(2, 33)
    """
    for d_final, accuracy in zip([10, 20, 32], d_batch_accuracies):
        plt.plot(x, accuracy, label = '$D_{final}'+ f'= {d_final}$')

    plt.legend()
    plt.show()
    """
    #fig, axs = plt.subplots(2)
    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0.3)
    axs = gs.subplots()


    for d_final, b_accuracy, e_accuracy in zip([10, 20, 32], d_batch_accuracies, d_encode_accuracies):
        axs[0].plot(x, b_accuracy, label = '$D_{final}'+ f'= {d_final}$')
        axs[1].plot(x, e_accuracy, label = '$D_{final}'+ f'= {d_final}$')
    axs[0].legend()
    axs[0].set_xlabel('$D_{batch}$')
    axs[1].set_xlabel('$D_{encode}$')
    axs[0].grid(alpha = 0.8)
    axs[1].grid(alpha = 0.8)
    #plt.ylabel('Test Accuracy')
    gs = fig.add_gridspec(3, hspace=0)

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.ylabel("Test Accuracy", labelpad = 10)
    plt.savefig('acc_vs_d_batch_d_encode.pdf')
    plt.show()
    assert()

def prediction_weights():
    incorrect_labels = np.load('incorrect_labels.npy')
    initial_incorrect_predictions = np.load('initial_incorrect_predictions.npy')

    variational_predictions = np.load('variational_incorrect_predictions_test_2.npy')

    fig = plt.figure(figsize=(10,8))
    a1 = fig.add_subplot(521)
    a2 = fig.add_subplot(522)
    a3 = fig.add_subplot(523)
    a4 = fig.add_subplot(524)
    a5 = fig.add_subplot(525)
    a6 = fig.add_subplot(526)
    a7 = fig.add_subplot(527)
    a8 = fig.add_subplot(528)
    a9 = fig.add_subplot(529)
    a10 = fig.add_subplot(5,2,10)

    axes = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]


    for i, ax in enumerate(axes):
        ax.bar(range(10), initial_incorrect_predictions[i][:10], label = 'Initial')
        ax.bar(range(10), variational_predictions[i], fill = False, label = 'Variational', edgecolor = 'tab:orange', linewidth = 1.5)
        ax.set_title(f'Correct Digit: {incorrect_labels[i]}')
        ax.tick_params(axis = 'both',which='both', bottom=True, top=False,labelbottom=True)
        ax.set_xticks(range(10))

    a9.tick_params(axis = 'both',which='both', bottom=True, top=False,labelbottom=True)
    a9.set_xlabel('Digit')
    a9.set_ylabel('Predicition Weight')
    a9.legend(loc = 'upper left')
    fig.tight_layout()
    plt.savefig('prediction_weights_before_after.pdf')
    plt.show()



if __name__ == "__main__":

    """

    losses = np.load('quantum_stacking_2_losses_test_3.npy', allow_pickle=True)
    plt.plot(losses[1])
    plt.show()
    assert()

    #prediction_weights()
    #assert()

    data = np.load('accuracies_cross_entropy_loss_L-BFGS-B_log_depth_8_ancillae_basin_hopping.npy')
    #data2 = np.load('accuracies_cross_entropy_loss_L-BFGS-B_basin_hopping.npy')
    plt.plot(data)
    #plt.plot(data2)
    plt.show()
    assert()

    loss_func_name_list = ['abs_green_loss']#, 'abs_mse_loss', 'cross_entropy_loss',  'abs_stoudenmire_loss']
    for name in loss_func_name_list:
        acc = np.load(f'accuracies_{name}_L-BFGS-BP_basin_hopping.npy')
        plt.plot(acc)
    plt.show()
    assert()
    """
    #Biggest equal size is n_train = 5329 * 10 with batch_num = 73
    #Can use n_train = 4913 with batch_num = 17
    #num_samples = 5329*10
    #batch_num = 73
    num_samples = 1000
    batch_num = 10
    one_site = True
    one_site_adding = False
    ortho_at_end = False

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
    """
    x_train, y_train, x_test, y_test = load_data(
        100,10, shuffle=False, equal_numbers=True
    )
    D_test = 32
    mps_test = mps_encoding(x_test, D_test)

    #loss_func_name_list = ['abs_green_loss', 'abs_mse_loss', 'cross_entropy_loss',  'abs_stoudenmire_loss']
    loss_func_name_list = ['abs_mse_loss', 'cross_entropy_loss',  'abs_stoudenmire_loss']
    #loss_func_list = [abs_green_loss, abs_mse_loss, cross_entropy_loss, abs_stoudenmire_loss]
    loss_func_list = [abs_mse_loss, cross_entropy_loss, abs_stoudenmire_loss]

    #for name, func in zip(loss_func_name_list, loss_func_list):
    quantum_stacking(classifier, bitstrings, mps_test, y_test, cross_entropy_loss, 'cross_entropy_loss')
    assert()
    """
    """

    predictions = classifier_predictions(classifier.squeeze(), mps_test, bitstrings)
    accuracy = evaluate_classifier_top_k_accuracy(predictions, y_test, 1)
    print(accuracy)

    linear_classifier = prepare_linear_classifier(mps_images, labels)
    linear_predictions = linear_classifier_predictions(linear_classifier, mps_test, labels)
    linear_accuracy = evaluate_classifier_top_k_accuracy(linear_predictions, y_test, 1)
    print(linear_accuracy)
    assert()
    """
    """
    x_train, y_train, x_test, y_test = load_data(
        100,10000, shuffle=False, equal_numbers=True
    )
    # MPS encode data
    D_encode = 32
    mps_test = mps_encoding(x_test, D_encode)

    predictions = classifier_predictions(classifier.squeeze(), mps_test, bitstrings)
    accuracy = evaluate_classifier_top_k_accuracy(predictions, y_test, 1)
    print(accuracy)
    assert()
    """
    """
    n_samples = 60000
    x_train, y_train, x_test, y_test = load_data(
        n_samples, shuffle=False, equal_numbers=False
    )
    print(x_train.shape)
    print(y_train.shape)

    mps_images = mps_encoding(x_train, 32)
    labels = y_train
    """

    """
    x_train, y_train, x_test, y_test = load_data(
        1000,10, shuffle=False, equal_numbers=True
    )
    # MPS encode data
    D_encode = 32
    mps_train = mps_encoding(x_train, D_encode)
    """

    #assert()
    #ortho_classifier = load_qtn_classifier('Big_Classifiers/ortho_mpo_classifier_32')
    #train_predictions(mps_images, labels, classifier.squeeze(), bitstrings)

    """
    #title = 'deterministic_initialisation_non_ortho'
    #classifier = load_qtn_classifier('Big_Classifiers/non_ortho_mpo_classifier_32')
    training_accuracies1 = []
    training_accuracies2 = []
    for n in tqdm(range(10)):
        acc1, acc2 = train_predictions(mps_images, labels, classifier.squeeze(), bitstrings, 'TEST', n)
        training_accuracies1.append(acc1)
        training_accuracies2.append(acc2)
        np.save('training_accuracies1', training_accuracies1)
        np.save('training_accuracies2', training_accuracies2)

    assert()
    classifier = load_qtn_classifier('Big_Classifiers/non_ortho_mpo_classifier_32')

    x_train, y_train, x_test, y_test = load_data(
        100,10000, shuffle=False, equal_numbers=True
    )
    mps_test = mps_encoding(x_test, 32)

    test_predictions = np.array(classifier_predictions(classifier.squeeze(), mps_test, bitstrings))

    print(evaluate_classifier_top_k_accuracy(test_predictions, y_test, 1))
    assert()
    """
    #my_model = no val split
    #model = tf.keras.models.load_model('saved_model2/my_model_batch_val_split')
    #model.summary()


    #model.evaluate(test_predictions, y_test)
    #assert()


    #obtain_deterministic_accuracies(bitstrings)
    #print(mps_images[0].squeeze().H @ (classifier.squeeze() @ bitstrings[5].squeeze()))

    #predictions = classifier_predictions(classifier.squeeze(), mps_images, bitstrings)
    #accuracy = evaluate_classifier_top_k_accuracy(predictions, labels, 1)
    #print(accuracy)
    """
    x_train, y_train, x_test, y_test = load_data(
        100,100, shuffle=False, equal_numbers=True
    )
    D_test = 32
    x_test = [x_test[label == y_test][0] for label in range(10)]
    y_test = np.array(range(10))
    mps_test = mps_encoding(x_test, D_test)
    """

    #train_predictions(mps_images, labels, classifier.squeeze(), bitstrings)
    for i in range(3):
        print('n_copies: ', i)
        deterministic_quantum_stacking(labels, bitstrings, i)
    assert()

    """
    #loss_func_name_list = ['abs_green_loss', 'abs_mse_loss', 'cross_entropy_loss',  'abs_stoudenmire_loss']
    loss_func_name_list = ['abs_mse_loss', 'cross_entropy_loss',  'abs_stoudenmire_loss']
    #loss_func_list = [abs_green_loss, abs_mse_loss, cross_entropy_loss, abs_stoudenmire_loss]
    loss_func_list = [abs_mse_loss, cross_entropy_loss, abs_stoudenmire_loss]
    """
    #predictions = classifier_predictions(classifier.squeeze(), mps_test, bitstrings)
    #accuracy = evaluate_classifier_top_k_accuracy(predictions, y_test, 1)
    #print(accuracy)
    #for name, func in zip(loss_func_name_list, loss_func_list):
    quantum_stacking_2(classifier, bitstrings, mps_test, y_test)#, cross_entropy_loss, 'cross_entropy_loss')
    assert()

    #x_train, y_train, x_test, y_test = load_data(
    #    1000, shuffle=False, equal_numbers=True
    #)
    #train_predictions(x_train, y_train, classifier, bitstrings)
    #tracing_over(classifier, bitstrings, mps_images, labels)

    #print('Evaluating Data!')
    #predictions = classifier_predictions(classifier.squeeze(), mps_images, bitstrings)
    #accuracy = evaluate_classifier_top_k_accuracy(predictions, labels, 1)
    #print(accuracy)
    #assert()

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
