from variational_mpo_classifiers import *
from deterministic_mpo_classifier import prepare_batched_classifier, prepare_ensemble, unitary_qtn
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

    # Load & Organise Data
    x_train, y_train, x_test, y_test = load_data(
        n_samples, shuffle=False, equal_numbers=True
    )
    x_train, y_train = arrange_data(x_train, y_train, arrangement=arrangement)

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

    # Initial Classifier
    if initialise_classifier:
        batch_num, one_site_adding, ortho_at_end = initialise_classifier_settings
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
        mpo_classifier = create_mpo_classifier(
            mps_train, q_hairy_bitstrings, seed=420, full_sized=True
        ).squeeze()

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

    initial_predictions = predict_func(classifier_opt, mps_train, q_hairy_bitstrings)

    predicitions_store = [initial_predictions]
    accuracies = [evaluate_classifier_top_k_accuracy(initial_predictions, y_train, 1)]
    # variances = [evaluate_prediction_variance(initial_predictions)]
    losses = [loss_func(classifier_opt, mps_train, q_hairy_bitstrings, y_train)]

    print(accuracies)
    assert()

    def optimiser(classifier):
        optmzr = TNOptimizer(
            classifier,  # our initial input, the tensors of which to optimize
            loss_fn=lambda c: loss_func(c, mps_train, q_hairy_bitstrings, y_train),
            norm_fn=normalize_tn,
            autodiff_backend="autograd",  # {'jax', 'tensorflow', 'autograd'}
            optimizer="nadam",  # supplied to scipy.minimize
        )
        return optmzr

    print(classifier_opt)
    print(q_hairy_bitstrings[0])
    best_accuracy = 0
    for i in range(1000):
        optmzr = optimiser(classifier_opt)
        classifier_opt = optmzr.optimize(1)

        predictions = predict_func(classifier_opt, mps_train, q_hairy_bitstrings)
        predicitions_store.append(predictions)
        accuracy = evaluate_classifier_top_k_accuracy(predictions, y_train, 1)
        accuracies.append(accuracy)
        # variances.append(evaluate_prediction_variance(predictions))

        losses.append(optmzr.loss)

        plot_results((accuracies, losses, predicitions_store), title)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_qtn_classifier(classifier_opt, title)

    return accuracies, losses


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


def quantum_stacking(classifier, bitstrings, mps_images, labels):
    import quimb as qu
    import quimb.tensor as qtn

    def single_qubit_layer(circ, gate_round=None):
        """Apply a parametrizable layer of single qubit ``U3`` gates.
        """
        for i in range(circ.N):
            # initialize with random parameters
            params = qu.randn(3, dist='uniform')
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

    def ansatz_circuit(n, depth, gate2='CX', **kwargs):
        """Construct a circuit of single qubit and entangling layers.
        """
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


    bitstrings_qubits = [bitstring.squeeze().tensors[-1].data.reshape(2,2,2,2) for bitstring in bitstrings]
    qtn_bitstrings = [qtn.Tensor(bitstring_qubits, inds = ['b4', 'b5', 'b6', 'b7']) for bitstring_qubits in bitstrings_qubits]

    n = 4
    ancillae_qubits = np.eye(2**n)[0]
    prediction_and_ancillae_qubits = [qtn.Tensor(np.kron(ancillae_qubits, (mps_image.H @ classifier).squeeze().data).reshape(*[2]*(2*n)), inds = [f'k{i}' for i in range(2*n)]) for mps_image in mps_images]


    circ = ansatz_circuit(2*n, n, gate2='CX')
    #circ = qtn.Circuit(8)
    #for i in range(8):
    #    circ.apply_gate('X',i)
    #    circ.apply_gate('X',i)
    V_opt = circ.uni
    """
    V.draw(color = ['U3', 'CX'], show_inds = True)

    def loss(V, prediction_and_ancillae_qubits, qtn_bitstrings, labels):
        overlaps = [
            anp.log(
                (
                    prediction_and_ancillae_qubits[i].squeeze().H
                    & (V & qtn_bitstrings[labels[i]].squeeze())
                ).contract(all).norm()
            )
            for i in range(len(prediction_and_ancillae_qubits))
        ]
        return -np.sum(overlaps) / len(prediction_and_ancillae_qubits)

    optmzr = qtn.TNOptimizer(
            V,  # our initial input, the tensors of which to optimize
            loss_fn = lambda V: loss(V, prediction_and_ancillae_qubits, qtn_bitstrings, labels),
            autodiff_backend="autograd",  # {'jax', 'tensorflow', 'autograd'}
            optimizer="nadam",  # supplied to scipy.minimize
            )
    #(qtn_bitstrings[0] & (V & prediction_and_ancillae_qubits[0])).draw(color = ['U3', 'CZ', 'B'], show_inds = True)
    V_opt = optmzr.optimize_basinhopping(n=10, nhop=10)

    predictions = [
            [
                (p.squeeze().H
                & (V & b.squeeze())
                ).contract(all).norm()
                for b in qtn_bitstrings
            ]
            for p in prediction_and_ancillae_qubits
        ]
    accuracies = evaluate_classifier_top_k_accuracy(predictions, labels, 1)
    print(accuracies)
    """


    def optimiser(circ):
        optmzr = qtn.TNOptimizer(
                circ,  # our initial input, the tensors of which to optimize
                loss_fn = lambda V: loss(V, prediction_and_ancillae_qubits, qtn_bitstrings, labels),
                autodiff_backend="autograd",  # {'jax', 'tensorflow', 'autograd'}
                optimizer="nadam",  # supplied to scipy.minimize
                )
        return optmzr

    predicitions_store = []
    accuracies = []
    for i in tqdm(range(100)):
        optmzr = optimiser(V_opt)
        V_opt = optmzr.optimize_basinhopping(n=1, nhop=10)

        predictions = [
                [
                    (p.squeeze().H
                    & (V & b.squeeze())
                    ).contract(all).norm()
                    for b in qtn_bitstrings
                ]
                for p in prediction_and_ancillae_qubits
            ]

        predicitions_store.append(predictions)
        accuracy = evaluate_classifier_top_k_accuracy(predictions, labels, 1)
        accuracies.append(accuracy)

        losses.append(optmzr.loss)

        plot_results((accuracies, losses, predicitions_store), 'quantum_circuit')









if __name__ == "__main__":

    num_samples = 100
    batch_num = 10

    one_site = True
    one_site_adding = False
    ortho_at_end = True

    D_total = 32
    D_encode = D_total
    D_batch = D_total
    D_final = D_total
    D = (D_encode, D_batch, D_final)
    #deterministic_mpo_classifier_experiment(1000, 10)
    #assert()

    data, classifier, bitstrings = initialise_experiment(
        num_samples,
        D,
        arrangement="one class",
        truncated=True,
        one_site=one_site,
        initialise_classifier=True,
        initialise_classifier_settings=(batch_num, one_site_adding, ortho_at_end),
    )

    mps_images, labels = data
    #print(mps_images[0].squeeze().H @ (classifier.squeeze() @ bitstrings[5].squeeze()))
    quantum_stacking(classifier, bitstrings, mps_images, labels)
    assert()

    #x_train, y_train, x_test, y_test = load_data(
    #    1000, shuffle=False, equal_numbers=True
    #)
    #train_predictions(x_train, y_train, classifier, bitstrings)
    #tracing_over(classifier, bitstrings, mps_images, labels)

    predictions = classifier_predictions(uclassifier.squeeze(), mps_images, bitstrings)
    accuracy = evaluate_classifier_top_k_accuracy(predictions, labels, 1)
    print(accuracy)
    assert()

    all_classes_experiment(
        classifier,
        mps_images,
        bitstrings,
        labels,
        classifier_predictions,
        abs_stoudenmire_loss,
        "TEST",
        # "full_sized_random_one_site_false",
    )
