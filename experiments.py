from variational_mpo_classifiers import *
from deterministic_mpo_classifier import prepare_batched_classifier
import os
from tqdm import tqdm

"""
Prepare Experiment
"""

def initialise_experiment(
    n_samples,
    D_total,
    arrangement = 'one class',
    truncated=False,
    one_site=False,
    initialise_classifier=False,
    initialise_classifier_settings = (10, False, False)
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
    #Load & Organise Data
    x_train, y_train, x_test, y_test = load_data(n_samples, shuffle = False, equal_numbers = True)
    x_train, y_train = arrange_data(x_train, y_train, arrangement = 'one class')

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
    mps_train = mps_encoding(x_train, D_total)

    # Initial Classifier
    if initialise_classifier:
        batch_num, one_site_adding, ortho_at_end  = initialise_classifier_settings
        fmpo_classifier = prepare_batched_classifier(x_train, y_train, D_total, batch_num, one_site = one_site_adding)

        if one_site:
            #Works for n_sites != 1. End result is a classifier with n_site = 1.
            classifier_data = fmpo_classifier.compress_one_site(D=D_total, orthogonalise=ortho_at_end)
            mpo_classifier = data_to_QTN(classifier_data.data)
        else:
            classifier_data = fmpo_classifier.compress(D=D_total, orthogonalise=ortho_at_end)
            mpo_classifier = data_to_QTN(classifier_data.data)

    else:
        # MPO encode data (already encoded as mps)
        # Has shape: # classes, mpo.shape
        mpo_classifier = create_mpo_classifier(mps_train, q_hairy_bitstrings, seed=420)

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
    squeezed=False,
    ortho_inbetween=False,
):
    print(title)

    classifier_opt = mpo_classifier

    if squeezed:
        q_hairy_bitstrings = [i.squeeze() for i in q_hairy_bitstrings]
        classifier_opt = mpo_classifier.squeeze()
        mps_train = [i.squeeze() for i in mps_train]

    initial_predictions = predict_func(classifier_opt, mps_train, q_hairy_bitstrings)

    predicitions_store = [initial_predictions]
    accuracies = [evaluate_classifier_top_k_accuracy(initial_predictions, y_train, 1)]
    # variances = [evaluate_prediction_variance(initial_predictions)]
    losses = [loss_func(classifier_opt, mps_train, q_hairy_bitstrings, y_train)]


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
    for i in range(1):
        optmzr = optimiser(classifier_opt)
        classifier_opt = optmzr.optimize(1)

        if ortho_inbetween:
            classifier_opt = compress_QTN(classifier_opt, D=None, orthogonalise=True)

        predictions = predict_func(classifier_opt, mps_train, q_hairy_bitstrings)
        predicitions_store.append(predictions)
        accuracies.append(evaluate_classifier_top_k_accuracy(predictions, y_train, 3))
        # variances.append(evaluate_prediction_variance(predictions))

        losses.append(optmzr.loss)

        plot_results((accuracies, losses, predicitions_store), title)
        save_qtn_classifier(classifier_opt, title)

    return accuracies, losses


def svd_classifier(dir, mps_images, bitstrings, labels):

    classifier_og = load_qtn_classifier(dir)
    # print('Original Classifier:', classifier_og)

    predictions_og = classifier_predictions(classifier_og, mps_images, bitstrings)
    og_acc = evaluate_classifier_top_k_accuracy(predictions_og, labels, 1)
    print("Original Classifier Accuracy:", og_acc)
    # print('Original Classifier Loss:', stoundenmire_loss(classifier_og, mps_images, bitstrings, labels))

    """
    Shifted, but not orthogonalised
    """
    classifier_shifted = compress_QTN(classifier_og, None, False)
    # print(classifier_shifted)

    # predictions_shifted = classifier_predictions(classifier_shifted, mps_images, bitstrings)
    # shifted_acc = evaluate_classifier_top_k_accuracy(predictions_shifted, labels, 1)
    # print('Shifted Classifier Accuracy:', shifted_acc)
    # print('Shifted Classifier Loss:', stoundenmire_loss(classifier_shifted, mps_images, bitstrings, labels))

    """
    Shifted, and orthogonalised
    """
    classifier_ortho = compress_QTN(classifier_og, None, True)
    # print(classifier_ortho)

    predictions_ortho = classifier_predictions(classifier_ortho, mps_images, bitstrings)
    ortho_acc = evaluate_classifier_top_k_accuracy(predictions_ortho, labels, 1)
    print("Orthogonalised Classifier Accuracy:", ortho_acc)
    # print('Orthogonalised Classifier Loss:', stoundenmire_loss(classifier_ortho, mps_images, bitstrings, labels))

    return og_acc, ortho_acc

def deterministic_mpo_classifier_experiment():

    n_train = 1000
    batch_num = 10

    #Load Data- ensuring particular order to be batched added
    x_train, y_train, x_test, y_test = load_data(n_train, shuffle = False, equal_numbers = True)


    # All possible class labels
    possible_labels = list(set(y_train))
    # Number of "label" sites
    n_hairysites = int(np.ceil(math.log(len(possible_labels), 4)))
    # Number of total sites (mps encoding)
    n_sites = int(np.ceil(math.log(x_train.shape[-1], 2)))

    # Create hairy bitstrings
    hairy_bitstrings_data = create_hairy_bitstrings_data(
        possible_labels, n_hairysites, n_sites, one_site = True
    )
    q_hairy_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, n_hairysites, n_sites, truncated = True
    )
    # MPS encode data

    q_hairy_bitstrings = [i.squeeze() for i in q_hairy_bitstrings]

    one_site = False
    ortho_at_end = False

    random_arrangement = []
    one_of_each_arrangement = []
    one_class_arrangement = []

    for arrangement in tqdm(['random', 'one of each', 'one class']):
        train_data, train_labels = arrange_data(x_train, y_train, arrangement = arrangement)
        for D_total in tqdm(range(2, 52, 2)):

            mps_train = mps_encoding(train_data, D_total)
            mps_train = [i.squeeze() for i in mps_train]

            fmpo_classifier = prepare_batched_classifier(train_data, train_labels, D_total, batch_num, one_site = one_site)
            classifier_data = fmpo_classifier.compress_one_site(D=D_total, orthogonalise=ortho_at_end)
            qtn_classifier = data_to_QTN(classifier_data.data).squeeze()

            predictions = squeezed_classifier_predictions(qtn_classifier, mps_train, q_hairy_bitstrings)
            result = evaluate_classifier_top_k_accuracy(predictions, train_labels, 1)

            if arrangement == 'random':
                random_arrangement.append(result)
                np.save('results/different_arrangements/random_arrangement', random_arrangement)

            elif arrangement == 'one of each':
                one_of_each_arrangement.append(result)
                np.save('results/different_arrangements/one_of_each_arrangement', one_of_each_arrangement)

            elif arrangement == 'one class':
                one_class_arrangement.append(result)
                np.save('results/different_arrangements/one_class_arrangement', one_class_arrangement)

            else:
                raise Exception('Do not understand arrangement')


"""
Results
"""


def plot_results(results, title):

    accuracies, losses, predictions = results

    os.makedirs("results/" + title , exist_ok=True)

    np.save("results/" + title + "/accuracies", accuracies)
    np.save("results/" + title + "/losses", losses)
    #np.save('results/' + title + '_variances', variances)
    #np.save("results/" + title + "/predictions", predictions)

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


def plot_acc_before_ortho_and_after(mps_images, bitstrings, labels):

    different_classifiers = [
        "squeezed_one_site_D_total_32_full_size_abs_mse_loss_seed_420",
        "squeezed_one_site_D_total_32_full_size_mse_loss_seed_420",
        "squeezed_one_site_D_total_32_full_size_cross_entropy_loss_seed_420",
        "squeezed_one_site_D_total_32_full_size_stoudenmire_loss_seed_420",
        "squeezed_one_site_D_total_32_full_size_abs_stoudenmire_loss_seed_420",
    ]
    different_names = ["abs_mse", "mse", "cross_entropy", "abs_stoud", "stoud"]

    results_og = []
    results_ortho = []
    for c in different_classifiers:
        og, orth = svd_classifier(c, mps_images, bitstrings, labels)
        results_og.append(og)
        results_ortho.append(orth)

    fig, ax1 = plt.subplots()

    # ax1.axhline(0.95, linestyle = 'dashed', color = 'grey', label = 'Stoudenmire: D=10')
    # legend_1 = ax1.legend(loc = 'lower right')
    # legend_1.remove()
    ax1.grid(zorder=0.0, alpha=0.4)
    ax1.set_xlabel("Cost Function", labelpad=10)
    ax1.set_ylabel("Top 1- Training Accuracy")  # , color = 'C0')
    ax1.bar(
        np.arange(len(results_og)) - 0.2,
        results_og,
        0.4,
        color="C0",
        label="Non-orthogonal",
        zorder=3,
    )
    ax1.bar(
        np.arange(len(results_ortho)) + 0.2,
        results_ortho,
        0.4,
        color="C1",
        label="Orthogonal",
        zorder=3,
    )

    legend_1 = ax1.legend(loc="lower right")

    # ax1.tick_params(axis="y", labelcolor='C0')
    # ax1.set_xlim([1.75,10.25])
    ax1.set_yticks(np.arange(0.1, 1.1, 0.1))
    # ax1.set_xticks(np.arange(2, 11, 1) )

    ax1.set_xticks(np.arange(0, len(results_og), 1))
    # ax1.set_xticklabels(different_names[:len(results_og)])

    ax1.set_xticklabels(different_names)
    plt.savefig("different_cost_functions_top_1.pdf")

    plt.show()


def plot_deterministic_mpo_classifier_results():
    random_arrangement = np.load('results/different_arrangements/random_arrangement.npy')
    one_class_arrangement = np.load('results/different_arrangements/one_class_arrangement.npy')
    one_of_each_arrangement = np.load('results/different_arrangements/one_of_each_arrangement.npy')

    x = list(range(2,52,2))

    plt.plot(x, random_arrangement, label = 'randomly batched')
    plt.plot(x, one_class_arrangement, label = 'same class batched')
    plt.plot(x, one_of_each_arrangement, label = 'one of each class batched')

    plt.xlim([2,50])

    plt.xlabel('$D_{total}$')
    plt.ylabel('Top-1 training accuracy')
    plt.legend()
    plt.title('n_samples = 1000, Multiple label site, Non-orthogonal, batch_num = 10')
    plt.savefig('results/different_arrangements/train_acc_vs_D_total.pdf')
    plt.show()


if __name__ == "__main__":

    num_samples = 1000
    D_total = 32

    data, classifier, bitstrings = initialise_experiment(
        num_samples,
        D_total,
        arrangement = 'one class',
        truncated=True,
        one_site=False,
        initialise_classifier=True,
        initialise_classifier_settings = (10, False, False)

    )


    mps_images, labels = data

    # plot_acc_before_ortho_and_after(mps_images, bitstrings, labels)
    # classifier = load_qtn_classifier('one_site_stoudenmire_truncated_seed_420_more_epochs')

    all_classes_experiment(
        classifier,
        mps_images,
        bitstrings,
        labels,
        squeezed_classifier_predictions,
        abs_stoudenmire_loss,
        "one_site_false_ortho_at_end_false_initialisation_abs_stoudenmire_loss",
        squeezed=True,
    )
