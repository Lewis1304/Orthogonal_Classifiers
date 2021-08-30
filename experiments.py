from variational_mpo_classifiers import *
import os

"""
Experiments
"""


def initialise_experiment(
    n_samples, D_total, padded=False, initialise_classifier=False, orthogonalise=False
):
    """
    param: n_samples: Number of data samples (total)
    param: D_total: Bond dimension of classifier and data
    """

    x_train, y_train, x_test, y_test = load_data(n_samples)

    # All possible class labels
    possible_labels = list(set(y_train))
    # Number of "label" sites
    n_hairysites = int(np.ceil(math.log(len(possible_labels), 4)))
    # Number of total sites (mps encoding)
    n_sites = int(np.ceil(math.log(x_train.shape[-1], 2)))

    # Create hairy bitstrings
    hairy_bitstrings_data = create_hairy_bitstrings_data(
        possible_labels, n_hairysites, n_sites
    )
    q_hairy_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, n_hairysites, n_sites, truncated=True
    )
    # q_hairy_bitstrings[0].draw(show_inds = False, show_tags = False)
    # MPS encode data
    mps_train = mps_encoding(x_train, D_total)

    # MPO encode data (already encoded as mps)
    # Has shape: # classes, mpo.shape
    mpo_train = mpo_encoding(mps_train, y_train, q_hairy_bitstrings)
    # mpo_train[0][0].draw(show_inds = False, show_tags = False)

    # Initial Classifier
    if initialise_classifier:
        mpo_classifier = initialise_sequential_mpo_classifier(x_train, y_train, D_total)
    else:
        mpo_classifier = create_mpo_classifier(mpo_train, seed=420)

    if padded:
        hairy_bitstrings_data_padded_data = create_padded_hairy_bitstrings_data(
            possible_labels, n_hairysites, n_sites
        )
        q_hairy_bitstrings = [
            bitstring_data_to_QTN(padding)
            for padding in hairy_bitstrings_data_padded_data
        ]

    return (mps_train, y_train), mpo_classifier, q_hairy_bitstrings


def single_image_experiment(mpo_classifier, mps_train, q_hairy_bitstrings, y_train):
    def single_loss(classifier, single_mps_train, bitstring):
        overlap = (single_mps_train @ (classifier @ bitstring)) ** 2
        return -overlap.norm()

    optmzr = TNOptimizer(
        mpo_classifier,  # our initial input, the tensors of which to optimize
        # loss_fn=lambda c: negative_loss(c, mps_train, q_hairy_bitstrings, y_train),
        loss_fn=single_loss,
        norm_fn=normalize_tn,
        loss_constants={
            "single_mps_train": mps_train[0],
            "bitstring": q_hairy_bitstrings[y_train[0]],
        },
        autodiff_backend="jax",  # {'jax', 'tensorflow', 'autograd'}
        optimizer="adam",  # supplied to scipy.minimize
    )

    classifier_opt = optmzr.optimize(10000)

    losses = optmzr.losses

    plt.plot(losses)
    plt.show()


def two_image_experiment(mpo_classifier, mps_train, q_hairy_bitstrings, y_train):
    def double_loss(classifier, mps_train, bitstring):
        overlap = (
            np.sum([((i @ (classifier @ bitstring)) ** 2).norm() for i in mps_train])
            / 2
        )
        return -overlap
        return -(overlap1.norm() + overlap2.norm())

    label = 0
    mps_train = np.array(mps_train)[y_train == label][:2]

    optmzr = TNOptimizer(
        mpo_classifier,  # our initial input, the tensors of which to optimize
        # loss_fn=lambda c: negative_loss(c, mps_train, q_hairy_bitstrings, y_train),
        loss_fn=lambda c: double_loss(c, mps_train, q_hairy_bitstrings[label]),
        norm_fn=normalize_tn,
        # loss_constants = {'mps_train': mps_train, 'bitstring': q_hairy_bitstrings[label]},
        autodiff_backend="jax",  # {'jax', 'tensorflow', 'autograd'}
        optimizer="nadam",  # supplied to scipy.minimize
    )

    classifier_opt = optmzr.optimize(1000)

    losses = optmzr.losses

    plt.plot(losses)
    plt.show()


def one_class_experiment(mpo_classifier, mps_train, q_hairy_bitstrings, y_train):
    def class_loss(classifier, mps_train, bitstring):
        overlap = np.sum(
            [((i @ (classifier @ bitstring)) ** 2).norm() for i in mps_train]
        ) / len(mps_train)
        return -overlap

    label = 0
    mps_train = np.array(mps_train)[y_train == label]

    initial_result = evaluate_classifier(
        mpo_classifier,
        mps_train,
        q_hairy_bitstrings,
        [label for _ in range(len(mps_train))],
    )
    # print(initial_result)

    optmzr = TNOptimizer(
        mpo_classifier,  # our initial input, the tensors of which to optimize
        # loss_fn=lambda c: negative_loss(c, mps_train, q_hairy_bitstrings, y_train),
        loss_fn=lambda c: class_loss(c, mps_train, q_hairy_bitstrings[label]),
        norm_fn=normalize_tn,
        autodiff_backend="jax",  # {'jax', 'tensorflow', 'autograd'}
        optimizer="nadam",  # supplied to scipy.minimize
    )

    results = [initial_result]
    losses = [class_loss(mpo_classifier, mps_train, q_hairy_bitstrings[label])]
    for i in range(1000):
        classifier_opt = optmzr.optimize(1)
        results.append(
            evaluate_classifier(
                classifier_opt,
                mps_train,
                q_hairy_bitstrings,
                [label for _ in range(len(mps_train))],
            )
        )
        losses.append(optmzr.loss)

    # plt.plot(losses)
    # plt.show()

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(losses, color=color, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color)  # we already handled the x-label with ax1
    ax2.plot(results, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    fig.suptitle("Loss / Accuracy One Class")
    plt.savefig("one_class_loss_accuracy.pdf")
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def two_classes_experiment(mpo_classifier, mps_train, q_hairy_bitstrings, y_train):

    zeros = np.array(mps_train)[y_train == 0]
    ones = np.array(mps_train)[y_train == 1]
    # mps test is a subset of mps_train, only includes images of classes trained over
    mps_test = np.append(zeros, ones)
    labels = [0 if i < len(zeros) else 1 for i in range(len(zeros) + len(ones))]

    initial_result = evaluate_classifier(
        mpo_classifier, mps_test, q_hairy_bitstrings, labels
    )
    results = [initial_result]
    losses = [negative_loss(mpo_classifier, mps_train, q_hairy_bitstrings, labels)]

    optmzr = TNOptimizer(
        mpo_classifier,  # our initial input, the tensors of which to optimize
        loss_fn=lambda c: negative_loss(c, mps_train, q_hairy_bitstrings, labels),
        norm_fn=normalize_tn,
        autodiff_backend="jax",  # {'jax', 'tensorflow', 'autograd'}
        optimizer="nadam",  # supplied to scipy.minimize
    )

    for i in range(1000):
        classifier_opt = optmzr.optimize(1)
        results.append(
            evaluate_classifier(classifier_opt, mps_test, q_hairy_bitstrings, labels)
        )
        losses.append(optmzr.loss)

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(losses, color=color, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color)  # we already handled the x-label with ax1
    ax2.plot(results, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    fig.suptitle("Loss / Accuracy One Class")
    plt.savefig("two_class_loss_accuracy.pdf")
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def all_classes_experiment(
    mpo_classifier,
    mps_train,
    q_hairy_bitstrings,
    y_train,
    predict_func,
    loss_func,
    title,
):
    classifier = compress_QTN(mpo_classifier, D=None, orthogonalise=True)
    initial_predictions = predict_func(mpo_classifier, mps_train, q_hairy_bitstrings)

    predicitions_store = [initial_predictions]
    accuracies = [evaluate_classifier_top_k_accuracy(initial_predictions, y_train, 3)]
    # variances = [evaluate_prediction_variance(initial_predictions)]
    losses = [loss_func(mpo_classifier, mps_train, q_hairy_bitstrings, y_train)]

    optmzr = TNOptimizer(
        classifier,  # our initial input, the tensors of which to optimize
        loss_fn=lambda c: loss_func(c, mps_train, q_hairy_bitstrings, y_train),
        norm_fn=normalize_tn,
        autodiff_backend="autograd",  # {'jax', 'tensorflow', 'autograd'}
        optimizer="nadam",  # supplied to scipy.minimize
    )

    for i in range(500):
        classifier_opt = optmzr.optimize(1)
        classifier_opt = compress_QTN(classifier_opt, D=None, orthogonalise=True)

        predictions = predict_func(classifier_opt, mps_train, q_hairy_bitstrings)
        predicitions_store.append(predictions)
        accuracies.append(evaluate_classifier_top_k_accuracy(predictions, y_train, 3))
        # variances.append(evaluate_prediction_variance(predictions))

        losses.append(optmzr.loss)

        plot_results((accuracies, losses, predicitions_store), title)
        save_qtn_classifier(classifier, title)

    return accuracies, losses


def sequential_mpo_classifier_experiment():
    n_samples = 1000
    possible_labels = list(range(10))

    # train_data, train_labels, test_data, test_labels = tn_classifiers().gather_data_mnist(n_samples, 1, 1, shuffle=True, equal_classes=False)

    train_data, train_labels, test_data, test_labels = load_data(n_samples)

    grouped_images = [
        [train_data[i] for i in range(n_samples) if train_labels[i] == label]
        for label in possible_labels
    ]

    train_data = [images for images in zip(*grouped_images)]
    train_labels = [possible_labels for _ in train_data]

    train_data = np.array([item for sublist in train_data for item in sublist])
    train_labels = np.array([item for sublist in train_labels for item in sublist])

    train_spread_mpo_product_states = mps_encoded_data(
        train_data, train_labels, D_total
    )
    mpo_classifier = sequential_mpo_classifier(
        train_labels=train_labels, mps_images=train_spread_mpo_product_states
    )

    train_spread_mpo_product_states = [
        image for label in train_spread_mpo_product_states.values() for image in label
    ]

    MPOs = train_spread_mpo_product_states
    while len(MPOs) > 1:
        MPOs = mpo_classifier.adding_batches(MPOs, D_total, 10, orthogonalise=False)
    classifier = MPOs[0].left_spread_canonicalise(D=D_total, orthogonalise=False)

    result = mpo_classifier.evaluate_classifiers(
        classifier, train_data, train_labels, mps_encode=True, D=D_total
    )

    return classifier, result


"""
Results
"""


def plot_results(results, title):

    accuracies, losses, predictions = results

    np.save("results/" + title + "_accuracies", accuracies)
    np.save("results/" + title + "_losses", losses)
    # np.save('results/' + title + '_variances', variances)
    np.save("results/" + title + "_predictions", predictions)

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
    plt.savefig("figures/" + title + ".pdf")
    plt.close(fig)


if __name__ == "__main__":

    num_samples = 1000
    D_total = 10

    data, classifier, bitstrings = initialise_experiment(
        num_samples,
        D_total,
        padded=False,
        initialise_classifier=False,
        orthogonalise=False,
    )
    mps_images, labels = data

    all_classes_experiment(
        classifier,
        mps_images,
        bitstrings,
        labels,
        classifier_predictions,
        stoundenmire_loss,
        "stoudenmire_orthogonalise_inbetween",
    )
