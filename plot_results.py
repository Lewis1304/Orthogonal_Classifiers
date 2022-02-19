import matplotlib.pyplot as plt
import numpy as np

def plot_acc_loss(results, title):

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

def acc_vs_d_total_figure():

    non_ortho_training_accuracies = np.load('Classifiers/mnist_sum_states/sum_state_non_ortho_d_final_vs_training_accuracy.npy')[::-1]
    ortho_training_accuracies = np.load('Classifiers/mnist_sum_states/sum_state_ortho_d_final_vs_training_accuracy.npy')[::-1]
    non_ortho_test_accuracies = np.load('Classifiers/mnist_sum_states/sum_state_non_ortho_d_final_vs_test_accuracy.npy')[::-1]
    ortho_test_accuracies = np.load('Classifiers/mnist_sum_states/sum_state_ortho_d_final_vs_test_accuracy.npy')[::-1]

    #x = [2, 10, 20, 32, 50, 100, 150, 200, 250, 300, 310, 320, 330, 350]#range(2, 50, 2)
    x = range(2, 33, 2)
    plt.plot(x, non_ortho_training_accuracies, linestyle = 'dashed', color = 'tab:blue')
    plt.plot(x, ortho_training_accuracies, linestyle = 'dashed', color = 'tab:orange')
    plt.plot(x, non_ortho_test_accuracies, color = 'tab:blue')
    plt.plot(x, ortho_test_accuracies, color = 'tab:orange')
    plt.plot([],[],linestyle = 'dashed', color = 'grey', label = 'Training Accuracy')
    plt.plot([],[],linestyle = 'solid', color = 'grey', label = 'Test Accuracy')
    plt.plot([],[],linewidth = 0, marker = '.', markersize = 12, color = 'tab:blue', label = 'Non-orthogonal')
    plt.plot([],[],linewidth = 0, marker = '.', markersize = 12, color = 'tab:orange', label = 'Orthogonal')
    plt.xlabel('$D_{final}$')
    plt.ylabel('Accuracy')
    #plt.title('$D_{encode}^{train} = D_{batch} = D_{final} = D_{total}$\n $D_{encode}^{test} = 32$')
    plt.title('$D_{encode} = D_{batch} = 32$')
    plt.legend()
    #plt.savefig('proper_norm_mnist_D_total_big_dataset_results.pdf')
    plt.show()
    assert()
    """
    non_ortho_training_accuracies2 = np.load('Classifiers/Big_Classifiers/non_ortho_training_accuracies_32_50.npy')[1:]
    ortho_training_accuracies2 = np.load('Classifiers/Big_Classifiers/ortho_training_accuracies_32_50.npy')[1:]
    non_ortho_test_accuracies2 = np.load('Classifiers/Big_Classifiers/non_ortho_test_accuracies_32_50.npy')[1:]
    ortho_test_accuracies2 = np.load('Classifiers/Big_Classifiers/ortho_test_accuracies_32_50.npy')[1:]

    non_ortho_training_accuracies = np.append(non_ortho_training_accuracies, non_ortho_training_accuracies2)
    ortho_training_accuracies = np.append(ortho_training_accuracies, ortho_training_accuracies2)
    non_ortho_test_accuracies = np.append(non_ortho_test_accuracies, non_ortho_test_accuracies2)
    ortho_test_accuracies = np.append(ortho_test_accuracies, ortho_test_accuracies2)
    """
    non_ortho_training_accuracies = np.load('Classifiers/fashion_mnist/non_ortho_training_accuracies_2_32_final_3.npy')
    ortho_training_accuracies = np.load('Classifiers/fashion_mnist/ortho_training_accuracies_2_32_final_3.npy')
    non_ortho_test_accuracies = np.load('Classifiers/fashion_mnist/non_ortho_test_accuracies_2_32_final_3.npy')
    ortho_test_accuracies = np.load('Classifiers/fashion_mnist/ortho_test_accuracies_2_32_final_3.npy')

    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0.3)
    axs = gs.subplots(sharey = True, sharex = False)

    x = range(2, 50, 2)
    axs[1].plot(x,non_ortho_training_accuracies, linestyle = 'dashed', color = 'tab:blue')
    axs[1].plot(x,ortho_training_accuracies, linestyle = 'dashed', color = 'tab:orange')
    axs[1].plot(x,non_ortho_test_accuracies, color = 'tab:blue')
    axs[1].plot(x,ortho_test_accuracies, color = 'tab:orange')
    axs[1].plot([],[],linestyle = 'dashed', color = 'grey', label = 'Training Accuracy')
    axs[1].plot([],[],linestyle = 'solid', color = 'grey', label = 'Test Accuracy')
    axs[1].plot([],[],linewidth = 0, marker = '.', markersize = 12, color = 'tab:blue', label = 'Non-orthogonal')
    axs[1].plot([],[],linewidth = 0, marker = '.', markersize = 12, color = 'tab:orange', label = 'Orthogonal')


    legend = axs[1].legend(loc = 'lower right', bbox_to_anchor = (1,0.2))
    axs[1].add_artist(legend)

    axs[1].set_xticks(np.arange(2, 52, 4))
    axs[1].set_yticks(np.arange(0.1, 0.86, 0.1))
    axs[1].set_xlabel('$D_{total}$')
    axs[1].set_ylabel('$Accuracy$')
    axs[1].grid(alpha = 0.8)

    #legend2 = axs[1].legend(handles=test, loc='lower right')


    label1 = axs[1].plot([],[],label = '$D_{encode} = D_{batch} = D_{final}$', linewidth = 0)
    legend2 = axs[1].legend(handles=label1, loc='lower right',handlelength=0, handletextpad=-0.1, bbox_to_anchor = (0.979,0.0))

    axs[1].set_xlabel('$D_{final}$')
    axs[1].set_ylabel('$Accuracy$')
    axs[1].yaxis.set_label_coords(-0.075,1.15)

    axs[1].set_xlim([2, 50])
    non_ortho_d_final_vs_training_acc = np.load('Classifiers/fashion_mnist/non_ortho_d_final_vs_training_accuracy.npy')
    ortho_d_final_vs_training_acc = np.load('Classifiers/fashion_mnist/ortho_d_final_vs_training_accuracy.npy')
    non_ortho_d_final_vs_test_acc = np.load('Classifiers/fashion_mnist/non_ortho_d_final_vs_test_accuracy.npy')
    ortho_d_final_vs_test_acc = np.load('Classifiers/fashion_mnist/ortho_d_final_vs_test_accuracy.npy')

    x2 = range(2, 33, 2)
    axs[0].set_xticks(np.arange(2, 33, 2))
    axs[0].set_yticks(np.arange(0.1, 0.86, 0.1))
    axs[0].grid(alpha = 0.8)

    axs[0].plot(x2,non_ortho_d_final_vs_training_acc, linestyle = 'dashed', color = 'tab:blue')
    axs[0].plot(x2,ortho_d_final_vs_training_acc, linestyle = 'dashed', color = 'tab:orange')
    axs[0].plot(x2,non_ortho_d_final_vs_test_acc, color = 'tab:blue')
    axs[0].plot(x2,ortho_d_final_vs_test_acc, color = 'tab:orange')
    axs[0].set_xlabel('$D_{final}$')



    axs[0].set_xlim([2, 32])

    label3 = axs[0].plot([],[],label = '$D_{encode} = D_{batch} = 32$', linewidth = 0, color = 'k')
    legend4 = axs[0].legend(handles=label3, loc='lower right',handlelength=0, handletextpad=-0.1, bbox_to_anchor = (0.945,0))

    plt.savefig('fashion_mnist_accuracy_vs_D_final.pdf')
    plt.tight_layout()
    plt.show()

def acc_vs_d_encode_d_batch_d_final():
    d_batch_accuracies = np.load('Classifiers/fashion_mnist/d_batch_vs_acc_d_final_10_20_32.npy').reshape(-1,3).T
    d_encode_accuracies = np.load('Classifiers/fashion_mnist/d_encode_vs_acc_d_final_10_20_32.npy').reshape(-1,3).T

    x = range(2, 33, 2)
    """
    for d_final, accuracy in zip([10, 20, 32], d_batch_accuracies):
        plt.plot(x, accuracy, label = '$D_{final}'+ f'= {d_final}$')

    plt.legend()
    plt.show()
    """
    #fig, axs = plt.subplots(2)
    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0.3)
    axs = gs.subplots(sharey = False)


    for d_final, b_accuracy, e_accuracy in zip([10, 20, 32], d_batch_accuracies, d_encode_accuracies):
        axs[0].plot(x, b_accuracy, label = '$D_{final}'+ f'= {d_final}$')
        axs[1].plot(x, e_accuracy, label = '$D_{final}'+ f'= {d_final}$')


    legend1 = axs[0].legend(loc = 'upper left')

    #legend2 = axs[1].legend(handles=test, loc='lower right')
    axs[0].add_artist(legend1)


    label1 = axs[0].plot([],[],label = '$D_{encode} = 32$', linewidth = 0)
    legend2 = axs[0].legend(handles=label1, loc='lower right',handlelength=0, handletextpad=-0.3)

    label2 = axs[1].plot([],[],label = '$D_{batch} = 32$', linewidth = 0)
    legend3 = axs[1].legend(handles=label2, loc='lower right',handlelength=0, handletextpad=-0.3)


    #axs[1].legend(loc = 'upper left')
    axs[0].set_xlabel('$D_{batch}$')
    axs[1].set_xlabel('$D_{encode}$')
    axs[0].grid(alpha = 0.8)
    axs[1].grid(alpha = 0.8)


    #plt.ylabel('Test Accuracy')
    gs = fig.add_gridspec(2, hspace=0)

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)

    axs[0].set_xticks(np.arange(2, 33, 2))
    axs[1].set_xticks(np.arange(2, 33, 2))

    axs[0].set_yticks(np.arange(0, 0.7, 0.1))
    axs[1].set_yticks(np.arange(0.35, 0.7, 0.05))
    #axs[2].set_yticks(np.arange(0.2, 0.86, 0.2))

    #plt.setp( axs[0].get_xticklabels(), visible=False)
    #plt.setp( axs[1].get_xticklabels(), visible=False)


    plt.ylabel("Test Accuracy", labelpad = 10)
    axs[0].set_xlim([2,32])
    axs[1].set_xlim([2,32])


    #axs[1].set_ylim([0.6,0.85])




    plt.savefig('fashion_acc_vs_d_batch_d_encode.pdf')
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

def variational_test_accuracies():
    test_accuracies = np.load('ortho_cross_entropy_loss_det_init_non_ortho_test_accuracies.npy')
    print(max(test_accuracies))
    plt.plot(test_accuracies)
    plt.show()

if __name__ == '__main__':
    acc_vs_d_total_figure()
    #variational_test_accuracies()

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
