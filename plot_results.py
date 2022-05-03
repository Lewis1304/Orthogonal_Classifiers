import matplotlib.pyplot as plt
import numpy as np
from tools import load_data, load_qtn_classifier, data_to_QTN
from experiments import create_experiment_bitstrings, adding_centre_batches
from fMPO import fMPO

"""
Plot D experiments
"""

def acc_vs_d_total_figure():


    non_ortho_training_predictions = np.load('Classifiers/mnist_mixed_sum_states/D_total/' + "non_ortho_d_total_vs_training_predictions.npy")
    ortho_training_predictions = np.load('Classifiers/mnist_mixed_sum_states/D_total/' + "ortho_d_total_vs_training_predictions.npy")
    non_ortho_test_predictions = np.load('Classifiers/mnist_mixed_sum_states/D_total/' + "non_ortho_d_total_vs_test_predictions.npy")
    ortho_test_predictions = np.load('Classifiers/mnist_mixed_sum_states/D_total/' + "ortho_d_total_vs_test_predictions.npy")


    n_train_samples = 5421*10
    n_test_samples = 10000

    x_train, y_train, x_test, y_test = load_data(
        n_train_samples,n_test_samples, shuffle=False, equal_numbers=True
    )

    x_train, y_train = arrange_data(x_train, y_train, arrangement='one class')


    non_ortho_training_accuracy = [evaluate_classifier_top_k_accuracy(i, y_train, 1) for i in non_ortho_training_predictions]
    ortho_training_accuracy = [evaluate_classifier_top_k_accuracy(i, y_train, 1) for i in ortho_training_predictions]
    non_ortho_test_accuracy = [evaluate_classifier_top_k_accuracy(i, y_test, 1) for i in non_ortho_test_predictions]
    ortho_test_accuracy = [evaluate_classifier_top_k_accuracy(i, y_test, 1) for i in ortho_test_predictions]

    #non_ortho_training_accuracies = np.load('Classifiers/mnist_sum_states/sum_state_non_ortho_d_final_vs_training_accuracy.npy')[::-1]
    #ortho_training_accuracies = np.load('Classifiers/mnist_sum_states/sum_state_ortho_d_final_vs_training_accuracy.npy')[::-1]
    #non_ortho_test_accuracies = np.load('Classifiers/mnist_sum_states/sum_state_non_ortho_d_final_vs_test_accuracy.npy')[::-1]
    #ortho_test_accuracies = np.load('Classifiers/mnist_sum_states/sum_state_ortho_d_final_vs_test_accuracy.npy')[::-1]

    #x = [2, 10, 20, 32, 50, 100, 150, 200, 250, 300, 310, 320, 330, 350]#range(2, 50, 2)
    x = range(2, 37, 2)
    plt.plot(x, non_ortho_training_accuracy, linestyle = 'dashed', color = 'tab:blue')
    plt.plot(x, ortho_training_accuracy, linestyle = 'dashed', color = 'tab:orange')
    plt.plot(x, non_ortho_test_accuracy, color = 'tab:blue')
    plt.plot(x, ortho_test_accuracy, color = 'tab:orange')
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
    non_ortho_test_accuracies = []
    ortho_test_accuracies = []

    n_test_samples = 10000
    x_train, y_train, x_test, y_test = load_data(
        100,n_test_samples, shuffle=False, equal_numbers=True
    )

    for D_final in [10, 20]:
        non_ortho_test_predictions = np.load('Classifiers/mnist_mixed_sum_states/D_encode/' + f"D_final_{D_final}_non_ortho_d_total_vs_test_predictions.npy")
        ortho_test_predictions = np.load('Classifiers/mnist_mixed_sum_states/D_encode/' + f"D_final_{D_final}_ortho_d_total_vs_test_predictions.npy")


        non_ortho_test_accuracy = [evaluate_classifier_top_k_accuracy(i, y_test, 1) for i in non_ortho_test_predictions]
        ortho_test_accuracy = [evaluate_classifier_top_k_accuracy(i, y_test, 1) for i in ortho_test_predictions]

        non_ortho_test_accuracies.append(non_ortho_test_accuracy)
        ortho_test_accuracies.append(ortho_test_accuracy)

    #fig, axs = plt.subplots(2)
    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0.3)
    axs = gs.subplots(sharey = False)


    for d_final, b_accuracy, e_accuracy in zip([10, 20], d_batch_accuracies, d_encode_accuracies):
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

"""
Plot confusion matrices
"""

def compute_predictions_confusion_matrix(bitstrings, rearrange = False):

    """
    MNIST
    """
    mnist_path = "mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_32/"

    mnist_sum_states = [load_qtn_classifier(mnist_path + f"digit_{i}") for i in range(10)]
    mnist_mps_sum_states = [s @ b for s, b in zip(mnist_sum_states, bitstrings)]

    mnist_sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in mnist_sum_states]
    mnist_ortho_classifier_data = adding_centre_batches(mnist_sum_states_data, 32, 10, orthogonalise = True)[0]
    mnist_ortho_mpo_classifier = data_to_QTN(mnist_ortho_classifier_data.data).squeeze()

    #Has shape (Class, Class_prediction)
    mnist_results = []
    for i in mnist_mps_sum_states:
        state_i = (mnist_ortho_mpo_classifier @ i).squeeze()
        state_i /= state_i.norm()
        mnist_results.append(abs(state_i.data[:10]))



    """
    FASHION MNIST
    """
    fashion_mnist_path = "fashion_mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_32/"

    fashion_mnist_sum_states = [load_qtn_classifier(fashion_mnist_path + f"digit_{i}") for i in range(10)]
    fashion_mnist_mps_sum_states = [s @ b for s, b in zip(fashion_mnist_sum_states, bitstrings)]

    fashion_mnist_sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in fashion_mnist_sum_states]
    fashion_mnist_ortho_classifier_data = adding_centre_batches(fashion_mnist_sum_states_data, 32, 10, orthogonalise = True)[0]
    fashion_mnist_ortho_mpo_classifier = data_to_QTN(fashion_mnist_ortho_classifier_data.data).squeeze()

    #Has shape (Class, Class_prediction)
    fashion_mnist_results = []
    for i in fashion_mnist_mps_sum_states:
        state_i = (fashion_mnist_ortho_mpo_classifier @ i).squeeze()
        state_i /= state_i.norm()
        fashion_mnist_results.append(abs(state_i.data[:10]))


    """
    PLOT RESULTS
    """

    mnist_results = np.array(mnist_results).reshape(10,10)
    fashion_mnist_results = np.array(fashion_mnist_results).reshape(10,10)

    if rearrange:

        rearranged_mnist_results = np.zeros_like(mnist_results)
        rearranged_fashion_mnist_results = np.zeros_like(fashion_mnist_results)
        inds = diagonal_indices(rearranged_mnist_results.shape[0])

        mnist_results = np.sort(mnist_results.flatten(), axis = 0)[::-1]
        fashion_mnist_results = np.sort(fashion_mnist_results.flatten(), axis = 0)[::-1]

        for i, j, k in zip(inds, mnist_results, fashion_mnist_results):
            rearranged_mnist_results[i[0],i[1]] = j
            rearranged_fashion_mnist_results[i[0],i[1]] = k

        mnist_results = rearranged_mnist_results
        fashion_mnist_results = rearranged_fashion_mnist_results

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(mnist_results, cmap = 'Greys')
    axarr[1].imshow(fashion_mnist_results, cmap = 'Greys')
    axarr[0].set_title('MNIST')
    axarr[1].set_title('FASHION MNIST')

    if rearrange:
        axarr[0].set_xticks([])
        axarr[1].set_xticks([])
        axarr[0].set_yticks([])
        axarr[1].set_yticks([])
        plt.suptitle('Rearranged Elements of Psuedo-Sum States')

    else:
        axarr[0].set_xticks(range(10))
        axarr[1].set_xticks(range(10))
        axarr[0].set_yticks(range(10))
        axarr[1].set_yticks(range(10))

        axarr[0].set_xlabel('Prediction of sum state i')
        axarr[0].set_ylabel('Sorted index of sum state i')
        axarr[1].set_xlabel('Prediction of sum state i')
        axarr[1].set_ylabel('Prediction Index')

    #if rearrange:
    #    plt.savefig('figures/rearranged_sum_state_predictions_confusion_matrix.pdf')
    #else:
    #    plt.savefig('figures/sum_state_predictions_confusion_matrix.pdf')

    plt.show()

def compute_sum_state_confusion_matrix(bitstrings, rearrange = False):

    """
    MNIST
    """
    mnist_path = "mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_32/"

    mnist_sum_states = [load_qtn_classifier(mnist_path + f"digit_{i}") for i in range(10)]
    mnist_mps_sum_states = [s @ b for s, b in zip(mnist_sum_states, bitstrings)]

    mnist_sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in mnist_sum_states]
    mnist_ortho_classifier_data = adding_centre_batches(mnist_sum_states_data, 32, 10, orthogonalise = True)[0]
    mnist_ortho_mpo_classifier = data_to_QTN(mnist_ortho_classifier_data.data).squeeze()

    #Has shape (Class, Class_prediction)
    for i in mnist_mps_sum_states:
        state_i = (mnist_ortho_mpo_classifier @ i).squeeze()
        state_i /= state_i.norm()

    mnist_results = []
    for i in mnist_mps_sum_states:
        for j in mnist_mps_sum_states:
            if i == j:
                mnist_results.append(i.H @ j)
            else:
                mnist_results.append((i.H @ j).norm())



    """
    FASHION MNIST
    """
    fashion_mnist_path = "fashion_mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_32/"

    fashion_mnist_sum_states = [load_qtn_classifier(fashion_mnist_path + f"digit_{i}") for i in range(10)]
    fashion_mnist_mps_sum_states = [s @ b for s, b in zip(fashion_mnist_sum_states, bitstrings)]

    fashion_mnist_sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in fashion_mnist_sum_states]
    fashion_mnist_ortho_classifier_data = adding_centre_batches(fashion_mnist_sum_states_data, 32, 10, orthogonalise = True)[0]
    fashion_mnist_ortho_mpo_classifier = data_to_QTN(fashion_mnist_ortho_classifier_data.data).squeeze()

    #Has shape (Class, Class_prediction)
    fashion_mnist_results = []
    for i in fashion_mnist_mps_sum_states:
        state_i = (fashion_mnist_ortho_mpo_classifier @ i).squeeze()
        state_i /= state_i.norm()

    fashion_mnist_results = []
    for i in fashion_mnist_mps_sum_states:
        for j in fashion_mnist_mps_sum_states:
            if i == j:
                fashion_mnist_results.append(i.H @ j)
            else:
                fashion_mnist_results.append((i.H @ j).norm())

    """
    PLOT RESULTS
    """
    mnist_results = np.array(mnist_results).reshape(10,10)
    fashion_mnist_results = np.array(fashion_mnist_results).reshape(10,10)

    if rearrange:

        mnist_results = np.array([np.roll(i, -k) for k, i in enumerate(mnist_results)]).T

        #print(mnist_results[:,0])
        #print(np.roll(mnist_results[:,0], 1))
        #assert()
        """
        rearranged_mnist_results = np.zeros_like(mnist_results)
        rearranged_fashion_mnist_results = np.zeros_like(fashion_mnist_results)
        inds = diagonal_indices(rearranged_mnist_results.shape[0])

        mnist_results = np.sort(mnist_results.flatten(), axis = 0)[::-1]
        fashion_mnist_results = np.sort(fashion_mnist_results.flatten(), axis = 0)[::-1]

        for i, j, k in zip(inds, mnist_results, fashion_mnist_results):
            rearranged_mnist_results[i[0],i[1]] = j
            rearranged_fashion_mnist_results[i[0],i[1]] = k

        mnist_results = rearranged_mnist_results
        fashion_mnist_results = rearranged_fashion_mnist_results
        """
    plt.figure()

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(mnist_results, cmap = 'Greys')
    axarr[1].imshow(fashion_mnist_results, cmap = 'Greys')
    axarr[0].set_title('MNIST')
    axarr[1].set_title('FASHION MNIST')

    if rearrange:
        axarr[0].set_xticks([])
        axarr[1].set_xticks([])
        axarr[0].set_yticks([])
        axarr[1].set_yticks([])
        plt.suptitle('Rearranged Psuedo-Sum State Overlaps')
    else:
        axarr[0].set_xticks(range(10))
        axarr[1].set_xticks(range(10))
        axarr[0].set_yticks(range(10))
        axarr[1].set_yticks(range(10))

        axarr[0].set_xlabel('Sum state i')
        axarr[1].set_xlabel('Sum state i')
        axarr[0].set_ylabel('Sum state j')
        axarr[1].set_ylabel('Sum state j')

    #if rearrange:
    #    plt.savefig('figures/rearranged_sum_state_confusion_matrix.pdf')
    #else:
    #    plt.savefig('figures/sum_state_confusion_matrix.pdf')

    plt.show()

def diagonal_indices(n):
    """
    param: n: Dimension of matrix. 1-indexed
    Rearranges along the diagonal. Alternating between each side.
    Its janky but it works!
    """

    a = [[i, i] for i in range(n)]
    for j in range(1,n):
        inds = [str(i) + str(i+j) for i in range(n-j)]
        a = a + list(np.array([[[int(i[0]), int(i[1])], [int(i[1]), int(i[0])]] for i in inds]).reshape(-1, 2))

    return np.array(a)

def test_rearrangement(bitstrings):

    """
    MNIST
    """
    mnist_path = "mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_32/"

    mnist_sum_states = [load_qtn_classifier(mnist_path + f"digit_{i}") for i in range(10)]
    mnist_mps_sum_states = [s @ b for s, b in zip(mnist_sum_states, bitstrings)]

    mnist_sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in mnist_sum_states]
    mnist_ortho_classifier_data = adding_centre_batches(mnist_sum_states_data, 32, 10, orthogonalise = True)[0]
    mnist_ortho_mpo_classifier = data_to_QTN(mnist_ortho_classifier_data.data).squeeze()

    #Has shape (Class, Class_prediction)
    for i in mnist_mps_sum_states:
        state_i = (mnist_ortho_mpo_classifier @ i).squeeze()
        state_i /= state_i.norm()

    mnist_results = []
    for i in mnist_mps_sum_states:
        for j in mnist_mps_sum_states:
            if i == j:
                mnist_results.append(i.H @ j)
            else:
                mnist_results.append((i.H @ j).norm())


    """
    FASHION MNIST
    """
    fashion_mnist_path = "fashion_mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_32/"

    fashion_mnist_sum_states = [load_qtn_classifier(fashion_mnist_path + f"digit_{i}") for i in range(10)]
    fashion_mnist_mps_sum_states = [s @ b for s, b in zip(fashion_mnist_sum_states, bitstrings)]

    fashion_mnist_sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in fashion_mnist_sum_states]
    fashion_mnist_ortho_classifier_data = adding_centre_batches(fashion_mnist_sum_states_data, 32, 10, orthogonalise = True)[0]
    fashion_mnist_ortho_mpo_classifier = data_to_QTN(fashion_mnist_ortho_classifier_data.data).squeeze()

    #Has shape (Class, Class_prediction)
    fashion_mnist_results = []
    for i in fashion_mnist_mps_sum_states:
        state_i = (fashion_mnist_ortho_mpo_classifier @ i).squeeze()
        state_i /= state_i.norm()

    fashion_mnist_results = []
    for i in fashion_mnist_mps_sum_states:
        for j in fashion_mnist_mps_sum_states:
            if i == j:
                fashion_mnist_results.append(i.H @ j)
            else:
                fashion_mnist_results.append((i.H @ j).norm())
    """
    PLOT RESULTS
    """
    from scipy.optimize import linear_sum_assignment

    mnist_results = np.array(mnist_results).reshape(10,10)
    #fashion_mnist_results = np.array(fashion_mnist_results).reshape(10,10)

    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    s_results = np.zeros_like(mnist_results)
    for i in range(len(mnist_results)):
        for j in range(len(mnist_results)):
                if i == j:
                    continue
                else:
                    s_results[i,j] = mnist_results[i,j]


    graph = csr_matrix(s_results)

    inds = reverse_cuthill_mckee(graph, symmetric_mode = True)
    rearranged_mnist_results = mnist_results[inds]

    plt.imshow(rearranged_mnist_results, cmap = 'Greys')
    plt.show()
    assert()
    """
    #masked_results = fashion_mnist_results.copy()
    #for i in range(len(masked_results)):
    #    masked_results[i,i] = 0

    #row_ind, col_ind = linear_sum_assignment(masked_results, maximize = True)

    #ixgrid = np.ix_(row_ind, col_ind)
    #rearranged_fashion_mnist_results = fashion_mnist_results[ixgrid].T

    rearranged_mnist_results = [] #np.array([np.roll(i, -k) for k, i in enumerate(mnist_results)]).T
    rolling_inds = [4,3,2,1,0,-1,-2,-3,-4, -5]
    for i, j in zip(rolling_inds, mnist_results):
        rearranged_mnist_results.append(np.roll(j, i))

    rearranged_mnist_results = np.array(rearranged_mnist_results).T

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(2,5)
    z = 0
    for i in range(2):
        for j in range(5):
            results = np.array([np.roll(k,-z) for k in rearranged_mnist_results])

            results_rearranged = np.array([np.roll(q,-p) for q,p in zip(results.T, rolling_inds)])


            axarr[i,j].imshow(results_rearranged, cmap = 'Greys')
            axarr[i,j].set_xticks([])
            axarr[i,j].set_yticks([])
            z += 1
            axarr[i,j].set_title(f'{z}')
            #axarr[1].imshow(rearranged_mnist_results, cmap = 'Greys')
            #axarr[0].set_title('MNIST')
            #axarr[1].set_title('CYCLIC REARRANGED MNIST')

            #axarr[0].set_xticks([])
            #axarr[1].set_xticks([])
            #axarr[0].set_yticks([])
            #axarr[1].set_yticks([])
    plt.suptitle('MNIST Rearranged Psuedo-Sum State Overlaps')
    #plt.savefig('figures/mnist_rearranged_sum_state_confusion_matrix.pdf')
    #if rearrange:
    #    plt.savefig('figures/rearranged_sum_state_confusion_matrix.pdf')
    #else:
    #    plt.savefig('figures/sum_state_confusion_matrix.pdf')
    plt.tight_layout()
    plt.show()

def test_rearrangement_2(bitstrings):

    """
    MNIST
    """
    """
    mnist_path = "mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_32/"

    mnist_sum_states = [load_qtn_classifier(mnist_path + f"digit_{i}") for i in range(10)]
    mnist_mps_sum_states = [s @ b for s, b in zip(mnist_sum_states, bitstrings)]

    mnist_sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in mnist_sum_states]
    mnist_ortho_classifier_data = adding_centre_batches(mnist_sum_states_data, 32, 10, orthogonalise = True)[0]
    mnist_ortho_mpo_classifier = data_to_QTN(mnist_ortho_classifier_data.data).squeeze()

    #Has shape (Class, Class_prediction)
    for i in mnist_mps_sum_states:
        state_i = (mnist_ortho_mpo_classifier @ i).squeeze()
        state_i /= state_i.norm()

    mnist_results = []
    for i in mnist_mps_sum_states:
        for j in mnist_mps_sum_states:
            if i == j:
                mnist_results.append(i.H @ j)
            else:
                mnist_results.append((i.H @ j).norm())
    """

    """
    FASHION MNIST
    """
    fashion_mnist_path = "fashion_mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_32/"

    fashion_mnist_sum_states = [load_qtn_classifier(fashion_mnist_path + f"digit_{i}") for i in range(10)]
    fashion_mnist_mps_sum_states = [s @ b for s, b in zip(fashion_mnist_sum_states, bitstrings)]

    fashion_mnist_sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in fashion_mnist_sum_states]
    fashion_mnist_ortho_classifier_data = adding_centre_batches(fashion_mnist_sum_states_data, 32, 10, orthogonalise = True)[0]
    fashion_mnist_ortho_mpo_classifier = data_to_QTN(fashion_mnist_ortho_classifier_data.data).squeeze()

    #Has shape (Class, Class_prediction)
    fashion_mnist_results = []
    for i in fashion_mnist_mps_sum_states:
        state_i = (fashion_mnist_ortho_mpo_classifier @ i).squeeze()
        state_i /= state_i.norm()
        fashion_mnist_results.append(state_i)
    """
    fashion_mnist_results = []
    for i in fashion_mnist_mps_sum_states:
        for j in fashion_mnist_mps_sum_states:
            if i == j:
                fashion_mnist_results.append(i.H @ j)
            else:
                fashion_mnist_results.append((i.H @ j).norm())
    """

    """
    REARRANGE
    """
    #mnist_results = np.array(mnist_results).reshape(10,10)
    f_mnist_results = np.array(fashion_mnist_results).reshape(10,10)


    #a_fmnist = np.roll([7,5,9,8,2,4,6,0,3,1], 0)

    def brute_force_permutations(results):
        #Create weight matrix
        W = np.eye(10)
        for k, i in enumerate(np.arange(0.9, 0.0, -0.1)):
            stripe = [i for _ in range(10-(k+1))]
            W += np.diag(stripe, k+1)
            W += np.diag(stripe, -(k+1))

        from itertools import permutations
        #Brute force method.
        #Use weight matrix to determine how "diagonal" a matrix is
        perms = []
        norms = []
        i = 0
        for p in permutations(range(10)):
            rearranged_results = np.array([row[[p]] for row in results[[p]]])

            norms.append(np.linalg.norm((rearranged_results - W)))
            perms.append(p)

            #assert i == 2
            if i % 100:
                print(f'{i+1} out of {3628800}')
            i += 1

        np.save('permutations', perms)
        np.save('norms', norms)

    def load_brute_force_permutations(max_perm):
        norms = np.load('Classifiers/norms.npy')
        permutations = np.load('Classifiers/permutations.npy')

        best_norm_inds = np.argsort(norms)
        best_permutations = permutations[best_norm_inds]

        return best_permutations[:max_perm]

    #a_mnist = [2,4,6,0,8,3,1,9,5,7]
    a_fmnist = np.roll([7,5,9,8,2,4,6,0,3,1], 0)
    #brute_force_permutations(mnist_results)


    """
    rearranged_results = []
    for row in mnist_results[a_mnist]:
        rearranged_results.append(row[a_mnist])
    mnist_rearranged_results = np.array(rearranged_results)
    rearranged_results = []
    for row in results:
        rearranged_results.append(np.argsort(row))
    r = np.array(rearranged_results).T
    print(r)
    assert()
    """



    """
    rearranged_results = []
    for row in f_mnist_results[a_fmnist]:
        rearranged_results.append(row[a_fmnist])
    f_mnist_rearranged_results = np.array(rearranged_results)
    """


    f, axarr = plt.subplots(2,2)

    best_permutations = load_brute_force_permutations(100)
    best_permutations = [''.join(str(e) for e in i) for i in best_permutations]

    def createLenList(n,LL):
        stubs = {}
        for l in LL:
          for i,e in enumerate(l):
              stub = l[i:i+n]
              if len(stub) == n:
                 if stub not in stubs: stubs[stub]  = 1
                 else:                 stubs[stub] += 1

        return {k: stubs[k] for k in sorted(stubs, key=stubs.get, reverse=True)}

    """
    i = 0
    for j in range(2):
        for k in range(2):
            maxStub =  createLenList(i+2,best_permutations)
            axarr[j,k].bar(list(maxStub.keys())[:20], list(maxStub.values())[:20])
            axarr[j,k].tick_params(axis='x', labelrotation=90)
            axarr[j,k].set_title(f'Size:{i+2}')
            axarr[j,k].set_ylabel('Count')
            i += 1
    plt.suptitle('MNIST:\n Most common groups of labels out of 100 most diagonal permutations')
    plt.tight_layout()
    #plt.savefig('mnist_most_common_label_groups.pdf')
    plt.show()

    assert()
    """


    f, axarr = plt.subplots(2,5)
    best_permutations = load_brute_force_permutations(100)
    print(best_permutations[0])
    assert()
    """

    z = 0
    for i in range(2):
        for j in range(5):
            axarr[i,j].imshow([row[best_permutations[z]] for row in mnist_results[best_permutations[z]]], cmap = 'Greys')
            axarr[i,j].set_xticks(range(10))
            axarr[i,j].set_yticks(range(10))
            axarr[i,j].set_xticklabels(best_permutations[z])
            axarr[i,j].set_yticklabels(best_permutations[z])

            z += 1
    plt.suptitle('MNIST: Top 10 most diagonal permutations')
    plt.tight_layout()
    plt.savefig('mnist_top_10_diagonal_permutations.pdf')
    plt.show()
    assert()
    """
    """
    axarr[0,0].imshow(mnist_results, cmap = "Greys")
    axarr[0,1].imshow(mnist_rearranged_results.T, cmap = "Greys")
    axarr[0, 0].set_title('MNIST RESULTS')
    axarr[0, 1].set_title('MNIST REARRANGED RESULTS')
    axarr[0, 0].set_xticks(range(10))
    axarr[0, 1].set_xticks(range(10))
    axarr[0, 0].set_yticks(range(10))
    axarr[0, 1].set_yticks(range(10))

    axarr[0, 0].set_xticklabels(range(10))
    axarr[0, 1].set_xticklabels(a_mnist)
    axarr[0, 0].set_yticklabels(range(10))
    axarr[0, 1].set_yticklabels(a_mnist)

    """
    """
    rearranged_results = []
    for row in results:
        rearranged_results.append(np.argsort(row))
    r = np.array(rearranged_results).T
    print(r)
    assert()
    """

    """
    axarr[0].imshow(f_mnist_results, cmap = "Greys")
    axarr[1].imshow(f_mnist_rearranged_results.T, cmap = "Greys")
    axarr[0].set_title('FASHION MNIST RESULTS')
    axarr[1].set_title('FASHION MNIST REARRANGED RESULTS')
    axarr[0].set_xticks(range(10))
    axarr[1].set_xticks(range(10))
    axarr[0].set_yticks(range(10))
    axarr[1].set_yticks(range(10))

    axarr[0].set_xticklabels(range(10))
    axarr[1].set_xticklabels(a_fmnist)
    axarr[0].set_yticklabels(range(10))
    axarr[1].set_yticklabels(a_fmnist)

    plt.tight_layout()
    #plt.savefig('rearranged_psuedo_sum_states.pdf')
    plt.show()
    assert()
    """

def compute_sum_states_confusion_matrix(bitstrings):

    """
    FASHION MNIST
    """
    fashion_mnist_path = "fashion_mnist_mixed_sum_states/D_total/" + f"sum_states_D_total_32/"

    fashion_mnist_sum_states = [load_qtn_classifier(fashion_mnist_path + f"digit_{i}") for i in range(10)]
    fashion_mnist_mps_sum_states = [s @ b for s, b in zip(fashion_mnist_sum_states, bitstrings)]

    fashion_mnist_sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in fashion_mnist_sum_states]
    fashion_mnist_ortho_classifier_data = adding_centre_batches(fashion_mnist_sum_states_data, 32, 10, orthogonalise = True)[0]
    fashion_mnist_ortho_mpo_classifier = data_to_QTN(fashion_mnist_ortho_classifier_data.data).squeeze()

    #Has shape (Class, Class_prediction)
    fashion_mnist_results = []
    for i in fashion_mnist_mps_sum_states:
        state_i = (fashion_mnist_ortho_mpo_classifier @ i).squeeze()
        state_i /= state_i.norm()
        fashion_mnist_results.append(np.real(state_i.data[:10]))

    """
    fashion_mnist_results = []
    for i in fashion_mnist_mps_sum_states:
        for j in fashion_mnist_mps_sum_states:
            if i == j:
                fashion_mnist_results.append(i.H @ j)
            else:
                fashion_mnist_results.append((i.H @ j).norm())
    """

    f_mnist_results = np.array(fashion_mnist_results).reshape(10,10)
    a_fmnist = np.roll([7,5,9,8,2,4,6,0,3,1], 0)

    rearranged_results = []
    for row in f_mnist_results[a_fmnist]:
        rearranged_results.append(row[a_fmnist])
    f_mnist_rearranged_results = np.array(rearranged_results)

    f, axarr = plt.subplots(1,2)

    axarr[0].imshow(f_mnist_results, cmap = "Greys")
    axarr[1].imshow(f_mnist_rearranged_results.T, cmap = "Greys")
    axarr[0].set_title('FASHION MNIST RESULTS')
    axarr[1].set_title('FASHION MNIST REARRANGED RESULTS')
    axarr[0].set_xticks(range(10))
    axarr[1].set_xticks(range(10))
    axarr[0].set_yticks(range(10))
    axarr[1].set_yticks(range(10))

    axarr[0].set_xticklabels(range(10))
    axarr[1].set_xticklabels(a_fmnist)
    axarr[0].set_yticklabels(range(10))
    axarr[1].set_yticklabels(a_fmnist)

    plt.tight_layout()
    #plt.savefig('rearranged_psuedo_sum_states.pdf')
    plt.show()
    assert()

##################################################################################

def produce_psuedo_sum_states(dataset):
    x_train, y_train, x_test, y_test = load_data(
        100, shuffle=False, equal_numbers=True
    )
    bitstrings = create_experiment_bitstrings(x_train, y_train)
    #print(bitstrings[4].tensors[5].data)
    #assert()

    path = dataset + "_mixed_sum_states/D_total/" + f"sum_states_D_total_32/"

    sum_states = [load_qtn_classifier(path + f"digit_{i}") for i in range(10)]
    mps_sum_states = [s @ b for s, b in zip(sum_states, bitstrings)]

    sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in sum_states]
    ortho_classifier_data = adding_centre_batches(sum_states_data, 32, 10, orthogonalise = True)[0]
    ortho_mpo_classifier = data_to_QTN(ortho_classifier_data.data).squeeze()

    #Has shape (Class, Class_prediction)
    results = []
    for i in mps_sum_states:
        state_i = (ortho_mpo_classifier @ i).squeeze()
        state_i /= state_i.norm()
        results.append(np.real(state_i.data[:10]))
        #results.append(np.real(state_i.data))
    return np.array(results).reshape(10,10)
    overlaps = []
    for i in results:
        for j in results:
            overlaps.append(i.conj().T @ j)
    return np.array(overlaps).reshape(10,10)

def brute_force_permutations(overlaps,dataset):
    #Results shape (10,10)
    #Create weight matrix
    W = np.eye(10)
    for k, i in enumerate(np.arange(0.9, 0.0, -0.1)):
        stripe = [i for _ in range(10-(k+1))]
        W += np.diag(stripe, k+1)
        W += np.diag(stripe, -(k+1))

    from itertools import permutations
    #Brute force method.
    #Use weight matrix to determine how "diagonal" a matrix is
    perms = []
    norms = []
    i = 0
    for p in permutations(range(10)):
        rearranged_results = np.array([row[[p]] for row in overlaps[[p]]])

        norms.append(np.linalg.norm((rearranged_results - W)))
        perms.append(p)

        if i % 1000:
            print(f'{i+1} out of {3628800}')
        i += 1

    np.save('Classifiers/' + dataset + '_psuedo_overlap_permutations', perms)
    np.save('Classifiers/' + dataset + '_psuedo_overlap_norms', norms)

def load_brute_force_permutations(max_perm,dataset):
    norms = np.load('Classifiers/' + dataset + '_psuedo_overlap_norms.npy')
    permutations = np.load('Classifiers/' + dataset + '_psuedo_overlap_permutations.npy')

    best_norm_inds = np.argsort(norms)
    best_permutations = permutations[best_norm_inds]

    return best_permutations[:max_perm]

def most_common_label_groups(dataset):

    best_permutations = load_brute_force_permutations(100,dataset)
    best_permutations = [''.join(str(e) for e in i) for i in best_permutations]

    def createLenList(n,LL):
        stubs = {}
        for l in LL:
          for i,e in enumerate(l):
              stub = l[i:i+n]
              if len(stub) == n:
                 if stub not in stubs: stubs[stub]  = 1
                 else:                 stubs[stub] += 1

        return {k: stubs[k] for k in sorted(stubs, key=stubs.get, reverse=True)}

    f, axarr = plt.subplots(2,2)
    i = 0
    for j in range(2):
        for k in range(2):
            maxStub =  createLenList(i+2,best_permutations)
            axarr[j,k].bar(list(maxStub.keys())[:20], list(maxStub.values())[:20])
            axarr[j,k].tick_params(axis='x', labelrotation=90)
            axarr[j,k].set_title(f'Size:{i+2}')
            axarr[j,k].set_ylabel('Count')
            i += 1
    plt.suptitle(dataset + ' :\n Most common groups of labels out of 100 most diagonal permutations')
    plt.tight_layout()
    plt.savefig('figures/' + dataset + '_most_common_label_groups_psuedo_overlap.pdf')
    plt.show()

def most_diagonal_permutations(dataset):
    f, axarr = plt.subplots(2,5)
    best_permutations = load_brute_force_permutations(100,dataset)
    results = produce_psuedo_sum_states(dataset)
    z = 0
    for i in range(2):
        for j in range(5):
            axarr[i,j].imshow([row[best_permutations[z]] for row in results[best_permutations[z]]], cmap = 'Greys')
            axarr[i,j].set_xticks(range(10))
            axarr[i,j].set_yticks(range(10))
            axarr[i,j].set_xticklabels(best_permutations[z])
            axarr[i,j].set_yticklabels(best_permutations[z])

            z += 1
    plt.suptitle(dataset + ' : Top 10 most diagonal permutations')
    plt.tight_layout()
    plt.savefig('figures/' + dataset + '_top_10_diagonal_permutations_psuedo_overlap.pdf')
    plt.show()

def plot_confusion_matrix(dataset):

    states = produce_psuedo_sum_states(dataset)
    permutation = load_brute_force_permutations(10,dataset)[1]
    rearranged_results = []
    for row in states[permutation]:
        rearranged_results.append(row[permutation])
    rearranged_results = np.array(rearranged_results)

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(states, cmap = "Greys")
    axarr[1].imshow(rearranged_results.T, cmap = "Greys")
    axarr[0].set_title(dataset + ' RESULTS')
    axarr[1].set_title(dataset + ' REARRANGED RESULTS')
    axarr[0].set_xticks(range(10))
    axarr[1].set_xticks(range(10))
    axarr[0].set_yticks(range(10))
    axarr[1].set_yticks(range(10))

    axarr[0].set_xticklabels(range(10))
    axarr[1].set_xticklabels(permutation)
    axarr[0].set_yticklabels(range(10))
    axarr[1].set_yticklabels(permutation)

    plt.tight_layout()
    plt.savefig('figures/' + dataset + '_rearranged_psuedo_overlap.pdf')
    plt.show()



if __name__ == '__main__':
    print(produce_psuedo_sum_states('mnist').shape)
    assert()
    #x_train, y_train, x_test, y_test = load_data(
    #    100, shuffle=False, equal_numbers=True
    #)
    #bitstrings = create_experiment_bitstrings(x_train, y_train)
    #overlaps = produce_psuedo_sum_states('mnist')
    #brute_force_permutations(overlaps,'mnist')
    plot_confusion_matrix('mnist')
    #compute_predictions_confusion_matrix(bitstrings, rearrange = True)
    #test_rearrangement(bitstrings)
    #compute_sum_states_confusion_matrix(bitstrings)
    #plot_confusion_matrix('fashion_mnist')
