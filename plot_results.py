import matplotlib.pyplot as plt
import numpy as np

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



    x = range(2, 33, 2)
    plt.plot(x, non_ortho_test_accuracies[1])
    plt.show()
    assert()
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

    if rearrange:
        mnist_results = np.sort(mnist_results, axis = 0)
        fashion_mnist_results = np.sort(fashion_mnist_results, axis = 0)

    """
    for i in range(10):
        plt.plot(fashion_mnist_results[:,i][::-1], label = f'sum state {i+1}')

    plt.title('FASHION MNIST Sum State Index Magnitude')
    plt.ylabel('Value')
    plt.xlabel('Elements sorted from largest to smallest')
    plt.xticks(range(10))
    plt.savefig('fashion_mnist_sorted_sum_states_prediction_magnitude')
    plt.show()
    assert()
    plt.figure()
    """

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(mnist_results, cmap = 'Greys')
    axarr[1].imshow(fashion_mnist_results, cmap = 'Greys')
    axarr[0].set_title('MNIST')
    axarr[1].set_title('FASHION MNIST')
    axarr[0].set_xticks(range(10))
    axarr[0].set_yticks(range(10))
    #axarr[0].set_yticks([])
    axarr[0].set_xlabel('Prediction of sum state i')
    axarr[0].set_ylabel('Sorted index of sum state i')
    #axarr[0].set_ylabel('Rearranged Prediction Index')
    axarr[1].set_xticks(range(10))
    axarr[1].set_yticks(range(10))
    #axarr[1].set_yticks([])
    axarr[1].set_xlabel('Prediction of sum state i')
    axarr[1].set_ylabel('Prediction Index')
    #axarr[1].set_ylabel('Rearranged Prediction Index')
    #plt.savefig('figures/sum_state_predictions_confusion_matrix.pdf')
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
        mnist_results = np.sort(mnist_results, axis = 0)
        fashion_mnist_results = np.sort(fashion_mnist_results, axis = 0)

    plt.figure()

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(mnist_results, cmap = 'Greys')
    axarr[1].imshow(fashion_mnist_results, cmap = 'Greys')
    axarr[0].set_title('MNIST')
    axarr[1].set_title('FASHION MNIST')
    axarr[0].set_xticks(range(10))
    axarr[0].set_yticks(range(10))
    #axarr[0].set_yticks([])
    axarr[0].set_xlabel('Sum state i')
    axarr[0].set_ylabel('Sum state j')
    #axarr[0].set_ylabel('Rearranged overlap with sum state j')
    axarr[1].set_xticks(range(10))
    axarr[1].set_yticks(range(10))
    #axarr[1].set_yticks([])
    axarr[1].set_xlabel('Sum state i')
    axarr[1].set_ylabel('Sum state j')
    #axarr[1].set_ylabel('Rearranged overlap with sum state j')

    plt.savefig('figures/sum_state_confusion_matrix.pdf')
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data(
        100, shuffle=False, equal_numbers=True
    )
    bitstrings = create_experiment_bitstrings(x_train, y_train)

    #compute_predictions_confusion_matrix(bitstrings, rearrange = True)
    compute_sum_state_confusion_matrix(bitstrings, rearrange = False)
