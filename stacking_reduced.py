#from svd_robust import svd
from numpy.linalg import svd
from scipy.linalg import polar
import numpy as np
import qutip
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Tools
"""


def load_data(dataset):
    initial_label_qubits = np.load(
        "data/" + dataset + "/new_ortho_d_final_vs_training_predictions.npy"
    )[15]
    y_train = np.load(
        "data/" + dataset + "/ortho_d_final_vs_training_predictions_labels.npy"
    )

    # Normalise predictions
    initial_label_qubits = np.array(
        [i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits]
    )

    return initial_label_qubits, y_train


def partial_trace(rho, qubit_2_keep):
    """ Calculate the partial trace for qubit system
    Parameters
    ----------
    rho: np.ndarray
        Density matrix
    qubit_2_keep: list
        Index of qubit to be kept after taking the trace
    Returns
    -------
    rho_res: np.ndarray
        Density matrix after taking partial trace
    """
    num_qubit = int(np.log2(rho.shape[0]))
    if num_qubit == 4:
        return np.outer(rho, rho.conj())
    else:
        rho = qutip.Qobj(rho, dims=[[2] * num_qubit, [1]])
        ptrace = rho.ptrace(qubit_2_keep)
        return ptrace


"""
Evaluation Functions
"""


def evaluate_stacking_unitary(U, dataset="fashion_mnist", verbose=1):
    n_copies = int(np.log2(U.shape[0]) // 4) - 1

    """
    Dummy function for verbose
    """
    if verbose == 0:
        verb = lambda x: x
    else:
        verb = tqdm

    """
    Load Test Data
    """
#    initial_label_qubits = np.load(
#        "data/" + dataset + "/new_ortho_d_final_vs_test_predictions.npy"
#    )[15]
#    y_test = np.load(
#        "data/" + dataset + "/ortho_d_final_vs_test_predictions_labels.npy"
#    )
    initial_label_qubits = np.load(
        "data/" + dataset + "/new_ortho_d_final_vs_training_predictions.npy"
    )[15]
    y_test = np.load(
        "data/" + dataset + "/ortho_d_final_vs_training_predictions_labels.npy"
    )

    initial_label_qubits = np.array(
        [i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits]
    )

    dim = initial_label_qubits[0].shape[0]

    for i in initial_label_qubits:
        assert np.allclose(np.linalg.norm(i), 1.0), "States not normalised"

    """
    Create copy states
    """
    outer_ket_states = initial_label_qubits
    # .shape = n_train, dim_l**n_copies+1
    for k in range(n_copies):
        outer_ket_states = [
            np.kron(i, j) for i, j in zip(outer_ket_states, initial_label_qubits)
        ]

    assert outer_ket_states[0].shape == (dim**(n_copies + 1),), "Wrong size"

    """
    Perform Overlaps
    """
    print("Performing Overlaps!")
#    preds_U = np.array([abs(U.dot(i)) for i in verb(outer_ket_states)])
    preds_U = np.array([U @ i for i in verb(outer_ket_states)])


    """
    Trace out other qubits/copies
    """
    print("Performing Partial Trace!")
    preds_U = np.array([np.diag(partial_trace(i, [0, 1, 2, 3])) for i in verb(preds_U)])

    test_predictions = evaluate_classifier_top_k_accuracy(preds_U, y_test, 1)
    print()
    print(
        "Test accuracy before:",
        evaluate_classifier_top_k_accuracy(initial_label_qubits, y_test, 1),
    )
    print("Test accuracy U:", test_predictions)
    print()

    return test_predictions


def evaluate_classifier_top_k_accuracy(predictions, y_test, k):
    top_k_predictions = [
        np.argpartition(image_prediction, -k)[-k:] for image_prediction in predictions
    ]
    results = np.mean([int(i in j) for i, j in zip(y_test, top_k_predictions)])
    return results


"""
Initialise stacking unitary V
"""


def initialise_V(n_copies, dataset="fashion_mnist", verbose=1):
    """
    args:
    n_copies: int: number of copies. STARTS FROM ZERO. e.g. n_copies=1 --> |\psi>^{\otimes m}
    dataset: string: mnist or fashion mnist
    verbose: int: Either 0 or 1. Shows progress bar for partial trace (1) or not (0)
    """

    print("Dataset: ", dataset)

    if verbose == 0:
        verb = lambda x: x
    else:
        verb = tqdm

    """
    Load data
    """
    initial_label_qubits, y_train = load_data(dataset)

    """
    Fixed parameters
    """
    possible_labels = list(set(y_train))
    dim_l = initial_label_qubits.shape[1]
    outer_ket_states = initial_label_qubits
    dim_lc = dim_l ** (1 + n_copies)

    """
    Construct V
    """
    V = []
    for l in verb(possible_labels):
        weighted_outer_states = np.zeros((dim_lc, dim_lc))
        for i in verb(initial_label_qubits[y_train == l]):

            # Construct copy matrix
            ket = i
            for k in range(n_copies):
                ket = np.kron(ket, i)
            outer = np.outer(ket.conj(), ket)

            # Add copy matrices (of same class) together
            weighted_outer_states += outer

        # Get eigenvectors (and singular values) of weighted outer states.
        U, S = svd(weighted_outer_states)[:2]

        # Keep 16**(n_copies) eigenvectors. E.g. for m=2 --> n_copies = 1.
        # Therefore keep 16.
        a, b = U.shape
        Vl = np.array(U[:, : b // 16])
        # Vl = np.array(U[:, :b//16] @ np.sqrt(np.diag(S)[:b//16,:b//16]))
        V.append(Vl)

    # Construct V from eigenvectors
    # Initially has shape [num_classes, 16**(n_copies+1) i.e. length of eigenvector, 16**(n_copies)]
    # num_classes padded to nearest multiple of 2. 10 ---> 16
    # V reshaped into ---> [num_classes_padded * 16**(n_copies), 16**(n_copies+1)]
    V = np.array(V)
    c, d, e = V.shape
    V = (
        np.pad(V, ((0, dim_l - c), (0, 0), (0, 0)))
        .transpose(0, 2, 1)
        .reshape(dim_l * e, d)
    )

    print("Performing Polar Decomposition!")
    U = polar(V)[0]
    print("Finished Computing Stacking Unitary!")
    return U


"""
Update stacking unitary
"""


def DMRG_update( initial_V, n_steps, experiment_name, dataset="mnist",store_test_accuracy=True, verbose=1):
    """
    args:
    initial_V: array: initialised stacking unitary
    n_steps: int: Number of training steps for DMRG scheme
    experiment_name: string: Saves results to file named experiment_name
    dataset: string: mnist or fashion mnist
    store_test_accuracy: bool: Whether to compute test accuracy during training or not
    verbose: int: Either 0 or 1. Shows progress bar for partial trace (1) or not (0)
    """

    """
    Load data
    """
    initial_label_qubits, y_train = load_data(dataset)

    """
    Fixed training parameters
    """
    possible_labels = list(set(y_train))
    n_copies = int(np.log2(initial_V.shape[0]) / 4) - 1
    pI = np.eye(initial_V.shape[0] // 16)
    alpha = 0.01

    """
    Compute rho_l and pL for all labels
    """
    pLs = [np.outer(np.eye(16)[int(l)], np.eye(16)[int(l)]) for l in possible_labels]
    weighted_outer_states = []
    for l in possible_labels:
        weighted_outer_state = np.zeros((initial_V.shape[0], initial_V.shape[0]))

        # Construct weighted sum (rho_l^m)
        for i in initial_label_qubits[y_train == l]:
            ket = i
            for k in range(n_copies):
                ket = np.kron(ket, i)
            outer = np.outer(ket.conj(), ket)
            weighted_outer_state += outer

        weighted_outer_state /= np.trace(weighted_outer_state)
        # weighted_outer_state *= (n_copies) * 16
        weighted_outer_states.append(weighted_outer_state)

    """
    Cost Function
    """
    # Has to be defined after rho_l and pLs are computed
    def C(V):
        return -sum_c_ll(V) + sum_c_ql(V)

    def sum_c_ll(V):
        C = [
            np.trace(V.conj().T @ np.kron(pLs[l], pI) @ V @ weighted_outer_states[l])
            for l in possible_labels
        ]
        return np.sum(C)

    def sum_c_ql(V):
        C = [
            [
                np.trace(
                    V.conj().T @ np.kron(pLs[q], pI) @ V @ weighted_outer_states[l]
                )
                for q in possible_labels
                if q != l
            ]
            for l in possible_labels
        ]
        return np.sum(C)

    """
    Gradient of Cost Function
    """

    def dC(V):
        return -sum_dC_ll(V) + sum_dC_ql(V)

    def sum_dC_ll(V):
        dC_ll = [
            np.kron(pI, pLs[l]) @ V @ weighted_outer_states[l] for l in possible_labels
        ]
        return np.sum(dC_ll, axis=0)

    def sum_dC_ql(V):
        dC_ql = [
            [
                np.kron(pI, pLs[q]) @ V @ weighted_outer_states[l]
                for q in possible_labels
                if q != l
            ]
            for l in possible_labels
        ]
        return np.sum(dC_ql, axis=(0, 1))

    """
    Compute initial accuracies/cost function
    """
    V1 = initial_V
    V2 = initial_V
    V3 = initial_V

    if store_test_accuracy:
        initial_accuracy = evaluate_stacking_unitary(
            initial_V, dataset=dataset, verbose=verbose
        )
        results1 = [initial_accuracy]
        results2 = [initial_accuracy]
        results3 = [initial_accuracy]

    initial_cost_function = C(initial_V)
    C1s = [initial_cost_function]
    C2s = [initial_cost_function]
    C3s = [initial_cost_function]

    """
    Update Step
    """
    for n in tqdm(range(n_steps)):

        dC1 = dC(V1)
        dC2 = dC(V2)
        dC3 = dC(V3)

        V1 = polar(V1 + alpha * polar(dC1)[0])[0]
        V2 = polar(V2 + alpha * dC2)[0]
        V3 = polar(dC3)[0]

        C1 = C(V1)
        C2 = C(V2)
        C3 = C(V3)

        if store_test_accuracy:
            results1.append(
                evaluate_stacking_unitary(V1, dataset=dataset, verbose=verbose)
            )
            results2.append(
                evaluate_stacking_unitary(V2, dataset=dataset, verbose=verbose)
            )
            results3.append(
                evaluate_stacking_unitary(V3, dataset=dataset, verbose=verbose)
            )

        C1s.append(C1)
        C2s.append(C2)
        C3s.append(C3)

        # Save accuracies
        np.save(
            f"update_V_results/{dataset}/test_accuracies/{experiment_name}_accuracies_1.npy",
            results1,
        )
        np.save(
            f"update_V_results/{dataset}/test_accuracies/{experiment_name}_accuracies_2.npy",
            results2,
        )
        np.save(
            f"update_V_results/{dataset}/test_accuracies/{experiment_name}_accuracies_3.npy",
            results3,
        )

        # Save cost function values
        np.save(
            f"update_V_results/{dataset}/cost_function/{experiment_name}_cost_function_1.npy",
            C1s,
        )
        np.save(
            f"update_V_results/{dataset}/cost_function/{experiment_name}_cost_function_2.npy",
            C2s,
        )
        np.save(
            f"update_V_results/{dataset}/cost_function/{experiment_name}_cost_function_3.npy",
            C3s,
        )


def stochastic_update(initial_V, n_steps, experiment_name, dataset="mnist", verbose=1):
    """
    args:
    initial_V: array: initialised stacking unitary
    n_steps: int: Number of training steps for DMRG scheme
    experiment_name: string: Saves results to file named experiment_name
    verbose: int: Either 0 or 1. Shows progress bar for partial trace (1) or not (0)
    """

    """
    Load data
    """
    initial_label_qubits, y_train = load_data(dataset)

    """
    Fixed training parameters
    """
    possible_labels = list(set(y_train))
    n_copies = int(np.log2(initial_V.shape[0]) / 4) - 1
    pI = np.eye(initial_V.shape[0] // 16)
    alpha = 0.01
    n_steps = int(n_steps)

    """
    Compute rho_l and pL for all labels
    """
    """
    pLs = [np.outer(np.eye(16)[int(l)], np.eye(16)[int(l)]) for l in possible_labels]
    weighted_outer_states = []
    for l in possible_labels:
        weighted_outer_state = np.zeros((initial_V.shape[0], initial_V.shape[0]))

        #Construct weighted sum (rho_l^m)
        for i in initial_label_qubits[y_train == l]:
            ket = i
            for k in range(n_copies):
                ket = np.kron(ket, i)
            outer = np.outer(ket.conj(), ket)
            weighted_outer_state += outer

        weighted_outer_state /= np.trace(weighted_outer_state)
        weighted_outer_states.append(weighted_outer_state)
    """

    """
    Cost Function
    """
    # Has to be defined after rho_l and pLs are computed
    """
    def C(V):
        return -sum_c_ll(V) + sum_c_ql(V)

    def sum_c_ll(V):
        C = [np.trace(abs(V.conj().T @ np.kron(pLs[l], pI) @ V @ weighted_outer_states[l])) for l in possible_labels]
        return np.mean(C)

    def sum_c_ql(V):
        C = [[np.trace(abs(V.conj().T @ np.kron(pLs[q], pI) @ V @ weighted_outer_states[l])) for q in possible_labels if q != l] for l in possible_labels]
        return np.mean(C)
    """
    kets = []
    for k in initial_label_qubits:
        ket = k
        for n in range(n_copies):
            ket = np.kron(ket, k)
        kets.append(ket)

    one_hot_labels = np.eye(16)

    def C(V):
        return np.mean(
            [
                np.linalg.norm(
                    np.diag(partial_trace(V @ ket, [0, 1, 2, 3]))
                    - one_hot_labels[label]
                )
                ** 2
                for ket, label in zip(kets, y_train)
            ]
        )

    """
    Compute initial accuracies/cost function
    """
    V_old = initial_V
    accuracies = [
        evaluate_stacking_unitary(V_old, dataset=dataset, verbose=verbose)
    ]
    C_old = C(V_old)
    Cs = [C_old]

    """
    Figure parameters
    """
    fig, ax = plt.subplots(1, 1)
    ax2 = ax.twinx()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost function")
    ax2.set_ylabel("Accuracy", color="r")
    # ax.plot([],[], label = r'$\Sigma_l c_{ll}$')
    # ax.plot([],[], label = r'$\Sigma_lq c_{ql}$')

    """
    Update steps
    """
    # c_ll_old = -sum_c_ll(V_old)
    # c_lls = [c_ll_old]
    # c_ql_old = sum_c_ql(V_old)
    # c_qls = [c_ql_old]
    for i in tqdm(range(n_steps)):

        # Random Matrix
        R = np.random.randn(*initial_V.shape)

        # Update V w/ random matrix
        V_new = polar(V_old + alpha * R)[0]
        # V_new = V_old + alpha*R

        # c_ll_new = -sum_c_ll(V_new)
        # c_ql_new = sum_c_ql(V_new)
        # C_new = c_ll_new + c_ql_new
        C_new = C(V_new)
        # C_new = c_ll_new

        # Keep V_new iff C_new is less than C_old
        if C_new <= C_old:
            tqdm.write(f"Updated V at step {i}")
            V_old = V_new
            C_old = C_new
            # c_ll_old = c_ll_new
            # c_ql_old = c_ql_new

        # c_lls.append(c_ll_old)
        # c_qls.append(c_ql_old)
        Cs.append(C_old)

        # Plot accuracy/fig every 100 steps
        # Figure should update during training, even when opened
        if i % 10 == 0 and i != 0:
            accuracies.append(
                evaluate_stacking_unitary(V_old, dataset=dataset, verbose=verbose)
            )

            ax.plot(Cs, color="tab:green")
            # ax.plot(c_lls, color = 'tab:blue')
            # ax.plot(c_qls, color = 'tab:orange')
            plt.cla()
            ax2.plot(accuracies, linewidth=0, marker=".", color="r", markersize="12")
            plt.savefig(f"figures/{dataset}_stochastic_training_{experiment_name}.pdf")
            plt.cla()
        else:
            accuracies.append(None)


"""
Plotting Functions
"""


def plot_update(experiment_name, dataset):

    initial_accuracy = 0.8033
    two_copy_accuracy = 0.8763

    fig, ax = plt.subplots(2, 1)

    V1_results = np.load(
        f"update_V_results/{dataset}/test_accuracies/{experiment_name}_accuracies_1.npy"
    )
    V2_results = np.load(
        f"update_V_results/{dataset}/test_accuracies/{experiment_name}_accuracies_2.npy"
    )
    V3_results = np.load(
        f"update_V_results/{dataset}/test_accuracies/{experiment_name}_accuracies_3.npy"
    )

    C1_results = np.load(
        f"update_V_results/{dataset}/cost_function/{experiment_name}_cost_function_1.npy"
    )
    C2_results = np.load(
        f"update_V_results/{dataset}/cost_function/{experiment_name}_cost_function_2.npy"
    )
    C3_results = np.load(
        f"update_V_results/{dataset}/cost_function/{experiment_name}_cost_function_3.npy"
    )

    x = range(len(V1_results))
    ax[0].plot(x, V1_results, color="b")
    ax[0].plot(x, V2_results, color="b", linestyle="dashed")
    ax[0].plot(x, V3_results, color="b", linestyle="dotted")
    ax[0].set_ylabel("Test Accuracy", color="b")
    ax[0].axhline(initial_accuracy, c="orange", alpha=0.4, label="Initial result")
    ax[0].axhline(
        two_copy_accuracy, c="green", alpha=0.4, label="Full Stacking 2 Copy Result"
    )

    ax[1].plot(
        x,
        C1_results,
        label=r"$V_{i+1} = Polar(V_i + \alpha * Polar[dC/dV_i])$",
        color="r",
    )
    ax[1].plot(
        x,
        C2_results,
        linestyle="dashed",
        label=r"$V_{i+1} = Polar(V_i + \alpha * dC/dV_i)$",
        color="r",
    )
    ax[1].plot(
        x,
        C3_results,
        linestyle="dotted",
        label=r"$V_{i+1} = Polar(dC/dV_i)$",
        color="r",
    )
    ax[1].set_ylabel("Cost Function", color="r")
    ax[1].set_xlabel("Iteration")

    ax[0].plot(
        [], [], label=r"$V_{i+1} = Polar(V_i + \alpha * Polar[dC/dV_i])$", color="grey"
    )
    ax[0].plot(
        [],
        [],
        linestyle="dashed",
        label=r"$V_{i+1} = Polar(V_i + \alpha * dC/dV_i)$",
        color="grey",
    )
    ax[0].plot(
        [], [], linestyle="dotted", label=r"$V_{i+1} = Polar(dC/dV_i)$", color="grey"
    )

    ax[0].legend()
    plt.suptitle(f"{dataset} DMRG update")
    plt.tight_layout()
    plt.savefig(f"{dataset}_DMRG_update_{experiment_name}.pdf")
    plt.show()
    assert ()


if __name__ == "__main__":

    n_copies = 1
    dataset = "mnist"
    experiment_name = "new_cost_func_mean"
    # plot_update(experiment_name, dataset)

#    initial_V = initialise_V(n_copies, dataset)
#    np.save("initialV_011122", initial_V)
    initial_V = np.load("initialV_011122.npy")
    print("Loaded saved initial_V")
    # DMRG_update(initial_V, 1000, experiment_name, dataset)
    stochastic_update(initial_V, 1e02, experiment_name, dataset)
