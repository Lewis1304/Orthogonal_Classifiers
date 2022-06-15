from xmps.svd_robust import svd
from scipy.linalg import polar
import numpy as np
from variational_mpo_classifiers import evaluate_classifier_top_k_accuracy, classifier_predictions
from plot_results import produce_psuedo_sum_states, load_brute_force_permutations
from tools import load_data
from scipy import sparse
import qutip
import matplotlib.pyplot as plt
from experiments import create_experiment_bitstrings
import tensorflow as tf

from tqdm import tqdm

from stacking import partial_trace


def conj(unitary):
    return unitary.conjugate().transpose()


def quantum_stacking_V_decomposition(n_copies, v_col=True, dataset='fashion_mnist',
                                     stacking_layer_idx=0):
    from numpy import linalg as LA
    print('Dataset: ', dataset)
    dir = f'data/{dataset}'

    # initial_label_qubits = np.load('Classifiers/' + dataset +
    # '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle = True)[
    # 'arr_0'][15] y_train = np.load('Classifiers/' + dataset +
    # '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_labels.npy')
    initial_label_qubits = \
        np.load(f'{dir}/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle=True)['arr_0'][15]
    y_train = np.load(f'{dir}/ortho_d_final_vs_training_predictions_labels.npy')
    initial_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])
    possible_labels = list(set(y_train))

    dim_l = initial_label_qubits.shape[1]
    outer_ket_states = initial_label_qubits

    dim_lc = dim_l ** (1 + n_copies)

    # .shape = n_train, dim_l**n_copies+1
    for k in range(n_copies):
        outer_ket_states = np.array([np.kron(i, j) for i, j in zip(outer_ket_states, initial_label_qubits)])

    # V = []
    V_1, V_2, V_3 = [[] for _ in range(3)]
    sqrt_D = 4
    D = 16
    proj = np.zeros((sqrt_D ** 2, sqrt_D ** 2))
    for i in range(sqrt_D):
        for j in range(sqrt_D):
            if i == j:
                proj[i, j] = 1

    for l in tqdm(possible_labels):
        weighted_outer_states = np.zeros((dim_lc, dim_lc), dtype=complex)
        for i in tqdm(initial_label_qubits[y_train == l]):
            ket = i

            for k in range(n_copies):
                ket = np.kron(ket, i)

            outer = np.outer(ket.conj(), ket)
            weighted_outer_states += outer

        # print('Performing SVD!')
        # First layer
        weighted_outer_states_svd = weighted_outer_states.reshape(*[D] * 2, *[D] * 2)
        print(weighted_outer_states_svd.shape)
        U, S, V = svd(weighted_outer_states_svd.reshape(D, -1))
        U_trunc = U[:, :sqrt_D]
        V_1.append(U)
        U_truncdag = conj(U_trunc)
        weighted_outer_states_svd_2 = (
                    (U_truncdag @ weighted_outer_states_svd.reshape(D, -1)).reshape(*[sqrt_D] * 1, *[D] * 3)
                    .transpose(0, 1, 3, 2) @ U_trunc).transpose(0, 1, 3, 2)

        weighted_outer_states_svd_2 = weighted_outer_states_svd_2.transpose(1, 0, 2, 3).reshape(D, -1)
        weighted_outer_states_svd_3 = weighted_outer_states_svd_2
        U, S, V = svd(weighted_outer_states_svd_3)
        U_trunc = U[:, :sqrt_D]
        # V_2 = U
        V_2.append(U)

        U_truncdag = conj(U_trunc)
        weighted_outer_states_svd_4 = (U_truncdag @ weighted_outer_states_svd_2).reshape(*[sqrt_D] * 3, *[D] * 1)
        weighted_outer_states_svd_4 = (weighted_outer_states_svd_4 @ U_trunc)
        #
        # weighted_outer_states_svd_4 = weighted_outer_states_svd_4.transpose(1, 0, 2, 3, 4, 5, 6, 7).reshape(sqrt_D, -1)
        # weighted_outer_states_svd_5 = weighted_outer_states_svd_4
        # U, S, V = svd(weighted_outer_states_svd_5)
        # U_trunc = U[:, :sqrt_D]
        # V_3 = U
        #
        # U_truncdag = conj(U_trunc)
        # weighted_outer_states_svd_6 = (U_truncdag @ weighted_outer_states_svd_4).reshape(*[sqrt_D] * sqrt_D, *[sqrt_D] * sqrt_D).transpose(1, 0, 2, 3, 4, 5, 6, 7)
        # weighted_outer_states_svd_6 = (weighted_outer_states_svd_6.transpose(0, 1, 2, 3, 4, 5, 7, 6) @ U_trunc).transpose(0, 1, 2, 3, 4, 5, 7, 6)
        #
        # weighted_outer_states_svd_6 = weighted_outer_states_svd_6.transpose(2, 1, 0, 3, 4, 5, 6, 7).reshape(sqrt_D, -1)
        # weighted_outer_states_svd_7 = weighted_outer_states_svd_6
        # U, S, V = svd(weighted_outer_states_svd_7)
        # U_trunc = U[:, :sqrt_D]
        # V_4 = U
        # U_truncdag = conj(U_trunc)
        # weighted_outer_states_svd_8 = (U_truncdag @ weighted_outer_states_svd_6).reshape(*[sqrt_D] * sqrt_D, *[sqrt_D] * sqrt_D).transpose(2, 1, 0, 3, 4, 5, 6, 7)
        # weighted_outer_states_svd_8 = (weighted_outer_states_svd_8.transpose(0, 1, 2, 3, 4, 7, 6, 5) @ U_trunc).transpose(0, 1, 2, 3, 4, 7, 6, 5)

        print(weighted_outer_states_svd_4.shape)
        # @ U @ proj
        V_3.append(weighted_outer_states_svd_4.reshape(sqrt_D ** 2, sqrt_D ** 2))
        # if v_col:
        #     # a = b = 16**n (using andrew's defn)
        #     a, b = U.shape
        #     p = int(np.log10(b)) - 1
        #     D_trunc = 16
        #     Vl = np.array(U[:, :b // 16] @ np.sqrt(np.diag(S)[:b // 16, :b // 16]))
        #     # Vl = np.array(U[:, :10**p] @ np.sqrt(np.diag(S)[:10**p, :10**p]))
        #     # Vl = np.array(U[:, :D_trunc] @ np.sqrt(np.diag(S)[:D_trunc, :D_trunc]))
        # else:
        #     Vl = np.array(U[:, :1] @ np.sqrt(np.diag(S)[:1, :1])).squeeze()

        # V.append(Vl)

    V_1 = np.array(V_1)
    V_2 = np.array(V_2)
    V_3 = np.array(V_3)
    print(V_1.shape)
    print(V_2.shape)
    print(V_3.shape)
    c, d, e = V_1.shape
    V_1 = np.pad(V_1, ((0, dim_l - c), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(dim_l * e, d)
    f, g, h = V_2.shape
    V_2 = np.pad(V_2, ((0, dim_l - f), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(dim_l * h, g)
    l, m, p = V_3.shape
    V_3 = np.pad(V_3, ((0, dim_l - l), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(dim_l * p, m)

    print('Performing Polar Decomposition!')
    U_1 = polar(V_1)[0]
    U_2 = polar(V_2)[0]
    U_3 = polar(V_3)[0]
    iden = np.identity(sqrt_D)
    U = np.kron(iden, np.kron(U_3, iden)) @ np.kron(U_1, U_2)
    print('Finished Computing Stacking Unitary!')
    print(U_1.shape, U_2.shape, U_3.shape)

    # for l in tqdm(possible_labels):
    #     weighted_outer_states = np.zeros((dim_lc, dim_lc), dtype=complex)
    #     for i in tqdm(initial_label_qubits[y_train == l]):
    #         ket = i
    #
    #         for k in range(n_copies):
    #             ket = np.kron(ket, i)
    #
    #         outer = np.outer(ket.conj(), ket)
    #         weighted_outer_states += outer
    #     sqrt_D = 4
    #     proj = np.zeros((sqrt_D**2, sqrt_D**2))
    #     for i in range(sqrt_D):
    #         for j in range(sqrt_D):
    #             if i == j:
    #                 proj[i, j] = 1
    #
    #     # print('Performing SVD!')
    #     #First layer
    #     weighted_outer_states_svd = weighted_outer_states.reshape(16, -1)
    #     U, S, V = svd(weighted_outer_states_svd)
    #     U_trunc = U[:, :sqrt_D]
    #     V_1.append(U)
    #     U_truncdag = conj(U_trunc)
    #     weighted_outer_states_svd_2 = ((U_truncdag @ weighted_outer_states_svd).reshape(-1, 16) @ U_trunc).reshape(sqrt_D, sqrt_D**2, sqrt_D, sqrt_D**2).transpose(1, 0, 3, 2)
    #     print(weighted_outer_states_svd_2)
    #     weighted_outer_states_svd_3 = weighted_outer_states_svd_2.reshape(16, -1)
    #     U, S, V = svd(weighted_outer_states_svd_3)
    #     U_trunc = U[:, :sqrt_D]
    #     V_2.append(U)
    #
    #     U_truncdag = conj(U_trunc)
    #     weighted_outer_states_svd_4 = ((U_truncdag @ weighted_outer_states_svd_2).reshape(-1, 16) @ U_trunc).reshape(sqrt_D, sqrt_D, sqrt_D, sqrt_D).transpose(1, 0, 3, 2)
    #
    #     print(weighted_outer_states_svd_4.shape)
    #                                   # @ U @ proj
    #     V_3.append(weighted_outer_states_svd_4.reshape(sqrt_D**2, sqrt_D**2))
    #     # if v_col:
    #     #     # a = b = 16**n (using andrew's defn)
    #     #     a, b = U.shape
    #     #     p = int(np.log10(b)) - 1
    #     #     D_trunc = 16
    #     #     Vl = np.array(U[:, :b // 16] @ np.sqrt(np.diag(S)[:b // 16, :b // 16]))
    #     #     # Vl = np.array(U[:, :10**p] @ np.sqrt(np.diag(S)[:10**p, :10**p]))
    #     #     # Vl = np.array(U[:, :D_trunc] @ np.sqrt(np.diag(S)[:D_trunc, :D_trunc]))
    #     # else:
    #     #     Vl = np.array(U[:, :1] @ np.sqrt(np.diag(S)[:1, :1])).squeeze()
    #
    #     # V.append(Vl)
    #
    # V_1 = np.array(V_1)
    # V_2 = np.array(V_2)
    # V_3 = np.array(V_3)
    # print(V_1.shape)
    # print(V_2.shape)
    # print(V_3.shape)
    # c, d, e = V_1.shape
    # V_1 = np.pad(V_1, ((0, dim_l - c), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(dim_l * e, d)
    # f, g, h = V_2.shape
    # V_2 = np.pad(V_2, ((0, dim_l - f), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(dim_l * h, g)
    # l, m, p = V_3.shape
    # V_3 = np.pad(V_3, ((0, dim_l - l), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(dim_l * p, m)
    #
    # print(V_1.shape, V_2.shape, V_3.shape)
    # # if v_col:
    # #     c, d, e = V.shape
    # #     # V = np.pad(V, ((0,dim_l - c), (0,0), (0,dim_l**p - D_trunc))).transpose(0, 2, 1).reshape(d , -1)
    # #     V = np.pad(V, ((0, dim_l - c), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(dim_l * e, d)
    # #
    # # else:
    # #     a, b = V.shape
    # #     V = np.pad(V, ((0, dim_l - a), (0, 0)))
    # # np.save('V', V)
    # print('Performing Polar Decomposition!')
    # U_1 = polar(V_1)[0]
    # U_2 = polar(V_2)[0]
    # U_3 = polar(V_3)[0]
    #
    # print('Finished Computing Stacking Unitary!')
    # print(U_1.shape, U_2.shape, U_3.shape)
    np.save(f'U', U.astype(np.float32))
    return U.astype(np.float32)


def get_stacking_unitary_mps(n_copies, dataset='mnist'):
    dir = f'data/{dataset}'

    U = np.load(f'U.npy')
    return U


def evaluate_stacking_unitary_mps(U, n_copies, dataset='fashion_mnist', training=False,
                                  ):
    """
    Evaluate Performance
    """
    # n_copies = int(np.log2(U.shape[0])//4)-1

    dir = f'data/{dataset}'
    # U = np.load(f'U.npy')

    if training:
        """
        Load Training Data
        """
        initial_label_qubits = \
            np.load(f'{dir}/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle=True)['arr_0'][15]
        y_train = np.load(f'{dir}/ortho_d_final_vs_training_predictions_labels.npy')
        prev_copies_string = 'initial_'

        # initial_label_qubits = np.load(f'data/{dataset}/ortho_d_final_vs_training_predictions_compressed.npz',
        # allow_pickle=True)['arr_0'][15].astype(np.float32) y_train = np.load(f'data/{
        # dataset}/ortho_d_final_vs_training_predictions_labels.npy').astype(np.float32) initial_label_qubits =
        # np.load('Classifiers/' + dataset +
        # '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle = True)[
        # 'arr_0'][15].astype(np.float32) y_train = np.load('Classifiers/' + dataset +
        # '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_labels.npy').astype(np.float32)

        """
        Rearrange test data to match new bitstring assignment
        """

        reassigned_preds = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])

        outer_ket_states = reassigned_preds
        # .shape = n_train, dim_l**n_copies+1
        for k in range(n_copies):
            outer_ket_states = [np.kron(i, j) for i, j in zip(outer_ket_states, reassigned_preds)]

        """
        Perform Overlaps
        """
        # We want qubit formation:
        # |l_0^0>|l_1^0>|l_0^1>|l_1^1> |l_2^0>|l_3^0>|l_2^1>|l_3^1>...
        # I.e. act only on first 2 qubits on all copies.
        # Since unitary is contructed on first 2 qubits of each copy.
        # So we want U @ SWAP @ |copy_preds>
        print('Performing Overlaps!')
        preds_U = np.array([abs(U.dot(i)) for i in tqdm(outer_ket_states)])

        """
        Trace out other qubits/copies
        """

        print('Performing Partial Trace!')
        preds_U = np.array([np.diag(partial_trace(i, [0, 1, 2, 3])) for i in tqdm(preds_U)])

        training_predictions = evaluate_classifier_top_k_accuracy(preds_U, y_train, 1)

        print()
        print('Training accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_train, 1))
        print('Training accuracy U:', training_predictions)
        print()
        hierarchy_string = f'{dir}/{prev_copies_string}{stacking_layer_idx}{n_copies}_'
        np.save(f'{hierarchy_string}ortho_d_final_vs_training_predictions.npy', preds_U)
        np.save(f'{hierarchy_string}ortho_d_final_vs_training_predictions_labels.npy', y_train)
        np.savetxt(f'{hierarchy_string}train_accuracy', np.array([training_predictions]), fmt='%0.4f')

    """
    Load Test Data
    """
    initial_label_qubits = np.load(f'{dir}/ortho_d_final_vs_test_predictions.npy', allow_pickle=True)[15]
    y_test = np.load(f'{dir}/ortho_d_final_vs_test_predictions_labels.npy')
    prev_copies_string = 'initial_'

    # initial_label_qubits = np.load(f'data/{dataset}/ortho_d_final_vs_test_predictions.npy', allow_pickle = True)[15]
    # # print(initial_label_qubits)
    # y_test = np.load(f'data/{dataset}/ortho_d_final_vs_test_predictions_labels.npy')
    # initial_label_qubits = np.load('Classifiers/' + dataset + '_mixed_sum_states/D_total/ortho_d_final_vs_test_predictions.npy')[15]#.astype(np.float32)
    # y_test = np.load('Classifiers/' + dataset + '_mixed_sum_states/D_total/ortho_d_final_vs_test_predictions_labels.npy')#.astype(np.float32)

    """
    Rearrange test data to match new bitstring assignment
    """

    reassigned_preds = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])

    outer_ket_states = reassigned_preds
    # .shape = n_train, dim_l**n_copies+1
    for k in range(n_copies):
        outer_ket_states = [np.kron(i, j) for i, j in zip(outer_ket_states, reassigned_preds)]

    """
    Perform Overlaps
    """
    # We want qubit formation:
    # |l_0^0>|l_1^0>|l_0^1>|l_1^1> |l_2^0>|l_3^0>|l_2^1>|l_3^1>...
    # I.e. act only on first 2 qubits on all copies.
    # Since unitary is contructed on first 2 qubits of each copy.
    # So we want U @ SWAP @ |copy_preds>
    print('Performing Overlaps!')
    preds_U = np.array([abs(U.dot(i)) for i in tqdm(outer_ket_states)])

    """
    Trace out other qubits/copies
    """
    print('Performing Partial Trace!')
    preds_U = np.array([np.diag(partial_trace(i, [0, 1, 2, 3])) for i in tqdm(preds_U)])

    test_predictions = evaluate_classifier_top_k_accuracy(preds_U, y_test, 1)
    np.save(f'hello_ortho_d_final_vs_test_predictions.npy', preds_U)
    np.save(f'hello_ortho_d_final_vs_test_predictions_labels.npy', y_test)
    np.savetxt(f'hello_test_accuracy', np.array([test_predictions]), fmt='%0.4f')

    print()
    print('Test accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_test, 1))
    print('Test accuracy U:', test_predictions)
    print()

    return None, test_predictions


# def test(n_copies, v_col = True, dataset = 'fashion_mnist'):
#     from numpy import linalg as LA
#     print('Dataset: ', dataset)
#
#     initial_label_qubits = np.load('Classifiers/' + dataset + '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle = True)['arr_0'][15]
#     y_train = np.load('Classifiers/' + dataset + '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_labels.npy')
#     initial_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])
#     possible_labels = list(set(y_train))
#
#     a, b, _, __ = load_data(
#         100, shuffle=False, equal_numbers=True
#     )
#     bitstrings = create_experiment_bitstrings(a, b)
#
#     dim_l = initial_label_qubits.shape[1]
#     dim_lc = dim_l ** (1 + n_copies)
#
#     weighted_outer_states = np.zeros((dim_lc, dim_lc))
#     for l in tqdm(possible_labels):
#         for i in tqdm(initial_label_qubits[y_train == l]):
#             ket = i
#
#             for k in range(n_copies):
#                 ket = np.kron(ket, i)
#
#             outer = np.outer(np.kron(bitstrings[l].squeeze().tensors[5].data, np.eye(dim_lc//16)[0]), ket)
#             #print(outer.shape)
#             weighted_outer_states += outer
#
#     """
#         #print('Performing SVD!')
#         U, S = svd(weighted_outer_states)[:2]
#         if v_col:
#             #a = b = 16**n (using andrew's defn)
#             a, b = U.shape
#             p = int(np.log10(b)) - 1
#             D_trunc = 16
#             Vl = np.array(U[:, :b//16] @ np.sqrt(np.diag(S)[:b//16, :b//16]))
#             #Vl = np.array(U[:, :10**p] @ np.sqrt(np.diag(S)[:10**p, :10**p]))
#             #Vl = np.array(U[:, :D_trunc] @ np.sqrt(np.diag(S)[:D_trunc, :D_trunc]))
#         else:
#             Vl = np.array(U[:, :1] @ np.sqrt(np.diag(S)[:1, :1])).squeeze()
#
#         V.append(Vl)
#
#     V = np.array(V)
#     if v_col:
#         c, d, e = V.shape
#         #V = np.pad(V, ((0,dim_l - c), (0,0), (0,dim_l**p - D_trunc))).transpose(0, 2, 1).reshape(d , -1)
#         V = np.pad(V, ((0,dim_l - c), (0,0), (0,0))).transpose(0, 2, 1).reshape(dim_l*e, d)
#
#     else:
#         a, b = V.shape
#         V = np.pad(V, ((0,dim_l - a), (0,0)))
#     #np.save('V', V)
#     """
#     print('Performing Polar Decomposition!')
#     U = polar(weighted_outer_states)[0]
#     print('Finished Computing Stacking Unitary!')
#     return U.astype(np.float32)


if __name__ == '__main__':
    # plot_confusion_matrix('mnist')
    # assert()
    # U = test(1, dataset = 'fashion_mnist')
    # classical_stacking()
    # assert()
    dataset = 'mnist'
    # n_copies_list = [1,1,1,1,1,1,1,1,1,1,1]
    n_copies = 1
    quantum_stacking_V_decomposition(n_copies, v_col=True, dataset=dataset)
    U = get_stacking_unitary_mps(n_copies, dataset=dataset)
    evaluate_stacking_unitary_mps(U, n_copies, dataset='mnist', training=False)
    # hierarchical_quantum_stacking(n_copies, v_col=True, dataset=dataset)

    # plot_confusion_matrix('fashion_mnist')
    # results = []
    # for i in range(1,10):
    #    print('NUMBER OF COPIES: ',i)
    #    U = specific_quantum_stacking(i, True)
    #    training_predictions, test_predictions = evaluate_stacking_unitary(U, True)
    #    results.append([training_predictions, test_predictions])
    #    #np.save('partial_stacking_results_2', results)
    # stacking_on_confusion_matrix(0, dataset = 'mnist')
    # U = delta_efficent_deterministic_quantum_stacking(1, dataset = 'mnist')
    # U = sum_state_deterministic_quantum_stacking(2, dataset = 'mnist')
    # evaluate_stacking_unitary(U, dataset = 'fashion_mnist')
