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

np.set_printoptions(precision=4, linewidth=100000, suppress=True, threshold=np.inf)

from tqdm import tqdm
from functools import reduce
from stacking import partial_trace

Zmat = np.array([[1., 0.], [0., -1.]])
Xmat = np.array([[0., 1.], [1., 0.]])
Ymat = np.array([[0., -1.j], [1.j, 0.]])
Imat = np.identity(2)


def conj(unitary):
    return unitary.conjugate().transpose()


def swap_transpose_indices(transpose_order, index_1, index_2):
    transpose_order[index_2], transpose_order[index_1] = transpose_order[index_1], transpose_order[index_2]
    return transpose_order


def permute_transpose_indices(transpose_order, permutation_start):
    new_order = transpose_order[:permutation_start]
    to_be_permed = transpose_order[permutation_start:]
    n = len(to_be_permed)
    b = [[to_be_permed[i - j] for i in range(n)] for j in range(n)]
    new_order += b
    return transpose_order

#
# swap_transpose_indices(legs_order, 2 * (n_copies + 1) - 3, 2 * (
#         n_copies + 1) - 2)  # Swap 1st, 2nd upwards leg with last upwards leg


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
    if n_copies == 1:
        layers = [2, 1]
    elif n_copies == 2:
        layers = [3, 2]
    elif n_copies == 3:
        layers = [4, 2, 1]

    v_mats = [[[] for _ in range(n_units_per_layer)] for n_units_per_layer in layers]
    sqrt_D = 16
    D = 16

    proj = np.zeros((sqrt_D ** 2, sqrt_D ** 2))
    for i in range(sqrt_D):
        proj[i, i] = 1

    for l in tqdm(possible_labels):
        weighted_outer_states = np.zeros((dim_lc, dim_lc), dtype=complex)
        for i in tqdm(initial_label_qubits[y_train == l]):
            ket = i

            for k in range(n_copies):
                ket = np.kron(ket, i)

            outer = np.outer(ket.conj(), ket)
            weighted_outer_states += outer

        # First layer
        legs_order = [i for i in range(2 * (n_copies + 1))]

        weighted_outer_states_svd = weighted_outer_states.reshape(*[D] * (n_copies + 1), *[D] * (n_copies + 1))

        U, S, V = svd(weighted_outer_states_svd.reshape(D, -1))
        U_trunc = U[:, :sqrt_D]
        U_truncdag = conj(U_trunc)

        v_mats[0][0].append(U[:, :1])

        legs_order = swap_transpose_indices(legs_order, (n_copies + 1), -1)  # Swap first upwards with last upwards leg
        weighted_outer_states_svd_2 = (
                (U_truncdag @ weighted_outer_states_svd.reshape(D, -1)).reshape(*[sqrt_D] * 1,
                                                                                *[D] * (2 * (n_copies + 1) - 1))
                .transpose(legs_order) @ U_trunc).transpose(legs_order)
        legs_order = swap_transpose_indices(legs_order, (n_copies + 1), -1)  # Swap back

        legs_order = swap_transpose_indices(legs_order, 0, 1)  # Swap first downwards with next downwards leg
        weighted_outer_states_svd_2 = weighted_outer_states_svd_2.transpose(legs_order).reshape(D, -1)
        weighted_outer_states_svd_3 = weighted_outer_states_svd_2
        U, S, V = svd(weighted_outer_states_svd_3)

        U_trunc = U[:, :sqrt_D]
        U_truncdag = conj(U_trunc)
        v_mats[0][1].append(U[:, :1])

        if n_copies > 1:
            legs_order = swap_transpose_indices(legs_order, (n_copies + 2),
                                                -1)  # Swap second upwards with last upwards leg
        step_no = 2
        weighted_outer_states_svd_4 = (U_truncdag @ weighted_outer_states_svd_2).reshape(
            *[sqrt_D] * step_no, *[D] * (n_copies - 1),
            *[sqrt_D] * (step_no - 1), *[D] * n_copies).transpose(legs_order) @ U_trunc

        if n_copies > 1:
            legs_order = swap_transpose_indices(legs_order, (n_copies + 2), -1)  # Swap back
            legs_order = swap_transpose_indices(legs_order, 0, n_copies)  # Swap first downwards with next downwards leg

            weighted_outer_states_svd_4 = weighted_outer_states_svd_4.transpose(legs_order).reshape(D, -1)
            weighted_outer_states_svd_5 = weighted_outer_states_svd_4
            U, S, V = svd(weighted_outer_states_svd_5)

            U_trunc = U[:, :sqrt_D]
            U_truncdag = conj(U_trunc)
            v_mats[0][2].append(U[:, :1])

            step_no = 3
            weighted_outer_states_svd_6 = (U_truncdag @ weighted_outer_states_svd_4).reshape(
                *[sqrt_D] * step_no, *[D] * (n_copies - 2),
                *[sqrt_D] * (step_no - 1), *[D] * (n_copies - 1)).transpose(legs_order) @ U_trunc
            legs_order = swap_transpose_indices(legs_order, 0, n_copies)  # Swap back

            weighted_outer_states_svd_6 = weighted_outer_states_svd_6.reshape(D, -1)
            weighted_outer_states_svd_7 = weighted_outer_states_svd_6
            U, S, V = svd(weighted_outer_states_svd_7)
            U_trunc = U[:, :sqrt_D]
            U_truncdag = conj(U_trunc)
            v_mats[1][0].append(U[:, :1])

            legs_order = swap_transpose_indices(legs_order, 0, n_copies)  # Swap back
            step_no = 3
            weighted_outer_states_svd_6 = (U_truncdag @ weighted_outer_states_svd_6).reshape(
                *[sqrt_D] * step_no, *[D] * (n_copies - 2),
                *[sqrt_D] * (step_no - 1), *[D] * (n_copies - 1)).transpose(legs_order) @ U_trunc
            legs_order = swap_transpose_indices(legs_order, 2 * (n_copies + 1) - 2, 2 * (
                    n_copies + 1) - 1)  # Swap 1st, 2nd upwards leg with last upwards leg
            legs_order = swap_transpose_indices(legs_order, 2 * (n_copies + 1) - 3, 2 * (
                    n_copies + 1) - 2)  # Swap 1st, 2nd upwards leg with last upwards leg

            U = polar(weighted_outer_states_svd_4.reshape(sqrt_D ** 2, sqrt_D ** 2))[0]

            v_mats[1][0].append(U[:1, :])
        else:
            U = polar(weighted_outer_states_svd_4.reshape(sqrt_D ** 2, sqrt_D ** 2))[0]

            v_mats[1][0].append(U[:, :1])

    V_1 = np.array(v_mats[0][0])
    V_2 = np.array(v_mats[0][1])
    if n_copies > 1:
        V_3 = np.array(v_mats[0][2])
        V_4 = np.array(v_mats[1][0])
        V_5 = np.array(v_mats[1][1])
    else:
        V_3 = np.array(v_mats[1][0])

    c, d, e = V_1.shape
    V_1 = np.pad(V_1, ((0, dim_l - c), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(dim_l * e, d)
    # V_1 = np.pad(V_1, ((0, dim_l- c), (0, 0), (0, 0))).reshape(dim_l * d, e)

    f, g, h = V_2.shape
    V_2 = np.pad(V_2, ((0, dim_l - f), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(dim_l * h, g)
    # V_2 = np.pad(V_2, ((0, dim_l - f), (0, 0), (0, 0))).reshape(dim_l * g, h)

    l, m, p = V_3.shape
    # V_3 = np.pad(V_3, ((0, dim_l - l), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(dim_l * p, m)
    V_3 = np.pad(V_3, ((0, 256 - l), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(256 * p, m)
    #
    # V_3 = np.pad(V_3, ((0, 256 - l), (0, 0), (0, 0))).reshape(256 * m, p)
    # V_3 = np.pad(V_3, ((0, dim_l - l), (0, 0), (0, 0))).reshape(dim_l * m, p)

    print('V1', V_1.shape)
    print('V2', V_2.shape)
    print('V3', V_3.shape)
    print('Performing Polar Decomposition!')
    U_1 = polar(V_1)[0]
    U_2 = polar(V_2)[0]
    U_3 = polar(V_3)[0]
    print('U_1', U_1.shape)
    print('U_2', U_2.shape)
    print('U_3', U_3.shape)

    iden = np.identity(sqrt_D)
    iden_16 = np.identity(D)
    U_layer_1 = np.kron(U_1, U_2)
    # U_layer_2 = np.kron(np.kron(iden, U_3), iden)
    U_layer_2 = U_3

    print(U_layer_2.shape)
    # swap_1 = build_swap_matrix((n_copies + 1) * 4, [0, 2])
    # swap_2 = build_swap_matrix((n_copies + 1) * 4, [1, 3])
    # swap_1 = build_swap_matrix((n_copies + 1) * 4, [4, 6])
    # swap_2 = build_swap_matrix((n_copies + 1) * 4, [5, 7])

    # print(swap_1.shape)
    # print(swap_2.shape)

    # U_3_swapped =  swap_4 @ swap_3 @  swap_2 @ swap_1 @ U_layer_2 @ swap_1 @ swap_2 @ swap_3 @ swap_4
    # U_3_swapped = swap_2 @ swap_1 @ U_layer_2 @ swap_1 @ swap_2

    U = U_layer_2
        # @ U_3_swapped
    # U =  np.kron(iden, np.kron(iden, U_3)) @ np.kron(U_1, U_2)
    # U =  np.kron(iden_16, U_3) @ np.kron(U_1, U_2)
    # U =  np.kron(iden_16, U_3) @ np.kron(U_1, iden_16)

    # U = np.kron(U_1, U_2) @  np.kron(iden, np.kron(iden_16, iden))

    print('Finished Computing Stacking Unitary!')
    print(U_1.shape, U_2.shape, U_3.shape)

    np.save(f'U', U)

    # def swap_gate(a, b, n):
    #     M = [np.eye(2, dtype=U.dtype) for _ in range(n)]
    #     result = sparse.eye(2 ** n, dtype=U.dtype) - sparse.eye(2 ** n, dtype=U.dtype)
    #
    #     # Same as qiskit convention
    #     # a = n-a-1
    #     # b = n-b-1
    #
    #     for i in [[1, 0], [0, 1]]:
    #         for j in [[1, 0], [0, 1]]:
    #             M[a] = np.outer(i, j)  # |i><j|
    #             M[b] = np.outer(j, i)  # |j><i|
    #             swap_gate = sparse.csr_matrix(M[0])
    #             for m in M[1:]:
    #                 swap_gate = sparse.kron(swap_gate, sparse.csr_matrix(m))
    #             result += swap_gate
    #     return result
    #
    # I = np.eye(4 ** (n_copies + 1), dtype=U.dtype)
    # U_circ = sparse.kron(np.outer(I[0], I[0]), U)
    # for i in tqdm(I[1:]):
    #     U_circ += sparse.kron(sparse.csr_matrix(np.outer(i, i)), sparse.csr_matrix(I))
    #
    # for i in tqdm(range(1, n_copies + 1)):
    #     s_sparse = sparse.csr_matrix(
    #         swap_gate(2 + 4 * (i - 1), 2 + 4 * (i - 1) + 2 * n_copies - 2 * (i - 1), 4 * (n_copies + 1)))
    #     U_circ = s_sparse @ U_circ @ s_sparse
    #
    #     s_sparse = sparse.csr_matrix(
    #         swap_gate(2 + 4 * (i - 1) + 1, 2 + 4 * (i - 1) + 2 * n_copies - 2 * (i - 1) + 1, 4 * (n_copies + 1)))
    #     U_circ = s_sparse @ U_circ @ s_sparse
    # return U_circ

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


def get_stacking_unitary_mps(n_copies, dataset='mnist'):
    dir = f'data/{dataset}'

    U = np.load(f'U.npy')
    return U


#
# def lewis_unitaries():
#     from numpy import linalg as LA
#     print('Dataset: ', dataset)
#     dir = f'data/{dataset}'
#
#     # initial_label_qubits = np.load('Classifiers/' + dataset +
#     # '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle = True)[
#     # 'arr_0'][15] y_train = np.load('Classifiers/' + dataset +
#     # '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_labels.npy')
#     initial_label_qubits = \
#         np.load(f'{dir}/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle=True)['arr_0'][15]
#     y_train = np.load(f'{dir}/ortho_d_final_vs_training_predictions_labels.npy')
#     initial_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])
#     possible_labels = list(set(y_train))
#
#     dim_l = initial_label_qubits.shape[1]
#     outer_ket_states = initial_label_qubits
#
#     dim_lc = dim_l ** (1 + n_copies)
#     V_1 = np.load('U0s.npy')[:, :1, :]
#     V_2 = np.load('U1s.npy')[:, :1, :]
#     V_3 = np.load('U2s.npy')[:, :1, :]
#
#     # V_1 = np.array(v_mats[0][0])
#     # V_2 = np.array(v_mats[0][1])
#     # if n_copies > 1:
#     #     V_3 = np.array(v_mats[0][2])
#     #     V_4 = np.array(v_mats[1][0])
#     #     V_5 = np.array(v_mats[1][1])
#     # else:
#     #     V_3 = np.array(v_mats[1][0])
#     print(V_1.shape, V_2.shape, V_3.shape)
#
#     c, d, e = V_1.shape
#     # V_1 = np.pad(V_1, ((0, dim_l - c), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(dim_l * e, d)
#     V_1 = np.pad(V_1, ((0, dim_l- c), (0, 0), (0, 0))).reshape(dim_l * d, e)
#
#     f, g, h = V_2.shape
#     # V_2 = np.pad(V_2, ((0, dim_l - f), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(dim_l * h, g)
#     V_2 = np.pad(V_2, ((0, dim_l - f), (0, 0), (0, 0))).reshape(dim_l * g, h)
#
#     l, m, p = V_3.shape
#     # V_3 = np.pad(V_3, ((0, dim_l - l), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(dim_l * p, m)
#     # V_3 = np.pad(V_3, ((0, 256 - l), (0, 0), (0, 0))).transpose(0, 2, 1).reshape(256 * p, m)
#
#     V_3 = np.pad(V_3, ((0, 256 - l), (0, 0), (0, 0))).reshape(256 * m, p)
#     # V_3 = np.pad(V_3, ((0, dim_l - l), (0, 0), (0, 0))).reshape(dim_l * m, p)
#
#     print('V1', V_1.shape)
#     print('V2', V_2.shape)
#     print('V3', V_3.shape)
#     print('Performing Polar Decomposition!')
#     U_1 = polar(V_1)[0]
#     U_2 = polar(V_2)[0]
#     U_3 = polar(V_3)[0]
#     print('U_1', U_1.shape)
#     print('U_2', U_2.shape)
#     print('U_3', U_3.shape)
#
#     # U_layer_1 = np.kron(U_1, U_2)
#     U_layer_1 = np.kron(U_2, U_1)
#
#
#     # U_layer_2 = np.kron(np.kron(iden, U_3), iden)
#     U_layer_2 = U_3
#
#     print(U_layer_2.shape)
#     # swap_1 = build_swap_matrix((n_copies + 1) * 4, [0, 2])
#     # swap_2 = build_swap_matrix((n_copies + 1) * 4, [1, 3])
#     swap_1 = build_swap_matrix((n_copies + 1) * 4, [4, 6])
#     swap_2 = build_swap_matrix((n_copies + 1) * 4, [5, 7])
#
#     print(swap_1.shape)
#     print(swap_2.shape)
#
#     # U_3_swapped =  swap_4 @ swap_3 @  swap_2 @ swap_1 @ U_layer_2 @ swap_1 @ swap_2 @ swap_3 @ swap_4
#     U_3_swapped = swap_2 @ swap_1 @ U_layer_2 @ swap_1 @ swap_2
#
#     U = conj(U_layer_1)
#         # @ U_3_swapped
#
#     # U =  np.kron(iden, np.kron(iden, U_3)) @ np.kron(U_1, U_2)
#     # U =  np.kron(iden_16, U_3) @ np.kron(U_1, U_2)
#     # U =  np.kron(iden_16, U_3) @ np.kron(U_1, iden_16)
#
#     # U = np.kron(U_1, U_2) @  np.kron(iden, np.kron(iden_16, iden))
#
#     print('Finished Computing Stacking Unitary!')
#     print(U_1.shape, U_2.shape, U_3.shape)
#
#     np.save(f'U', U)


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
    # preds_U = np.array([np.diag(partial_trace(i, [4, 5, 6, 7])) for i in tqdm(preds_U)])

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

def kron_at_position(n_qubits, qubit_indices, unitaries):
    [unitary_0, unitary_1] = unitaries
    [qubit_idx_0, qubit_idx_1] = qubit_indices
    value = (n_qubits - 1) - qubit_idx_1
    x = [Imat] * qubit_idx_0 + [unitary_0] + [Imat] * (qubit_idx_1 - qubit_idx_0 - 1) + \
        [unitary_1] + \
        [Imat] * value
    # for mat in x:
    #     print('\n', mat)
    return reduce(np.kron,
                  [Imat] * qubit_idx_0 + [unitary_0] +
                  [Imat] * ((qubit_idx_1 - 1) - qubit_idx_0) +
                  [unitary_1] +
                  [Imat] * value)


def build_swap_matrix(n_qubits, qbs):
    # print('iterm')
    i_term = kron_at_position(n_qubits, qbs, [Imat, Imat])
    # print('xterm')

    x_term = kron_at_position(n_qubits, qbs, [Xmat, Xmat])
    # print('yterm')

    y_term = kron_at_position(n_qubits, qbs, [Ymat, Ymat])

    # print('zterm')
    z_term = kron_at_position(n_qubits, qbs, [Zmat, Zmat])

    swap = 1 / 2 * (i_term + x_term + z_term + y_term)
    return swap


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
    # lewis_unitaries()
    U = get_stacking_unitary_mps(n_copies, dataset=dataset)
    evaluate_stacking_unitary_mps(U, n_copies, dataset='mnist', training=False)
    # unitaries = [Xmat, Xmat]
    # n_qubits = 3
    # qubit_indices = [0, 2]
    # # U = kron_at_position(n_qubits, qubit_indices, unitaries)
    # # print(U)
    # swap = build_swap_matrix(n_qubits, qubit_indices)
    # print(swap)
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
