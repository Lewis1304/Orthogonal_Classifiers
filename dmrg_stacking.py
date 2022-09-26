from xmps.svd_robust import svd
from scipy.linalg import polar
import numpy as np
from plot_results import produce_psuedo_sum_states, load_brute_force_permutations
from scipy import sparse
import qutip
import matplotlib.pyplot as plt
from functools import reduce
from scipy.linalg import eig
from tqdm import tqdm
from ncon import ncon

from tools import load_qtn_classifier, data_to_QTN, arrange_data, load_data, bitstring_data_to_QTN
from fMPO import fMPO
from experiments import adding_centre_batches, label_last_site_to_centre, create_experiment_bitstrings, \
    centred_bitstring_to_qtn, prepare_centred_batched_classifier, adding_centre_batches
from deterministic_mpo_classifier import unitary_qtn
from variational_mpo_classifiers import evaluate_classifier_top_k_accuracy, classifier_predictions, mps_encoding, \
    create_hairy_bitstrings_data

import quimb.tensor as qtn
from quimb.tensor.tensor_core import rand_uuid
from oset import oset

from scipy.stats import unitary_group
from stacking import partial_trace, evaluate_stacking_unitary, delta_efficent_deterministic_quantum_stacking


def update_V(initial_V, n_copies, n_steps, dataset='mnist'):
    initial_label_qubits = np.load('data/' + dataset + '/new_ortho_d_final_vs_training_predictions.npy')[15].astype(
        np.float32)
    y_train = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_labels.npy').astype(np.float32)

    print('TRAINING ACCURACY BEFROE DMRG')
    evaluate_stacking_unitary(initial_V, n_copies, dataset='mnist', training=False)

    initial_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])
    possible_labels = list(set(y_train))
    # n_copies = int(np.log2(initial_V.shape[0]) / 4) - 1
    pI = np.eye(int(initial_V.shape[0] / 16), dtype=np.float32)

    alpha_results1 = []
    alpha_results2 = []

    # alpha_results3 = []
    # Update Step
    def tensorise_V(V_new, V_old_tens):
        return qtn.Tensor(data=V_new, inds=V_old_tens.inds)

    # def build_P():
    #     P = np.expand_dims(P, axis=2)
    #     P_tens = qtn.Tensor(data=P, inds=('i', 'j', 'l'))
    #     return P_tens
    #
    # def build_rho():
    #     rho = np.expand_dims(rho, axis=2)
    #
    #     rho_tens = qtn.Tensor(data=rho, inds=('p', 'q', 'l'))
    #
    #     return rho_tens
    for alpha in np.logspace(-4, -1, num=4):
        # for alpha in [1, 0.001, 0.0001]:
        # for alpha in [0.0001]:
        V1 = initial_V
        V_tens = qtn.Tensor(data=V1, inds=('u', 'v'))

        # V2 = initial_V
        # V3 = initial_V

        results1 = []
        results2 = []
        # results3 = []

        for n in tqdm(range(n_steps)):

            dV1 = np.zeros((initial_V.shape[0], initial_V.shape[0]), dtype=np.float32)
            dV2 = np.zeros((initial_V.shape[0], initial_V.shape[0]), dtype=np.float32)
            # dV3 = np.zeros((initial_V.shape[0], initial_V.shape[0]),dtype = np.float32)

            pL_all, rho_all = [], []
            for l in possible_labels:
                weighted_outer_states = np.zeros((initial_V.shape[0], initial_V.shape[0]), dtype=np.float32)
                pL = np.outer(np.eye(16, dtype=np.float32)[int(l)], np.eye(16, dtype=np.float32)[int(l)])
                pL_all.append(np.kron(pL, pI))
                for i in initial_label_qubits[y_train == l]:
                    ket = i
                    # single_copy_tensor = qtn.Tensor(weighted_outer_states, inds=('i', 's'))

                    for k in range(n_copies[0]):
                        ket = np.kron(ket, i)

                    outer = np.outer(ket.conj(), ket)
                    weighted_outer_states += outer
                rho_all.append(weighted_outer_states)
            pL_all = np.array(pL_all).transpose(1, 2, 0)
            rho_all = np.array(rho_all).transpose(1, 2, 0)

            P_tens = qtn.Tensor(data=pL_all, inds=('i', 'j', 'l'))

            rho_tens = qtn.Tensor(data=rho_all, inds=('p', 'q', 'l'))

            M = qtn.TensorNetwork([P_tens, rho_tens]) ^ ...
            M_data = sparse.csr_matrix(np.float32(M.data).reshape(initial_V.shape[0] ** 2, initial_V.shape[0] ** 2))
            eig_vals, eig_vecs = sparse.linalg.eigs(M_data)
            new_vec = (eig_vals[0] ** 2) * eig_vecs[:, 0]
            for eig, vec in zip(eig_vals[:, 1:], eig_vecs[:, 1:]):
                new_vec += (eig ** 2) * vec
                print(new_vec)
            print(eig_vals)
            print(M)
            # print('EIG VEC ROW',eig_vecs[0])
            # print('EIG VEC column',eig_vecs[:, 0])
            # dV1 += np.kron(pL, pI) @ V1 @ weighted_outer_states
            # dV2 += np.kron(pL, pI) @ V2 @ weighted_outer_states
            # dV3 += np.kron(pI, pL) @ V3 @ weighted_outer_states
            V1 = new_vec.reshape(initial_V.shape[0], initial_V.shape[0])

            # V1 = eig_vecs[:, 0].reshape(initial_V.shape[0], initial_V.shape[0])
            # V1 = polar(V1 + alpha * polar(dV1.conj().T)[0])[0]
            # V2 = polar(V2 + alpha * dV2.conj().T)[0]
            # V3 = polar(dV3.conj().T)[0]
            # print('\n\n\nV1 is', V1[:20, :20])

            _, test_results1 = evaluate_stacking_unitary(V1, n_copies, dataset='mnist', training=False)
            # _, test_results2 = evaluate_stacking_unitary(V2, n_copies, dataset='mnist', training=False)
            # _, test_results3 = evaluate_stacking_unitary(V3, n_copies,  dataset = 'mnist', training = False)

            # V1 = eig_vecs[:, 1].reshape(initial_V.shape[0], initial_V.shape[0])
            #
            # print('\n\n\nV1 is', V1[:20, :20])
            #
            # _, test_results1 = evaluate_stacking_unitary(V1, n_copies, dataset='mnist', training=False)
            #
            # V1 = eig_vecs[:, 2].reshape(initial_V.shape[0], initial_V.shape[0])
            #
            # print('\n\n\nV1 is', V1[:20, :20])
            #
            # _, test_results1 = evaluate_stacking_unitary(V1, n_copies, dataset='mnist', training=False)

            results1.append(test_results1)
            # results2.append(test_results2)
            # results3.append(test_results3)

        alpha_results1.append(results1)
        # alpha_results2.append(results2)
        # alpha_results3.append(results3)

        np.save(f'update_V_results/rearranged_gradient_update_results_1_n_copies_{n_copies[0]}', alpha_results1)
        # np.save(f'update_V_results/rearranged_gradient_update_results_2_n_copies_{n_copies[0]}', alpha_results2)
        # np.save(f'update_V_results/gradient_update_results_3_n_copies_{n_copies}', alpha_results3)


def get_M(initial_V, n_copies, n_steps, dataset='mnist'):
    initial_label_qubits = np.load('data/' + dataset + '/new_ortho_d_final_vs_training_predictions.npy')[15].astype(
        np.float32)
    y_train = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_labels.npy').astype(np.float32)

    # print('TRAINING ACCURACY BEFROE DMRG')
    # evaluate_stacking_unitary(initial_V, n_copies, dataset='mnist', training=False)

    initial_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])
    possible_labels = list(set(y_train))
    D = 16
    M_data = []
    m = n_copies[0]

    for l in possible_labels:
        weighted_outer_states = np.zeros((D, D), dtype=np.float32)

        for i_idx, i in enumerate(initial_label_qubits[y_train == l]):
            ket = i
            weighted_outer_states += np.outer(ket.conj(), ket)
        M_data.append(weighted_outer_states)

    M_data = np.array(M_data).transpose(1, 2, 0)
    print(M_data.shape)
    P = 16
    L = len(possible_labels)
    assert L == 10
    U_data, L_data = [[], []]
    Udata = np.zeros((D, P, L), dtype=np.float64)
    Ldata = np.zeros((D, L), dtype=np.float64)
    [ULPm, ELPm, ELLm] = [np.zeros((D ** (m+1), D, D), dtype=np.float64) for _ in range(3)]
    # M = sparse.csr_matrix(np.zeros((D ** (m+1) * D ** (m+1), D ** (m+1) * D ** (m+1)), dtype=np.float64))

    """ a) First we strip out the SVD data from Mdata """

    build=True
    if build:
        for l in range(L):
            MI = M_data[:, :, l]
            UI, LI, VI = np.linalg.svd(MI)

            Udata[:, :, l] = UI

            Ldata[:, l] = LI

        """ b) Next, construct the higher dimensional vectors ulp and elp """

        """ Udata[i,p,l] - u[i,l,p] in notation from notes """

        """ ELP[index,p,l] - |el ep^m>  in notation from notes """

        """ NB: the ordering is swapped in ELP to |ep^m el > """
        P = 16
        for l in range(L):

            el = np.eye(D)[:, l]

            for p in range(P):
                ULP1 = Udata[:, p, l]
                ULP = ULP1
                ep = np.eye(D)[:, p]
                ELL, ELP = el, el

                for k in range(0, m):
                    ELL = np.kron(el, ELL)
                    ULP = np.kron(ULP1, ULP)
                    ELP = np.kron(ep, ELP)


                ULPm[:, p, l] = ULP
                ELPm[:, p, l] = ELP
                ELLm[:, p, l] = ELL

        """print((ncon([np.conj(Udata[:,0,0]),Udata[:,0,1]],([1],[1])))**m)
        print(ncon([np.conj(ULPm[:,0,0]),ULPm[:,0,1]],([1],[1])))"""

        """ b) Construct enlarged M """
        elq_vec = sparse.csr_matrix(ELPm[:, 0, 0])
        elq_vec = elq_vec.reshape((elq_vec.shape[1], 1))
        ulp_vec = sparse.csr_matrix(ULPm[:, 0, 0])
        ulp_vec = ulp_vec.reshape((ulp_vec.shape[1], 1))

        # print(initial_V.shape)
        # print(elq_vec.shape)
        # print(ulp_vec.shape)
        # print(elq_vec.conj().T.shape)
        first_term = (elq_vec.conj().T @ initial_V @ ulp_vec)[0, 0]
        # EULPQm = np.kron(ELPm[:, 0, 0], ULPm[:, 0, 0])
        # EULPQm = sparse.csr_matrix(EULPQm.astype(np.float64))
        print(0, 0, 0)

        # EULPQm = sparse.csr_matrix(EULPQm)
        # x = sparse.csr_matrix(np.outer(EULPQm, EULPQm.conj().T))
        M = Ldata[0, 0] * first_term * sparse.csr_matrix(elq_vec.dot(ulp_vec.conj().T).astype(np.float64))

        # M = Ldata[0, 0] * first_term * sparse.csr_matrix(EULPQm.conj().T.dot(EULPQm).astype(np.float64))
        # V = sparse.csr_matrix(initial_V.reshape((initial_V.shape[0] * initial_V.shape[1], 1)))
        for l in range(L):
            print(l)

            for p in range(P):
                for q in range(1, P):
                    # EULPQm = sparse.csr_matrix(ELPm[:, q, l]), ULPm[:, p, l])
                    # print(EULPQm.shape)
                    # EULPQm = EULPQm.reshape((EULPQm.shape[0], 1))
                    # print(EULPQm.astype(np.float64).shape)
                    # print(EULPQm.astype(np.float64).conj().T.shape)
                    #
                    # EULPQm = sparse.csr_matrix(EULPQm.astype(np.float64))
                    # print(V.shape)
                    # rhs = EULPQm.conj().T @ V
                    elq_vec = sparse.csr_matrix(ELPm[:, q, l])
                    elq_vec = elq_vec.reshape((elq_vec.shape[1], 1))

                    ulp_vec = sparse.csr_matrix(ULPm[:, p, l])
                    ulp_vec = ulp_vec.reshape((ulp_vec.shape[1], 1))

                    first_term = (elq_vec.conj().T @ initial_V @ ulp_vec)[0, 0]
                    # print(first_term.shape)
                    # print(l, p, q)
                    # print(elq_vec.shape, ulp_vec.conj().T.shape)
                    # print(elq_vec.T.shape, ulp_vec.conj().shape)
                    #
                    # print(elq_vec.T.dot(ulp_vec.conj()).shape)
                    #
                    # print(elq_vec.dot(ulp_vec.conj().T).shape)
                    # print(sparse.csr_matrix(elq_vec.dot(ulp_vec.conj().T)))
                    # print(sparse.csr_matrix(elq_vec.dot(ulp_vec.conj().T).astype(np.float64)))
                    # print(first_term.shape)
                    # print(M.shape)
                    M = M + Ldata[p, l] * first_term * sparse.csr_matrix(elq_vec.dot(ulp_vec.conj().T).astype(np.float64))

                    # M = M + Ldata[p, l] * sparse.csr_matrix(rhs * EULPQm).astype(np.float64)
        # print(type(M))

        M = sparse.csr_matrix(M)
        sparse.save_npz(f'UPDATE_V_{m}', M)
    M = sparse.load_npz(f'UPDATE_V_{m}.npz')
    # print(M)

    return M


def get_V_directly(initial_V, n_copies, n_steps, dataset='mnist'):
    initial_label_qubits = np.load('data/' + dataset + '/new_ortho_d_final_vs_training_predictions.npy')[15].astype(
        np.float32)
    y_train = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_labels.npy').astype(np.float32)

    # print('TRAINING ACCURACY BEFROE DMRG')
    # evaluate_stacking_unitary(initial_V, n_copies, dataset='mnist', training=False)

    initial_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])
    possible_labels = list(set(y_train))
    D = 16
    M_data = []
    m = n_copies[0]

    for l in possible_labels:
        weighted_outer_states = np.zeros((D, D), dtype=np.float32)

        for i_idx, i in enumerate(initial_label_qubits[y_train == l]):
            ket = i
            weighted_outer_states += np.outer(ket.conj(), ket)
        M_data.append(weighted_outer_states)

    M_data = np.array(M_data).transpose(1, 2, 0)
    print(M_data.shape)
    P = 16
    L = len(possible_labels)
    assert L == 10
    U_data, L_data = [[], []]
    Udata = np.zeros((D, P, L), dtype=np.float64)
    Ldata = np.zeros((D, L), dtype=np.float64)
    [ULPm, ELPm, ELLm] = [np.zeros((D ** (m+1), D, D), dtype=np.float64) for _ in range(3)]
    # M = sparse.csr_matrix(np.zeros((D ** (m+1) * D ** (m+1), D ** (m+1) * D ** (m+1)), dtype=np.float64))

    """ a) First we strip out the SVD data from Mdata """

    build=True
    if build:
        for l in range(L):
            MI = M_data[:, :, l]
            UI, LI, VI = np.linalg.svd(MI)

            Udata[:, :, l] = UI

            Ldata[:, l] = LI

        """ b) Next, construct the higher dimensional vectors ulp and elp """

        """ Udata[i,p,l] - u[i,l,p] in notation from notes """

        """ ELP[index,p,l] - |el ep^m>  in notation from notes """

        """ NB: the ordering is swapped in ELP to |ep^m el > """
        P = 16
        for l in range(L):

            el = np.eye(D)[:, l]

            for p in range(P):
                ULP1 = Udata[:, p, l]
                ULP = ULP1
                ep = np.eye(D)[:, p]
                ELL, ELP = el, el

                for k in range(0, m):
                    ELL = np.kron(el, ELL)
                    ULP = np.kron(ULP1, ULP)
                    ELP = np.kron(ep, ELP)


                ULPm[:, p, l] = ULP
                ELPm[:, p, l] = ELP
                ELLm[:, p, l] = ELL

        """print((ncon([np.conj(Udata[:,0,0]),Udata[:,0,1]],([1],[1])))**m)
        print(ncon([np.conj(ULPm[:,0,0]),ULPm[:,0,1]],([1],[1])))"""

        """ b) Construct enlarged M """
        elq_vec = sparse.csr_matrix(ELPm[:, 0, 0])
        elq_vec = elq_vec.reshape((elq_vec.shape[1], 1))
        ulp_vec = sparse.csr_matrix(ULPm[:, 0, 0])
        ulp_vec = ulp_vec.reshape((ulp_vec.shape[1], 1))

        # print(initial_V.shape)
        # print(elq_vec.shape)
        # print(ulp_vec.shape)
        # print(elq_vec.conj().T.shape)
        # first_term = (elq_vec.conj().T @ initial_V @ ulp_vec)[0, 0]
        # EULPQm = np.kron(ELPm[:, 0, 0], ULPm[:, 0, 0])
        # EULPQm = sparse.csr_matrix(EULPQm.astype(np.float64))
        print(0, 0, 0)

        # EULPQm = sparse.csr_matrix(EULPQm)
        # x = sparse.csr_matrix(np.outer(EULPQm, EULPQm.conj().T))
        # M = Ldata[0, 0] * first_term * sparse.csr_matrix(elq_vec.dot(ulp_vec.conj().T).astype(np.float64))

        # M = Ldata[0, 0] * first_term * sparse.csr_matrix(EULPQm.conj().T.dot(EULPQm).astype(np.float64))
        # V = sparse.csr_matrix(initial_V.reshape((initial_V.shape[0] * initial_V.shape[1], 1)))
        V_new = elq_vec.dot(ulp_vec.conj().T)
        for l in range(L):
            print(l)
            for p in range(P):
                for q in range(1, P):
                    # EULPQm = sparse.csr_matrix(ELPm[:, q, l]), ULPm[:, p, l])
                    # print(EULPQm.shape)
                    # EULPQm = EULPQm.reshape((EULPQm.shape[0], 1))
                    # print(EULPQm.astype(np.float64).shape)
                    # print(EULPQm.astype(np.float64).conj().T.shape)
                    #
                    # EULPQm = sparse.csr_matrix(EULPQm.astype(np.float64))
                    # print(V.shape)
                    # rhs = EULPQm.conj().T @ V
                    elq_vec = sparse.csr_matrix(ELPm[:, q, l])
                    elq_vec = elq_vec.reshape((elq_vec.shape[1], 1))

                    ulp_vec = sparse.csr_matrix(ULPm[:, p, l])
                    ulp_vec = ulp_vec.reshape((ulp_vec.shape[1], 1))

                    # print(first_term.shape)
                    # print(elq_vec.shape, ulp_vec.conj().T.shape)
                    # print(elq_vec.T.shape, ulp_vec.conj().shape)
                    #
                    # print(elq_vec.T.dot(ulp_vec.conj()).shape)
                    #
                    # print(elq_vec.dot(ulp_vec.conj().T).shape)
                    # print(sparse.csr_matrix(elq_vec.dot(ulp_vec.conj().T)))
                    # print(sparse.csr_matrix(elq_vec.dot(ulp_vec.conj().T).astype(np.float64)))
                    # print(first_term.shape)
                    # print(M.shape)
                    V_new = V_new + sparse.csr_matrix(elq_vec.dot(ulp_vec.conj().T))
                    # M = M + Ldata[p, l] * first_term * sparse.csr_matrix(elq_vec.dot(ulp_vec.conj().T).astype(np.float64))

        V = sparse.csr_matrix(V_new)
        sparse.save_npz(f'UPDATE_V_{m}', V)
    V = sparse.load_npz(f'UPDATE_V_{m}.npz')
    # print(M)

    return V


def get_C(MV, V):
    """ Calulate Cost Function """
    return np.trace(V.conj().T @ MV)


# def get_VDMRG(Vseed, M):
def get_VDMRG(MV, V):

    """ Returns V using DMRG: this works best   """

    # V = Vseed

    Niterations = 20
    all_vs = []
    for i in range(1, Niterations):
        # X, L, Y = np.linalg.svd(ncon([M, V], ([-1, -2, 1, 2], [1, 2])))
        X, L, Y = np.linalg.svd(MV)
        # X, L, Y = sparse.linalg.svd(MV.reshape(D ** (1+m), D ** (1+m)))

        """rather than one step we have introduced a finite learning rate """

        V = X @ Y + 2.0 * V

        X, L, Y = np.linalg.svd(V)

        V = X @ Y

        C = get_C(MV, V)
        print(C)
        all_vs.append(V)
        # print(f'C iteration {i} = {C}')

    return all_vs

if __name__ == '__main__':
    n_copies = [2]
    # plot_update_V()
    # initial_V = delta_efficent_deterministic_quantum_stacking(n_copies, v_col=True, dataset='mnist')
    # initial_V = polar(np.random.randn(16 ** (1 + n_copies[0]), 16 ** (1 + n_copies[0]))
    #                   + 1.0j * np.random.randn(16 ** (1 + n_copies[0]), 16 ** (1 + n_copies[0])))[0]
    D = 16

    # mmax = 7

    m = n_copies[0]
    initial_V = np.eye(D**(m+1))
    print('COMPUTE MV FOR INITIAL V')

    # evaluate_stacking_unitary(initial_V, n_copies, dataset='mnist', training=False)
    # print('COMPUTE MV FOR INITIAL V')
    # MV = get_M(initial_V, n_copies, 10)
    # MV = MV.todense().reshape((D ** (1 + m), D ** (1 + m)))


    print('GET V DIRECTLY FOR INITIAL V')
    V_directly = get_V_directly(initial_V, n_copies, 10)

    MV_directly = get_M(V_directly, n_copies, 10)
    MV_directly = MV_directly.todense().reshape((D ** (1 + m), D ** (1 + m)))

    V_directly = V_directly.todense().reshape((D ** (1 + m), D ** (1 + m)))
    # print(np.array(V_directly).shape)
    # cost_for_optimal_V = get_C(MV_directly, V_directly)
    # print('OPTIMAL V COST', cost_for_optimal_V)

    evaluate_stacking_unitary(np.array(V_directly), n_copies, dataset='mnist', training=False)

    # print(MV.todense().shape)
    # print(D)
    # M, _, _ = get_MV(Mdata, D, m)
    # used_V = initial_V.reshape((D ** (1+m) * D ** (1+m), 1))
    #
    # used_V = sparse.csr_matrix(used_V)
    # MV = M @ used_V
    # M = M.reshape([D ** (1+m), D ** (1+m), D ** (1+m), D ** (1+m)])

    """C = get_C(M,V,D**m)

    print(C) this is always equal to D so no need to print"""

    """Copt = get_optC(M,D**m,50000)

    print(Copt)"""
    # MV = MV.todense().reshape((D ** (1+m), D ** (1+m)))
    # all_vs = get_VDMRG(MV, initial_V)

    print('OPTIMISE DIRECTLY')
    all_vs_directly = get_VDMRG(MV_directly, V_directly)

    i=0
    # for V in all_vs:
    #     print(i)
    #     VDMRG = np.array(V)
    #     evaluate_stacking_unitary(VDMRG, n_copies, dataset='mnist', training=False)
    #     i += 1
    # for V in all_vs_directly:
    #     print(i)
    #     VDMRG = np.array(V)
    #     evaluate_stacking_unitary(VDMRG, n_copies, dataset='mnist', training=False)
    #     i += 1
    # CDMRG = get_C(M, VDMRG)

    # print(CDMRG)
    assert ()

    # for k in range(n_copies[0]):
    #     ket = np.kron(ket, i)
    # single_copy_tensor = qtn.Tensor(weighted_outer_states, inds=('i', 's'))
    #
    #         for k in range(n_copies[0]):
    #             ket = np.kron(ket, i)
    #
    #         outer = np.outer(ket.conj(), ket)
    #         weighted_outer_states += outer
    #     rho_all.append(weighted_outer_states)
    # pL_all = np.array(pL_all).transpose(1, 2, 0)
    # rho_all = np.array(rho_all).transpose(1, 2, 0)
    #
    # P_tens = qtn.Tensor(data=pL_all, inds=('i', 'j', 'l'))
    #
    # rho_tens = qtn.Tensor(data=rho_all, inds=('p', 'q', 'l'))
    #
    # M = qtn.TensorNetwork([P_tens, rho_tens]) ^ ...
    # M_data = sparse.csr_matrix(np.float32(M.data).reshape(initial_V.shape[0] **2, initial_V.shape[0] ** 2))
    # eig_vals, eig_vecs = sparse.linalg.eigs(M_data)
    # new_vec = (eig_vals[0] ** 2) * eig_vecs[:,0]
    # for eig, vec in zip(eig_vals[:,1:], eig_vecs[:,1:]):
    #     new_vec += (eig ** 2) * vec
    #     print(new_vec)
    # print(eig_vals)
    # print(M)
    # print('EIG VEC ROW',eig_vecs[0])
    # print('EIG VEC column',eig_vecs[:, 0])
    # dV1 += np.kron(pL, pI) @ V1 @ weighted_outer_states
    # dV2 += np.kron(pL, pI) @ V2 @ weighted_outer_states
    # dV3 += np.kron(pI, pL) @ V3 @ weighted_outer_states
    # V1 = new_vec.reshape(initial_V.shape[0], initial_V.shape[0])

    # V1 = eig_vecs[:, 0].reshape(initial_V.shape[0], initial_V.shape[0])
    # V1 = polar(V1 + alpha * polar(dV1.conj().T)[0])[0]
    # V2 = polar(V2 + alpha * dV2.conj().T)[0]
    # V3 = polar(dV3.conj().T)[0]
    # print('\n\n\nV1 is', V1[:20, :20])

    # _, test_results1 = evaluate_stacking_unitary(V1, n_copies, dataset='mnist', training=False)
    # _, test_results2 = evaluevaluate_stacking_unitaryate_stacking_unitary(V2, n_copies, dataset='mnist', training=False)

    # V1 = eig_vecs[:, 1].reshape(initial_V.shape[0], initial_V.shape[0])
    #
    # print('\n\n\nV1 is', V1[:20, :20])
    #
    # _, test_results1 = evaluate_stacking_unitary(V1, n_copies, dataset='mnist', training=False)
    #
    # V1 = eig_vecs[:, 2].reshape(initial_V.shape[0], initial_V.shape[0])
    #
    # print('\n\n\nV1 is', V1[:20, :20])
    #
    # _, test_results1 = evaluate_stacking_unitary(V1, n_copies, dataset='mnist', training=False)

    # results1.append(test_results1)
    # results2.append(test_results2)

    # alpha_results1.append(results1)
    # alpha_results2.append(results2)

    # np.save(f'update_V_results/rearranged_gradient_update_results_1_n_copies_{n_copies[0]}', alpha_results1)
    # np.save(f'update_V_results/rearranged_gradient_update_results_2_n_copies_{n_copies[0]}', alpha_results2)


def plot_update_V():
    V1_results = np.load('update_V_results/gradient_update_results_1_n_copies_1.npy')
    V2_results = np.load('update_V_results/gradient_update_results_2_n_copies_1.npy')
    V3_results = np.load('update_V_results/gradient_update_results_3_n_copies_1.npy')

    two_copy1 = np.load('update_V_results/gradient_update_results_1_n_copies_2.npy')
    two_copy2 = np.load('update_V_results/gradient_update_results_2_n_copies_2.npy')

    V1_results_max = [max(i) for i in V1_results]
    V2_results_max = [max(i) for i in V2_results]
    V3_results_max = [max(i) for i in V3_results]
    two_copy_1_results_max = [max(i) for i in two_copy1]
    two_copy_2_results_max = [max(i) for i in two_copy2]

    best_V1_results = V1_results[np.argmax(V1_results_max)]
    best_V2_results = V2_results[np.argmax(V2_results_max)]
    best_V3_results = V3_results[np.argmax(V3_results_max)]
    best_two_copy_1_results = two_copy1[np.argmax(two_copy_1_results_max)]
    best_two_copy_2_results = two_copy2[np.argmax(two_copy_2_results_max)]

    x = range(1, 11)
    alphas = np.logspace(-4, 1, num=6)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    # fig, ax1 = plt.subplots(1, 1)

    ax1.plot(x, best_V1_results,
             label=fr'$V = Polar(V + \alpha * Polar[dV]) \ ,\alpha:{np.round(alphas[np.argmax(V1_results_max)], 5)}$')
    ax1.plot(x, best_V2_results, linestyle='dashed',
             label=fr'$V = Polar(V + \alpha * dV) \ ,\alpha:{np.round(alphas[np.argmax(V2_results_max)], 5)}$')
    ax1.plot(x, best_V3_results, linestyle='dotted',
             label=fr'$V = Polar(dV) \ ,\alpha:{np.round(alphas[np.argmax(V3_results_max)], 5)}$')
    ax1.axhline(0.8565, c='k', alpha=0.4, label='Full Stacking 2 Copy Result')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('2 Copy Case. Best result plotted from ' + r'$\alpha = [0.0001,10]$')
    ax1.legend()

    alphas = np.logspace(-4, 0, num=5)
    x = range(1, 6)
    ax2.plot(x, best_two_copy_1_results,
             label=fr'$V = Polar(V + \alpha * Polar[dV]) \ ,\alpha:{np.round(alphas[np.argmax(two_copy_1_results_max)], 5)}$')
    ax2.plot(x, best_two_copy_2_results, linestyle='dashed',
             label=fr'$V = Polar(V + \alpha * dV) \ ,\alpha:{np.round(alphas[np.argmax(two_copy_2_results_max)], 5)}$')
    # ax2.plot(x,two_copy3,linestyle = 'dotted', label = fr'$V = Polar(dV)$')
    ax2.axhline(0.8703, c='k', alpha=0.4, label='Full Stacking 3 Copy Result')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('3 Copy Case. Plotted for ' + r'$\alpha = [0.0001, 1]$')
    ax2.legend()
    plt.tight_layout()
    # plt.savefig('update_V_results/new_updating_V.pdf')
    plt.show()

    assert ()



    # evaluate_stacking_unitary(initial_V, dataset='mnist', training=False)
    # mpo_stacking('mnist')
    # classical_stacking('fashion_mnist')
    # assert ()

    # mps_stacking('mnist',1)
    # for i in range(3):
    #    U = delta_efficent_deterministic_quantum_stacking(i, v_col = True, dataset = 'mnist')
    #    evaluate_stacking_unitary(U, dataset = 'mnist', training = True)
    # assert()
    # U = test(1, dataset = 'fashion_mnist')
    # classical_stacking()
    # assert()
    # plot_confusion_matrix('mnist')
    # results = []
    # for i in range(1,10):
    # print('NUMBER OF COPIES: ',i)
    # U = specific_quantum_stacking(i,'fashion_mnist', True)
    # training_predictions, test_predictions = evaluate_stacking_unitary(U, True, dataset = 'fashion_mnist', training = False)
    # results.append([training_predictions, test_predictions])
    # np.save('partial_stacking_results_new', results)
    # stacking_on_confusion_matrix(0, dataset = 'mnist')
    # U = delta_efficent_deterministic_quantum_stacking(1, dataset = 'mnist')
    # U = sum_state_deterministic_quantum_stacking(2, dataset = 'mnist')
