import numpy as np
import quimb
from scipy.linalg import polar
from sympy import Matrix
from xmps.svd_robust import svd
import quimb.tensor as qtn
np.set_printoptions(precision=4, linewidth=100000, suppress=True, threshold=np.inf)
from scipy.linalg import eig

small_idx = 4
big_index = small_idx ** 2

def build_M():
    # vec = np.random.rand(big_index, 1)
    # M = np.outer(vec, vec.conj().T)
    # M = M.reshape(small_idx, small_idx, small_idx, small_idx)
    M = np.random.rand(big_index, big_index) \
        # + 1j * np.random.rand(big_index, big_index)
    # M = M + M.conj().T
    # print(M)
    M_tens = M.reshape(small_idx, small_idx, small_idx, small_idx)  # reshape to (i, p, j, q)

    M_tens = qtn.Tensor(M_tens, inds=('i', 'p', 'j', 'q'))
    return M_tens

def build_P():
    vec = np.random.rand(small_idx, 1)
    P = np.outer(vec, vec.conj().T)
    P = np.expand_dims(P, axis=2)
    P_tens = qtn.Tensor(data=P, inds=('i', 'j', 'l'))
    return P_tens


def build_rho():
    vec = np.random.rand(small_idx, 1)
    rho = np.outer(vec, vec.conj().T)
    rho = np.expand_dims(rho, axis=2)

    rho_tens = qtn.Tensor(data=rho, inds=('p', 'q', 'l'))

    return rho_tens


def build_V():
    rand_mat = np.random.rand(small_idx, small_idx)
    V = polar(rand_mat)[0]
    V_tens = qtn.Tensor(data=V, inds=('u', 'v'))

    return V_tens


def tensorise_V(V_new, V_old_tens):
    return qtn.Tensor(data=V_new, inds=V_old_tens.inds)


def get_principle_eig_M(M):
    eig_vals, eig_vecs = eig(M)
    # print(eig_vals.real, '\n\n', eig_vecs.real, '\n\n')
    # idx = eig_vals.argsort()[::-1]
    # eig_vals = eig_vals[idx]
    # eig_vecs = eig_vecs[:, idx]
    # print(eig_vals.real, '\n\n', eig_vecs.real, '\n\n')
    # U_svd , S_svd , V_svd = svd(M)
    # eig_val_main = S_svd[0]
    # eig_vec_main = V_svd[0, :]
    # print('Real e_val', eig_vals[0], '\n', np.expand_dims(eig_vecs[:, 0], 1))
    # print('multiplied', eig_vals[0] * np.expand_dims(eig_vecs[:, 0], 1))
    #
    # print('HERE', M @ np.expand_dims(eig_vecs[:, 0], 1))
    return eig_vals[0], eig_vecs[:, 0]
    # print(S_svd , '\n\n\n', V_svd )
    # print('\n', eig_vec_main)

    # return eig_val_main, eig_vec_main

def create_M(P, rho):
    M_tens = qtn.TensorNetwork([P, rho])
    contracted_M = M_tens ^ ...

    # print(contracted_M)

    # M = contracted_M.data.reshape(big_index, big_index)
    M = contracted_M.data
    return M


def compute_cost(V_tens, contracted_M):
    # matrix = V.conj().T @ M
    V_dag_tens = qtn.Tensor(V_tens.data.conj(), inds=('r', 's'))
    # V_dag_tens = V_tens.H

    # M_tens = qtn.TensorNetwork([P, rho])
    # contracted_M = M_tens ^ ...
    # print(contracted_M)
    # print('HERE', V_tens, V_dag_tens)

    relabel_V = {V_tens.inds[0]: contracted_M.inds[2], V_tens.inds[1]: contracted_M.inds[3]}
    relabel_Vdag = {V_dag_tens.inds[0]: contracted_M.inds[0], V_dag_tens.inds[1]: contracted_M.inds[1]}
    V_tens.reindex(relabel_V, inplace=True)
    V_dag_tens.reindex(relabel_Vdag, inplace=True)

    # print(V_tens, V_dag_tens)
    cost_tensor = qtn.TensorNetwork([V_tens, V_dag_tens, M_tens])
    # print('\n', cost_tensor)
    cost = cost_tensor ^ ...
    # print('\n', cost)
    return cost


if __name__ == "__main__":
    # M = build_M()
    # build_V()
    # P = build_P()
    # rho = build_rho()
    # print(P, '\n', rho)

    # V_old_tens = build_V()
    for i in range(100):
        M_tens = build_M()

        V_rand_tens = np.random.rand(small_idx, small_idx) + 1j * np.random.rand(small_idx, small_idx)
        V_rand_tens = V_rand_tens / np.linalg.norm(V_rand_tens)
        V_rand_tens = qtn.Tensor(data=V_rand_tens, inds=('u', 'v'))

        # print('OLD COST')
        old_cost = compute_cost(V_rand_tens, M_tens)
        M = M_tens.data.reshape(big_index, big_index)
        # M = create_M(P, rho)
        # trans_M = M.transpose(2, 3, 0, 1)
        # trans_M = trans_M.reshape(big_index, big_index).conj()
        # print(trans_M - M.reshape(big_index, big_index))
        eig_val_main, eig_vec_main = get_principle_eig_M(M)
        # print(eig_val_main, len(eig_vec_main))
        eig_vec_main = eig_vec_main.reshape(small_idx, small_idx)
        # V_new = polar(eig_vec_main)[0]
        V_new = eig_vec_main

        # print('NEW COST')
        V_new_tens = tensorise_V(V_new, V_rand_tens)
        new_cost = compute_cost(V_new_tens, M_tens)
        if new_cost.real > old_cost.real:
            print(True)
