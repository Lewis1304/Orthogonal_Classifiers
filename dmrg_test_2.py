"""
Some tests to verify that the stacking transfer matrix code should work
"""
import numpy as np
from scipy.linalg import eig
from scipy.stats import unitary_group
from ncon import ncon
np.set_printoptions(precision=4, linewidth=100000, suppress=True, threshold=np.inf)
import random
# from numpy.random import MT19937
# from numpy.random import RandomState, SeedSequence
# rs = RandomState(MT19937(SeedSequence(123456789)))
# np.random.set_state(MT19937(SeedSequence(123456789)))

np.random.seed(0)
small_idx = 4
big_index = small_idx ** 2
def with_arb_M():
    # Generate random Hermitian matrix M
    M = np.random.rand(big_index, big_index) \
        # + 1j * np.random.rand(big_index, big_index)
    # M = M + M.conj().T
    print(M)
    M_tens = M.reshape(small_idx, small_idx, small_idx, small_idx)  # reshape to (i, p, j, q)

    # Find eigenvalues and evecs + verify
    evals, evecs = eig(M)

    evec = evecs[:, 0]  # get the principle eigenvector
    eval = evals[0]  # get the principle eigenvalue

    # Generate tensor V + contract
    V_tens = evec.reshape(small_idx, small_idx)

    # Generate cost func V^dagger = M = V
    V_vec = evec

    # Verify that a random V does not produce a better result
    V_tens = V_tens / np.linalg.norm(V_vec)
    V_rand_tens = np.random.rand(*V_tens.shape) + 1j * np.random.rand(*V_tens.shape)
    V_rand_tens = V_rand_tens / np.linalg.norm(V_rand_tens)

    cost_tens = calculate_cost(V_tens, M_tens)
    cost_rand = calculate_cost(V_rand_tens, M_tens)

    print('Cost principle: ', cost_tens)
    print('Cost rand: ', cost_rand)
    #
    # for i, e in enumerate(evecs[1:]):
    #     V_evec = e.reshape(small_idx, small_idx)
    #     V_evec = V_evec / np.linalg.norm(V_evec)
    #     cost_evec = calculate_cost(V_evec, M_tens)
    #     print('Evec cost {}: {}'.format(i, cost_evec))


def construct_M(P, rho):
    '''
    P.shape = (i, l, j)
    rho.shape = (p, l, q)
​
    i -- P -- j
         |
         | l
         |
    p -- rho -- q
    '''

    return ncon([P, rho], ((-1, 1, -3), (-2, 1, -4)))


def with_rho_P():
    P = np.random.rand(small_idx, 1, small_idx) + 1j * np.random.rand(small_idx, 1, small_idx)
    P = P + np.transpose(P.conj(), (2, 1, 0))

    rho = np.random.rand(small_idx, 1, small_idx) + 1j * np.random.rand(small_idx, 1, small_idx)
    rho = rho + np.transpose(rho.conj(), (2, 1, 0))

    M_tens = construct_M(P, rho)
    M = M_tens.reshape(big_index, big_index)


    V_tens, evecs = calculate_V_from_M(M, ret_evecs=True)

    V_rand_tens = np.random.rand(*V_tens.shape) + 1j * np.random.rand(*V_tens.shape)
    V_rand_tens = V_rand_tens / np.linalg.norm(V_rand_tens)

    cost_tens = calculate_cost(V_tens, M_tens)
    cost_rand = calculate_cost(V_rand_tens, M_tens)
    print('Cost V*: ', cost_tens)
    print('Cost rand: ', cost_rand)

    # Check if any of the eigenvectors will give a larger cost output
    for i, e in enumerate(evecs[1:]):
        V_evec = e.reshape(small_idx, small_idx)
        V_evec = V_evec / np.linalg.norm(V_evec)
        cost_evec = calculate_cost(V_evec, M_tens)
        if np.linalg.norm(cost_evec) > np.linalg.norm(cost_tens):
            print('Evec cost {}: {}'.format(i, cost_evec))


def calculate_V_from_M(M, ret_evecs=False):
    '''
    Create V tensor from principle eigenvector of M
    '''
    _, evecs = eig(M)
    V_tens = evecs[:, 0].reshape(small_idx, small_idx)
    V_tens = V_tens / np.linalg.norm(V_tens)  # not sure if this normalisation is necessary
    if ret_evecs:
        return V_tens, evecs
    return V_tens


def calculate_cost(V_tens, M_tens):
    V_dagg_tens = V_tens.conj()

    return ncon([V_dagg_tens, M_tens, V_tens], ((1, 2), (1, 2, 3, 4), (3, 4)))


if __name__ == "__main__":
    # with_rho_P()
    with_arb_M()