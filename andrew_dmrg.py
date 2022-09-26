from scipy import integrate

from scipy.io import loadmat

from ncon import ncon

import matplotlib.pyplot as plt

from scipy.linalg import expm

import scipy as sp

from pylab import *

import numpy as np

# import scipy.sparse.linalg.eigen.arpack as arp
from scipy.linalg import eig
import pylab as pl

# import tables

from scipy import integrate

""" Conventions:

This code tries to optimise V^dagger M V with M hermitian and V unitary  """


def init_V(D):
    """ Returns a random unitary """

    A = np.random.rand(D, D)
    V, x = np.linalg.qr(A)

    return V


def get_M(D):
    """ returns a random hermitian D^2xD^2 M """

    """reshaped as [D,D,D,D]= [l,l,r,r]"""

    A = np.random.rand(D * D, D * D)
    M = A + np.transpose(np.conj(A))
    M = M.reshape([D, D, D, D])

    return M


def get_testM(D):
    """Produce M as outer product of random vectors"""

    A = np.random.rand(D * D)
    M = np.outer(A, np.conj(A))
    M = M.reshape([D, D, D, D])

    return M


def get_VDMRGPlus(M, D):
    """ Returns V evaluated from the R fixed point of M   """

    M = M.reshape([D * D, D * D])
    e, R = eig(M)
    R = R.reshape([D, D])
    X, L, Y = np.linalg.svd(R)
    VDMRGPlus = ncon([X, Y], ([-1, 1], [1, -2]))

    return VDMRGPlus


def get_VDMRGPlus2(M, D):
    """ Returns V evaluated from the R fixed point of M   """

    M = M.reshape([D * D, D * D])
    u, l, v = np.linalg.svd(M)
    sql = sqrt(l)
    s = np.zeros((D * D), dtype=np.float64)
    s[0] = 1.0
    Rsum = ncon([s, v], ([1], [1, -1]))
    R = Rsum.reshape([D, D])
    X, L, Y = np.linalg.svd(R)
    VDMRGPlus2 = ncon([X, Y], ([-1, 1], [1, -2]))

    return VDMRGPlus2


def get_VDMRG(M, D):
    """ Returns V using DMRG   """

    V = init_V(D)
    Niterations = 100
    i = 1

    while i < Niterations:
        X, L, Y = np.linalg.svd(ncon([M, V], ([-1, -2, 1, 2], [1, 2])))
        V = ncon([X, Y], ([-1, 1], [1, -2]))
        C = get_C(M, V, D)
        i += 1

    return V


def get_C(M, V, D):
    """ Calulate Cost Function """

    C = ncon([np.conj(V), M, V], ([1, 2], [1, 2, 3, 4], [3, 4]))

    return C


def get_optC(M, D):
    """ get optimum C by random sampling over V"""

    Nsample = 10000
    optC = 0.0
    i = 1

    while i < Nsample:
        V = init_V(D)
        C = get_C(M, V, D)
        optC = max(optC, C)
        i += 1

    return optC


if __name__ == "__main__":
    # np.random.seed(1)

    D = 2

    """M = get_testM(D) """

    M = get_M(D)
    m = M.reshape([D * D, D * D])
    X, L, Y = np.linalg.svd(m)

    print(L)

    optC = get_optC(M, D)
    VDMRGPlus = get_VDMRGPlus2(M, D)
    CDMRGPlus = get_C(M, VDMRGPlus, D)
    VDMRG = get_VDMRG(M, D)
    CDMRG = get_C(M, VDMRG, D)

    print(VDMRGPlus)
    print(VDMRG)
    print('OPTC, ', optC)
    print(CDMRG)
    print(real(CDMRGPlus))
