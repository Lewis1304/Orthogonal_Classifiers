"""fMPO: Finite length matrix product operators"""
import numpy as np
from numpy import diag, expand_dims, zeros, concatenate

from scipy.linalg import polar
from tqdm import tqdm
import uuid

from xmps.svd_robust import svd
from xmps.ncon import ncon as nc
from xmps.tensor import rank
from xmps.fMPS import fMPS


def ncon(*args, **kwargs):
    return nc(
        *args, check_indices=False, **kwargs
    )  # make default ncon not check indices


class fMPO:
    """finite MPO:
    lists of numpy arrays (1d) of numpy arrays (2d). Finite"""

    def __init__(self, data=None, d=None, s=None, D=None):
        """__init__

        :param data: data for internal fMPO
        :param d: First local state space dimension
        :param s: Second local state space dimension
        :param D: Bond dimension: if not none, will truncate with right
        canonicalisation
        """
        self.id = uuid.uuid4().hex  # for memoization
        self.id_ = uuid.uuid4().hex  # for memoization
        self.id__ = uuid.uuid4().hex  # for memoization
        if data is not None:
            data = list(data)
            self.L = len(data)
            if d is not None:
                self.d = d
            else:
                self.d = data[-1].shape[0]
            if s is not None:
                self.s = s
            else:
                self.s = max([x.shape[1] for x in data]) #data[-1].shape[1]

            self.D = max([max(x.shape[2:]) for x in data])

            self.data = data

    def __call__(self, k):
        """__call__: 1-based indexing

        :param k: item to get
        """
        return self.data[k + 1]

    def __getitem__(self, k):
        """__getitem__: 0-based indexing

        :param k: item to get
        """
        return self.data[k]

    def __setitem__(self, key, value):
        """__setitem__"""
        self.data[key] = value

    def __str__(self):
        return "fMPO: L={}, d={}, s={}, D={}".format(self.L, self.d, self.s, self.D)

    def compress(self, D=None, orthogonalise=False):
        """compress: compress internal bonds of fMPO,
        potentially with a orthogonalisation

        :param D: bond dimension to truncate to during left & right sweep
        """

        # Check that right most site is hairy, and no others are.
        # Since procedure only works for truncated mpo
        for i, site in enumerate(self):
            if i < self.L - 2:
                assert site.shape[1] == 1
            else:
                assert site.shape[1] == 4

        if D is not None:
            self.D = min(D, self.D)

        def split(datum):
            """split: Do SVD and reshape A matrix
            :param M: matrix
            """
            d, s, i, j = datum.shape
            """
            reshapes d with i, s with j such that M.shape = (d*i,s*j)
            """
            M = datum.transpose(0, 2, 1, 3).reshape(d * i, s * j)
            u, S, v = svd(M, full_matrices=False)
            """
            u is reshaped from (d*i,k) to (d,i,k)
            v is reshaped from (k,s*j) to (k,s,j)
            k = min(d*i,s*j)
            """
            u = expand_dims(u.reshape(d, i, -1), 1)
            v = expand_dims(v.reshape(-1, s, j), 0).transpose(0, 2, 1, 3)
            return u, diag(S), v

        def split_hairs(datum):
            """split: Do SVD and reshape A matrix
                Regroup the legs such that (n-1) legs are w/ index i
                and j*d*2 on other leg
            :param M: matrix
            """
            d, s, i, j = datum.shape
            s_1 = int(s / 4)
            s_2 = 4
            """
            reshape: (d,s,i,j) to (d,s_1,s_2,i,j)
            transpose: (i,s_1,s_2,j,d)
            reshape: (i*s_1,s_2*j*d)
            """

            if s_1 >= 1:
                M = (
                    datum.reshape(d, s_1, s_2, i, j)
                    .transpose(3, 1, 2, 4, 0)
                    .reshape(i * s_1, s_2 * j * d)
                )
                # print('datum:', datum.shape)
                u, S, v = svd(M, full_matrices=False)
                # print(datum.shape)
                # print(M.shape)
                # print(u.shape)
                # print(S.shape)
                # print(v.shape)
                # print(v @ v.conj().T )
                """
                u is reshape from (i*s_1,k) to (i,s_1,,k)
                v is reshape from (k,s_2*j*d) to (k,s_2,j,d)
                tranpose to (d,s,k,j)
                k = min(i*s_1,s_2*j*d) = "-1"
                """
                u = expand_dims(u.reshape(i, s_1, -1), 0).transpose(0, 2, 1, 3)
                v = v.reshape(-1, s_2, j, d).transpose(3, 1, 0, 2)
                # print('u', u.shape)
                # print('v', v.shape)
                # print('S', S.shape)
                # print(datum.shape)
                # print(M.shape)
                # print(S.shape)
                # print()

                return u, diag(S), v
            else:
                M = datum.transpose(2, 1, 3, 0).reshape(i, s * j * d)
                u, S, v = svd(M, full_matrices=False)

                u = expand_dims(expand_dims(u.reshape(i, -1), 0), 0)
                v = v.reshape(-1, s, j, d).transpose(3, 1, 0, 2)

                return u, diag(S), v

        def truncate_MPO(A, S, V, D):
            """truncate: truncate A, S, V to D. Ditch all zero diagonals

            :param A: SVD U matrix reshaped
            :param S: SVD S matrix
            :param V: SVD V matrix
            :param D: Bond dimension to truncate to
            """

            if D is None:
                D = rank(S)

            A = A[:, :, :, :D]
            S = S[:D, :D]
            V = V[:, :, :D, :]

            return A, S, V

        for m in range(len(self.data)):

            # sort out canonicalisation
            if m + 1 < len(self.data):
                A, S, V = split(self[m])
                self[m], S, V = truncate_MPO(A, S, V, D)
                """
                ncon: Contract S@V j leg with self[m+1] i leg.
                transpose: take (d_1,s_1,i_1,d_2,s_2,j_2) to (d_1,d_2,s_1,s_2,i_1,j_2)
                reshape: group together d and s legs such that .shape = (d_1d_2,s_1s_2,i_1,j_2)
                """
                d_1, s_1, i_1, j_1 = (S @ V).shape
                d_2, s_2, i_2, j_2 = self[m + 1].shape

                self[m + 1] = (
                    ncon((S @ V, self[m + 1]), [[-1, -2, -3, 4], [-4, -5, 4, -6]])
                    .transpose(0, 3, 1, 4, 2, 5)
                    .reshape(d_1 * d_2, s_1 * s_2, i_1, j_2)
                )
            else:
                d, s, i, j = self[m].shape
                if orthogonalise:
                    self[m] = (
                        polar(self[m].transpose(0, 2, 1, 3).reshape(d * i, s * j))[0]
                        .reshape(d, i, s, j)
                        .transpose(0, 2, 1, 3)
                    )
                self.S = S
                self.V = V
        #d, s, i, j = (self.V).shape

        for m in range(len(self.data))[::-1]:

            A, S, V = split_hairs(self[m])
            U, S, self[m] = truncate_MPO(A, S, V, D)
            if m > 0:

                #ncon: Contract self[m-1] j leg with U@S i leg
                #transpose: take (d_1,s_1,i_1,d_2,s_2,j_2) to (d_1,d_2,s_1,s_2,i_1,j_2)
                #reshape: group together d and s legs such that .shape = (d_1d_2,s_1s_2,i_1,j_2)

                d_1, s_1, i_1, j_1 = self[m - 1].shape
                d_2, s_2, i_2, j_2 = (U @ S).shape

                self[m - 1] = (
                    ncon((self[m - 1], U @ S), [[-1, -2, -3, 4], [-5, -6, 4, -7]])
                    .transpose(0, 3, 1, 4, 2, 5)
                    .reshape(d_1 * d_2, s_1 * s_2, i_1, j_2)
                )
                # print('After:', self[m-1].shape)
                # print()

        # test2 = self
        # print([i.shape for i in test2])
        return self

    def compress_one_site(self, D=None, orthogonalise=False, sweep_back = True):
        """compress: compress internal bonds of fMPO with one hairy site,
        potentially with a orthogonalisation

        :param D: bond dimension to truncate to during left sweep
        """

        def split(datum):
            """split: Do SVD and reshape A matrix
            :param M: matrix
            """
            d, s, i, j = datum.shape
            """
            reshapes d with i, s with j such that M.shape = (d*i,s*j)
            """
            M = datum.transpose(0, 2, 1, 3).reshape(d * i, s * j)
            u, S, v = svd(M, full_matrices=False)

            """
            u is reshaped from (d*i,k) to (d,i,k)
            v is reshaped from (k,s*j) to (k,s,j)
            k = min(d*i,s*j)
            """
            u = expand_dims(u.reshape(d, i, -1), 1)
            v = expand_dims(v.reshape(-1, s, j), 0).transpose(0, 2, 1, 3)
            return u, diag(S), v

        def split_back(datum):
            """split: Do SVD and reshape A matrix
            :param M: matrix
            """
            d, s, i, j = datum.shape
            """
            reshapes s with d and j such that M.shape = (i,s*j*d)
            """
            M = datum.transpose(2, 1, 3, 0).reshape(i, s * j * d)
            u, S, v = svd(M, full_matrices=False)

            u = expand_dims(expand_dims(u.reshape(i, -1), 0), 0)
            v = v.reshape(-1, s, j, d).transpose(3, 1, 0, 2)

            return u, diag(S), v

        def truncate_MPO(A, S, V, D):
            """truncate: truncate A, S, V to D. Ditch all zero diagonals

            :param A: SVD U matrix reshaped
            :param S: SVD S matrix
            :param V: SVD V matrix
            :param D: Bond dimension to truncate to
            """

            if D is None:
                D = rank(S)

            A = A[:, :, :, :D]
            S = S[:D, :D]
            V = V[:, :, :D, :]

            return A, S, V

        for m in range(len(self.data)):

            # sort out canonicalisation

            if m + 1 < len(self.data):
                A, S, V = split(self.data[m])
                self.data[m], S, V = truncate_MPO(A, S, V, D)
                """
                ncon: Contract S@V j leg with self[m+1] i leg.
                transpose: take (d_1,s_1,i_1,d_2,s_2,j_2) to (d_1,d_2,s_1,s_2,i_1,j_2)
                reshape: group together d and s legs such that .shape = (d_1d_2,s_1s_2,i_1,j_2)
                """
                d_1, s_1, i_1, j_1 = (S @ V).shape
                d_2, s_2, i_2, j_2 = self.data[m + 1].shape

                self.data[m + 1] = (
                    ncon((S @ V, self.data[m + 1]), [[-1, -2, -3, 4], [-4, -5, 4, -6]])
                    .transpose(0, 3, 1, 4, 2, 5)
                    .reshape(d_1 * d_2, s_1 * s_2, i_1, j_2)
                )
            else:
                if orthogonalise:
                    d, s, i, j = self.data[m].shape
                    U, P = polar(self.data[m].transpose(0, 2, 1, 3).reshape(d * i, s * j))
                    self.data[m] = (U
                        .reshape(d, i, s, j)
                        .transpose(0, 2, 1, 3)
                    )

        #Sweep back normalises the operator. This is IMPORTANT otherwise batches
        #Will have different weightings.
        from variational_mpo_classifiers import data_to_QTN
        qtn = data_to_QTN(self.data)
        self.data[-1] /= np.sqrt(qtn.H @ qtn)

        return fMPO(self.data)

    def compress_centre_one_site(self, D=None, orthogonalise=False, sweep_back = True):
        """compress: compress internal bonds of fMPO with one hairy site,
        potentially with a orthogonalisation

        :param D: bond dimension to truncate to during left sweep
        """

        def split_left_to_right(datum):
            """split: Do SVD and reshape A matrix
            :param M: matrix
            """
            d, s, i, j = datum.shape
            """
            reshapes d with i, s with j such that M.shape = (d*i,s*j)
            """
            M = datum.transpose(0, 2, 1, 3).reshape(d * i, s * j)
            u, S, v = svd(M, full_matrices=False)

            """
            u is reshaped from (d*i,k) to (d,i,k)
            v is reshaped from (k,s*j) to (k,s,j)
            k = min(d*i,s*j)
            """
            u = expand_dims(u.reshape(d, i, -1), 1)
            v = expand_dims(v.reshape(-1, s, j), 0).transpose(0, 2, 1, 3)
            return u, diag(S), v

        def split_right_to_left(datum):
            """split: Do SVD and reshape A matrix
            :param M: matrix
            """
            d, s, i, j = datum.shape
            """
            reshapes d with j, s with i such that M.shape = (i*s,j*d)
            """
            M = datum.transpose(2, 1, 3, 0).reshape(i * s, j * d)
            u, S, v = svd(M, full_matrices=False)

            u = expand_dims(expand_dims(u.reshape(i, -1), 0), 0)
            v = v.reshape(-1, s, j, d).transpose(3, 1, 0, 2)

            return u, diag(S), v

        def truncate_MPO(A, S, V, D):
            """truncate: truncate A, S, V to D. Ditch all zero diagonals

            :param A: SVD U matrix reshaped
            :param S: SVD S matrix
            :param V: SVD V matrix
            :param D: Bond dimension to truncate to
            """

            if D is None:
                D = rank(S)

            A = A[:, :, :, :D]
            S = S[:D, :D]
            V = V[:, :, :D, :]

            return A, S, V


        centre_site = np.argmax([i.shape[1] for i in self.data])
        #left to middle sweep
        for m in range(centre_site):

            A, S, V = split_left_to_right(self.data[m])
            self.data[m], S, V = truncate_MPO(A, S, V, D)
            """
            ncon: Contract S@V j leg with self[m+1] i leg.
            transpose: take (d_1,s_1,i_1,d_2,s_2,j_2) to (d_1,d_2,s_1,s_2,i_1,j_2)
            reshape: group together d and s legs such that .shape = (d_1d_2,s_1s_2,i_1,j_2)
            """
            d_1, s_1, i_1, j_1 = (S @ V).shape
            d_2, s_2, i_2, j_2 = self.data[m + 1].shape

            self.data[m + 1] = (
                ncon((S @ V, self.data[m + 1]), [[-1, -2, -3, 4], [-4, -5, 4, -6]])
                .transpose(0, 3, 1, 4, 2, 5)
                .reshape(d_1 * d_2, s_1 * s_2, i_1, j_2)
            )

        #right to middle sweep
        for m in range(centre_site + 1, len(self.data))[::-1]:

            A, S, V = split_right_to_left(self[m])
            U, S, self[m] = truncate_MPO(A, S, V, D)

            #ncon: Contract self[m-1] j leg with U@S i leg
            #transpose: take (d_1,s_1,i_1,d_2,s_2,j_2) to (d_1,d_2,s_1,s_2,i_1,j_2)
            #reshape: group together d and s legs such that .shape = (d_1d_2,s_1s_2,i_1,j_2)

            d_1, s_1, i_1, j_1 = self.data[m - 1].shape
            d_2, s_2, i_2, j_2 = (U @ S).shape

            self.data[m - 1] = (
                ncon((self.data[m - 1], U @ S), [[-1, -2, -3, 4], [-5, -6, 4, -7]])
                .transpose(0, 3, 1, 4, 2, 5)
                .reshape(d_1 * d_2, s_1 * s_2, i_1, j_2)
            )

        if orthogonalise:
            d, s, i, j = self.data[centre_site].shape
            U, P = polar(self.data[centre_site].transpose(0, 2, 1, 3).reshape(d, i * s * j))
            self.data[centre_site] = (U
                .reshape(d, i, s, j)
                .transpose(0, 2, 1, 3)
            )
        else:
            #Sweep back normalises the operator. This is IMPORTANT otherwise batches
            #Will have different weightings.
            from variational_mpo_classifiers import data_to_QTN
            qtn = data_to_QTN(self.data)
            self.data[centre_site] /= np.sqrt(qtn.H @ qtn)

        return fMPO(self.data)

    def compress_to_centre_one_site(self, D=None, orthogonalise=False, sweep_back = True):
        """compress: compress internal bonds of fMPO with one hairy site,
        potentially with a orthogonalisation

        :param D: bond dimension to truncate to during left sweep
        """

        def split_left_to_right(datum):
            """split: Do SVD and reshape A matrix
            :param M: matrix
            """
            d, s, i, j = datum.shape
            """
            reshapes d with i, s with j such that M.shape = (d*i,s*j)
            """
            M = datum.transpose(0, 2, 1, 3).reshape(d * i, s * j)
            u, S, v = svd(M, full_matrices=False)

            """
            u is reshaped from (d*i,k) to (d,i,k)
            v is reshaped from (k,s*j) to (k,s,j)
            k = min(d*i,s*j)
            """
            u = expand_dims(u.reshape(d, i, -1), 1)
            v = expand_dims(v.reshape(-1, s, j), 0).transpose(0, 2, 1, 3)
            return u, diag(S), v

        def split_right_to_left(datum):
            """split: Do SVD and reshape A matrix
            :param M: matrix
            """
            d, s, i, j = datum.shape
            """
            reshapes d with j, s with i such that M.shape = (i*s,j*d)
            """
            M = datum.transpose(2, 1, 3, 0).reshape(i * s, j * d)
            u, S, v = svd(M, full_matrices=False)

            """
            u is reshaped from (i*s,k) to (i,s,k)
            v is reshaped from (k,j*d) to (k,j,d)
            """
            u = expand_dims(u.reshape(i,s,-1),0).transpose(0,2,1,3)
            v = expand_dims(v.reshape(-1,j,d),0).transpose(3,0,1,2)

            return u, diag(S), v

        def truncate_MPO(A, S, V, D):
            """truncate: truncate A, S, V to D. Ditch all zero diagonals

            :param A: SVD U matrix reshaped
            :param S: SVD S matrix
            :param V: SVD V matrix
            :param D: Bond dimension to truncate to
            """

            if D is None:
                D = rank(S)

            A = A[:, :, :, :D]
            S = S[:D, :D]
            V = V[:, :, :D, :]

            return A, S, V


        centre_site = 5

        #right to middle sweep
        for m in range(centre_site + 1, len(self.data))[::-1]:

            A, S, V = split_right_to_left(self[m])
            U, S, self[m] = truncate_MPO(A, S, V, D)

            #ncon: Contract self[m-1] j leg with U@S i leg
            #transpose: take (d_1,s_1,i_1,d_2,s_2,j_2) to (d_1,d_2,s_1,s_2,i_1,j_2)
            #reshape: group together d and s legs such that .shape = (d_1d_2,s_1s_2,i_1,j_2)

            d_1, s_1, i_1, j_1 = self.data[m - 1].shape
            d_2, s_2, i_2, j_2 = (U @ S).shape

            self.data[m - 1] = (
                ncon((self.data[m - 1], U @ S), [[-1, -2, -3, 4], [-5, -6, 4, -7]])
                .transpose(0, 3, 1, 4, 2, 5)
                .reshape(d_1 * d_2, s_1 * s_2, i_1, j_2)
            )

        if orthogonalise:
            d, s, i, j = self.data[centre_site].shape
            U, P = polar(self.data[centre_site].transpose(0, 2, 1, 3).reshape(d * i, s * j))
            self.data[centre_site] = (U
                .reshape(d, i, s, j)
                .transpose(0, 2, 1, 3)
            )



        #Sweep back normalises the operator. This is IMPORTANT otherwise batches
        #Will have different weightings.
        from variational_mpo_classifiers import data_to_QTN
        qtn = data_to_QTN(self.data)
        self.data[centre_site] /= np.sqrt(qtn.H @ qtn)

        return fMPO(self.data)

    def compress_right_to_left(self, D=None, orthogonalise=False, sweep_back = True):
        """compress: compress internal bonds of fMPO with one hairy site,
        potentially with a orthogonalisation

        :param D: bond dimension to truncate to during left sweep
        """

        def split_left_to_right(datum):
            """split: Do SVD and reshape A matrix
            :param M: matrix
            """
            d, s, i, j = datum.shape
            """
            reshapes d with i, s with j such that M.shape = (d*i,s*j)
            """
            M = datum.transpose(0, 2, 1, 3).reshape(d * i, s * j)
            u, S, v = svd(M, full_matrices=False)

            """
            u is reshaped from (d*i,k) to (d,i,k)
            v is reshaped from (k,s*j) to (k,s,j)
            k = min(d*i,s*j)
            """
            u = expand_dims(u.reshape(d, i, -1), 1)
            v = expand_dims(v.reshape(-1, s, j), 0).transpose(0, 2, 1, 3)
            return u, diag(S), v

        def split_right_to_left(datum):
            """split: Do SVD and reshape A matrix
            :param M: matrix
            """
            d, s, i, j = datum.shape
            """
            reshapes d with j, s with i such that M.shape = (i*s,j*d)
            """
            M = datum.transpose(2, 1, 3, 0).reshape(i * s, j * d)
            u, S, v = svd(M, full_matrices=False)

            """
            u is reshaped from (i*s,k) to (i,s,k)
            v is reshaped from (k,j*d) to (k,j,d)
            """
            u = expand_dims(u.reshape(i,s,-1),0).transpose(0,2,1,3)
            v = expand_dims(v.reshape(-1,j,d),0).transpose(3,0,1,2)

            return u, diag(S), v

        def truncate_MPO(A, S, V, D):
            """truncate: truncate A, S, V to D. Ditch all zero diagonals

            :param A: SVD U matrix reshaped
            :param S: SVD S matrix
            :param V: SVD V matrix
            :param D: Bond dimension to truncate to
            """

            if D is None:
                D = rank(S)

            A = A[:, :, :, :D]
            S = S[:D, :D]
            V = V[:, :, :D, :]

            return A, S, V


        centre_site = 0

        #right to middle sweep
        for m in range(0, len(self.data))[::-1]:
            if m > 0:
                A, S, V = split_right_to_left(self[m])
                U, S, self[m] = truncate_MPO(A, S, V, D)

                #ncon: Contract self[m-1] j leg with U@S i leg
                #transpose: take (d_1,s_1,i_1,d_2,s_2,j_2) to (d_1,d_2,s_1,s_2,i_1,j_2)
                #reshape: group together d and s legs such that .shape = (d_1d_2,s_1s_2,i_1,j_2)

                d_1, s_1, i_1, j_1 = self.data[m - 1].shape
                d_2, s_2, i_2, j_2 = (U @ S).shape

                self.data[m - 1] = (
                    ncon((self.data[m - 1], U @ S), [[-1, -2, -3, 4], [-5, -6, 4, -7]])
                    .transpose(0, 3, 1, 4, 2, 5)
                    .reshape(d_1 * d_2, s_1 * s_2, i_1, j_2)
                )

        if orthogonalise:
            d, s, i, j = self.data[centre_site].shape
            U, P = polar(self.data[centre_site].transpose(0, 2, 1, 3).reshape(d * i, s * j))
            self.data[centre_site] = (U
                .reshape(d, i, s, j)
                .transpose(0, 2, 1, 3)
            )



        #Sweep back normalises the operator. This is IMPORTANT otherwise batches
        #Will have different weightings.
        from variational_mpo_classifiers import data_to_QTN
        qtn = data_to_QTN(self.data)
        self.data[centre_site] /= np.sqrt(qtn.H @ qtn)

        return fMPO(self.data)


    def add(self, other):
        """add: proper mpo addition here"""
        new_data = [
            1j
            * zeros(
                (
                    self[i].shape[0],
                    self[i].shape[1],
                    self[i].shape[2] + other[i].shape[2],
                    self[i].shape[3] + other[i].shape[3],
                )
            )
            for i in range(self.L)
        ]
        for i in range(self.L):
            if i == 0:
                new_data[i] = concatenate([self[i], other[i]], 3)
            elif i == self.L - 1:
                new_data[i] = concatenate([self[i], other[i]], 2)
            else:
                new_data[i][:, :, : self[i].shape[2], : self[i].shape[3]] = self[i]
                new_data[i][:, :, self[i].shape[2] :, self[i].shape[3] :] = other[i]

        return fMPO(new_data)


if __name__ == "__main__":
    pass
