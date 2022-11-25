import matplotlib.pyplot as plt
import numpy as np
import qutip

from sklearn.datasets import make_blobs
from matplotlib import cm, colors
from tqdm import tqdm
from svd_robust import svd
from scipy.linalg import polar


pi = np.pi
cos = np.cos
sin = np.sin
exp = np.exp

"""
Tools
"""
def spherical_coords(theta_phi):
    r = 1.01 #Having r = r + \epsilon allows plotting to look better
    theta, phi = theta_phi[0], theta_phi[1]
    return (r*sin(theta)*cos(phi),r*sin(theta)*sin(phi),r*cos(theta))

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
    if num_qubit == 1:
        return np.outer(rho, rho.conj())
    else:
        rho = qutip.Qobj(rho, dims=[[2] * num_qubit, [1]])
        return rho.ptrace(qubit_2_keep)

"""
Create funcs
"""

def create_sphere(r=1):
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    return x,y,z

def create_dataset(seed=2, sigma = 0.4):
    X, Y = make_blobs(100, n_features=2, centers=2, cluster_std=sigma, random_state=2)
    theta = X[:, 0]
    phi = X[:, 1]
    return (theta,phi), Y

def create_states():
    theta_phi , Y = create_dataset()
    theta, phi = theta_phi[0], theta_phi[1]
    return np.array([cos(theta/2), exp(1j*phi)*sin(theta/2)]).T, Y

"""
Plotting funcs
"""
def plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x,y,z = create_sphere()
    theta_phi, yy = create_dataset()

    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r, alpha=0.8, linewidth=0)

    ax.scatter(*spherical_coords(theta_phi), marker="o", s=25, edgecolor="k",c=yy)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()
    assert()

"""
Stacking funcs
"""

def initialise_V(n_copies, verbose=1):
    """
    args:
    n_copies: int: number of copies. STARTS FROM ZERO. e.g. n_copies=1 --> |\psi>^{\otimes m}
    dataset: string: mnist or fashion mnist
    verbose: int: Either 0 or 1. Shows progress bar for partial trace (1) or not (0)
    """

    if verbose == 0:
        verb = lambda x: x
    else:
        verb = tqdm

    """
    Load data
    """
    initial_label_qubits, y_train = create_states()

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
        weighted_outer_states = np.zeros((dim_lc, dim_lc),dtype = np.complex128)
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
        #Vl = np.array(U[:, : b // 2])
        Vl = np.array(U[:, :b//2] @ np.sqrt(np.diag(S)[:b//2,:b//2]))
        V.append(Vl)

    # Construct V from eigenvectors
    # Initially has shape [num_classes, 16**(n_copies+1) i.e. length of eigenvector, 16**(n_copies)]
    # num_classes padded to nearest multiple of 2. 10 ---> 16
    # V reshaped into ---> [num_classes_padded * 16**(n_copies), 16**(n_copies+1)]
    V = np.array(V)
    c, d, e = V.shape

    V = V.transpose(0, 2, 1).reshape(dim_l * e, d)
    #V = V.transpose(2, 0, 1).reshape(dim_l * e, d)

    #print("Performing Polar Decomposition!")
    U = polar(V)[0]
    #print("Finished Computing Stacking Unitary!")
    return U

"""
Evaluation funcs
"""

def evaluate_classifier_top_k_accuracy(predictions, y_test, k):
    top_k_predictions = [
        np.argpartition(image_prediction, -k)[-k:] for image_prediction in predictions
    ]
    results = np.mean([int(i in j) for i, j in zip(y_test, top_k_predictions)])
    return results

def evaluate_stacking_unitary(U, verbose=1):
    n_copies = int(np.log2(U.shape[0]) ) - 1

    """
    Dummy function for verbose
    """
    if verbose == 0:
        verb = lambda x: x
    else:
        verb = tqdm

    """
    Load Training Data
    """

    initial_label_qubits, y_train = create_states()


    """
    Rearrange test data to match new bitstring assignment
    """

    outer_ket_states = initial_label_qubits
    #.shape = n_train, dim_l**n_copies+1
    for k in range(n_copies):
        outer_ket_states = [np.kron(i, j) for i,j in zip(outer_ket_states, initial_label_qubits)]

    """
    Perform Overlaps
    """
    #We want qubit formation:
    #|l_0^0>|l_1^0>|l_0^1>|l_1^1> |l_2^0>|l_3^0>|l_2^1>|l_3^1>...
    #I.e. act only on first 2 qubits on all copies.
    #Since unitary is contructed on first 2 qubits of each copy.
    #So we want U @ SWAP @ |copy_preds>
    #print('Performing Overlaps!')
    preds_U = np.array([abs(U.dot(i)) for i in verb(outer_ket_states)])

    """
    Trace out other qubits/copies
    """

    #print('Performing Partial Trace!')
    preds_U = np.array([np.diag(partial_trace(i, [0])) for i in verb(preds_U)])

    training_predictions = evaluate_classifier_top_k_accuracy(preds_U, y_train, 1)
    print()
    print('Training accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_train, 1))
    print('Training accuracy U:', training_predictions)

"""
SVM classification
"""

if __name__ == '__main__':
    #plot()
    n_copies = 1

    for n_copies in range(10):

        U = initialise_V(n_copies, verbose = 0)
        evaluate_stacking_unitary(U, verbose = 0)
