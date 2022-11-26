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
    r = 1.0 #Having r = r + \epsilon allows plotting to look better
    theta, phi = theta_phi[0], theta_phi[1]
    return np.array([r*sin(theta)*cos(phi),r*sin(theta)*sin(phi),r*cos(theta)]).T

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

def create_sphere(r=0.96):
    #Having r < 1 allows plotting to look nicer
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    return x,y,z

def create_dataset(n_train = 500, sigma = 0.6, seed=42):
    #sigma=0.6   looks good
    np.random.seed(seed)
    sigma_0 = np.diag([sigma,sigma])
    sigma_1 = np.diag([sigma,sigma])

    #p(theta,phi) with theta,phi in radians
    #dist_1 = np.random.multivariate_normal([pi/4,-pi/2],sigma_0,n_train//2)
    dist_1 = np.random.multivariate_normal([0,0],sigma_0,n_train//2)
    #dist_2 = np.random.multivariate_normal([pi/2,-pi/4],sigma_1,n_train//2)
    dist_2 = np.random.multivariate_normal([pi,pi],sigma_1,n_train//2)

    #plt.scatter(*dist_1.T)
    #plt.scatter(*dist_2.T)
    #plt.show()
    return np.array(list(dist_1) + list(dist_2)).T, np.array([0]*len(dist_1) + [1]*len(dist_2))

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

    #ax.plot_surface(
    #    x, y, z,  rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r, alpha=0.8, linewidth=0)
    ax.plot_wireframe(
        x, y, z,  rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r, alpha=0.8, linewidth=0.5)

    ax.scatter(*spherical_coords(theta_phi).T, marker="o", s=25, edgecolor="k",c=yy)

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
            outer = np.outer(ket, ket.conj())

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
    print('Training accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_train, 1))
    print('Training accuracy U:', training_predictions)
    print()

"""
SVM classification
"""

def svms():
    from sklearn.svm import SVC, LinearSVC
    from sklearn.metrics import classification_report

    theta_phi, y = create_dataset()
    #SVM cannot handle complex numbers
    #Convert to spherical coords instead
    x = spherical_coords(theta_phi)

    x_initial, y_initial = create_states()
    print('Initial training accuracy: ', evaluate_classifier_top_k_accuracy(x_initial,y_initial,1))
    print()

    linear_classifier = SVC(kernel = 'linear', verbose = 0)
    linear_classifier.fit(x,y)
    linear_preds = linear_classifier.predict(x)
    print('Linear svm accuracy: ', classification_report(linear_preds, y, output_dict = True)['accuracy'])

    another_linear_classifier = LinearSVC(max_iter=10000)
    another_linear_classifier.fit(x,y)
    another_linear_preds = another_linear_classifier.predict(x)
    print('Another linear svm accuracy: ', classification_report(another_linear_preds, y, output_dict = True)['accuracy'])
    print()

    for i in ['scale', 'auto']:
        gaussian_linear_classifier = SVC(kernel = 'rbf', gamma = i, verbose = 0)
        gaussian_linear_classifier.fit(x,y)
        gaussian_linear_preds = gaussian_linear_classifier.predict(x)
        print(f'Gaussian svm {i} accuracy: ', classification_report(gaussian_linear_preds, y, output_dict = True)['accuracy'])
    print()

    for j in range(1,5):
        poly_linear_classifier = SVC(kernel = 'poly', degree = j, gamma = 'scale', verbose = 0)
        poly_linear_classifier.fit(x,y)
        poly_linear_preds = poly_linear_classifier.predict(x)
        print(f'Poly-{j} svm scale accuracy: ', classification_report(poly_linear_preds, y, output_dict = True)['accuracy'])
    print()

    for j in range(1,5):
        poly_linear_classifier = SVC(kernel = 'poly', degree = j, gamma = 'auto', verbose = 0)
        poly_linear_classifier.fit(x,y)
        poly_linear_preds = poly_linear_classifier.predict(x)
        print(f'Poly-{j} svm auto accuracy: ', classification_report(poly_linear_preds, y, output_dict = True)['accuracy'])
    print()

    for i in ['scale', 'auto']:
        sigmoid_linear_classifier = SVC(kernel = 'sigmoid', gamma = i, verbose = 0)
        sigmoid_linear_classifier.fit(x,y)
        sigmoid_linear_preds = sigmoid_linear_classifier.predict(x)
        print(f'Sigmoid svm {i} accuracy: ', classification_report(sigmoid_linear_preds, y, output_dict = True)['accuracy'])
    print()
    print('######################################')
    print()
    #assert()



if __name__ == '__main__':
    #plot()
    svms()
    #n_copies = 1

    for n in range(10):
        print(f"n_copies = {n}")
        U = initialise_V(n, verbose = 0)
        evaluate_stacking_unitary(U, verbose = 0)
