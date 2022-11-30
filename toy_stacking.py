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
    r = 1.0
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

def create_dataset(n_total = 1000, n_test = 0.2, sigma = 0.2, seed=42):
    """
    sigma=0.5 looks good
    """
    np.random.seed(seed)
    sigma_0 = np.diag([sigma,pi*sigma])
    sigma_1 = np.diag([pi*sigma,sigma])

    """
    p(theta,phi) with theta,phi in radians
    """
    #dist_1 = np.random.multivariate_normal([0,0],sigma_0,n_total//2)
    dist_1 = np.random.multivariate_normal([0,0],sigma_0,n_total//2)
    #dist_2 = np.random.multivariate_normal([10,10],sigma_1,n_total//2)
    dist_2 = np.random.multivariate_normal([pi/2,0],sigma_1,n_total//2)

    #plt.scatter(*dist_1.T)
    #plt.scatter(*dist_2.T)
    #plt.show()

    n_train = int(n_total*(1-n_test))
    train_x = np.array(list(dist_1)[:n_train//2] + list(dist_2)[:n_train//2]).T
    train_y = np.array([0]*(train_x.shape[1]//2) + [1]*(train_x.shape[1]//2))

    test_x = np.array(list(dist_1)[n_train//2:] + list(dist_2)[n_train//2:]).T
    test_y = np.array([0]*(test_x.shape[1]//2) + [1]*(test_x.shape[1]//2))

    return (train_x, train_y), (test_x, test_y)

def create_states(theta_phi):
    theta, phi = theta_phi[0], theta_phi[1]
    return np.array([cos(theta/2), exp(1j*phi)*sin(theta/2)]).T

"""
Plotting funcs
"""

def plot_sphere():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x,y,z = create_sphere()
    (theta_phi_train, y_train), (theta_phi_test, y_test) = create_dataset()

    #ax.plot_surface(
    #    x, y, z,  rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r, alpha=0.8, linewidth=0)
    ax.plot_wireframe(
        x, y, z,  rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r, alpha=0.8, linewidth=0.5)

    ax.scatter(*spherical_coords(theta_phi_train).T, marker="o", s=25, edgecolor="k",c=y_train)
    #ax.scatter(*spherical_coords(theta_phi_test).T, marker="o", s=25, edgecolor="k",c=['red' if i == 0 else 'blue' for i in y_test])
    ax.scatter(*spherical_coords(theta_phi_test).T, marker="o", s=25, edgecolor="k",c=y_test)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.savefig('toy_stacking_results/sphere_n_total_500_n_test_02_sigma_06_seed_42.pdf')
    plt.show()
    assert()

def plot_results():

    #Results using the hyperparameters of the stacking results
    #initial_training_accuracy = 0.528375
    #initial_test_accuracy = 0.5255
    #best_svm_training_accuracy = 0.971875
    #best_svm_test_accuracy = 0.9725

    #initial_training_accuracy = 0.55
    #initial_test_accuracy = 0.54
    #best_svm_training_accuracy = 0.9625
    #best_svm_test_accuracy = 0.98

    initial_training_accuracy = 0.69
    initial_test_accuracy = 0.67
    best_svm_training_accuracy = 0.8675
    best_svm_test_accuracy = 0.8050

    #stacking_training_accuracy = np.load('toy_stacking_results/training_accuracy_n_total_10000_n_test_02_sigma_06_seed_42.npy')
    #stacking_test_accuracy = np.load('toy_stacking_results/test_accuracy_n_total_10000_n_test_02_sigma_06_seed_42.npy')

    #stacking_training_accuracy = np.load('toy_stacking_results/training_accuracy_n_total_500_n_test_02_sigma_06_seed_42.npy')
    #stacking_test_accuracy = np.load('toy_stacking_results/test_accuracy_n_total_500_n_test_02_sigma_06_seed_42.npy')

    stacking_training_accuracy = np.load('toy_stacking_results/training_accuracy_n_total_1000_n_test_02_sigma_06_seed_42_mu_00_pi20.npy')
    stacking_test_accuracy = np.load('toy_stacking_results/test_accuracy_n_total_1000_n_test_02_sigma_06_seed_42_mu_00_pi20.npy')


    x = range(1,len(stacking_training_accuracy)+1)
    plt.axhline(best_svm_training_accuracy, linestyle = "-",c = "tab:orange",alpha = 0.6)
    plt.axhline(best_svm_test_accuracy, linestyle = "--",c = "tab:orange",alpha = 0.6)
    plt.plot(x, stacking_training_accuracy, c = "tab:blue",linestyle = "-",marker = "o",linewidth = 1)
    plt.plot(x, stacking_test_accuracy, c = "tab:blue", linestyle = "--",marker = "o",linewidth = 1)

    plt.plot([],[],c = "grey",linestyle = "-", label = "Training")
    plt.plot([],[],c = "grey",linestyle = "--", label = "Test")
    plt.scatter([],[],c = "tab:blue",linewidth = 0, label = "Stacking")
    plt.scatter([],[],c = "tab:orange",linewidth = 0, label = "Best SVM")
    plt.title(f"Initial Training (Test) Accuracy: {round(initial_training_accuracy,4)} ({round(initial_test_accuracy,4)})")
    plt.xlabel('Number of Copies')
    plt.ylabel('Accuracy')
    plt.xlim([1,10])
    plt.legend()

    plt.savefig('toy_stacking_results/results_n_total_1000_n_test_02_sigma_06_seed_42_mu_00_pi20.pdf')
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
    (theta_phi_train, y_train), (theta_phi_test, y_test) = create_dataset()
    initial_label_qubits = create_states(theta_phi_train)

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
    for l in possible_labels:
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
    Load Data
    """

    (theta_phi_train, y_train), (theta_phi_test, y_test) = create_dataset()
    initial_label_train_qubits = create_states(theta_phi_train)
    initial_label_test_qubits = create_states(theta_phi_test)


    """
    Rearrange test data to match new bitstring assignment
    """

    outer_ket_train_states = initial_label_train_qubits
    outer_ket_test_states = initial_label_test_qubits
    #.shape = n_train, dim_l**n_copies+1
    for k in range(n_copies):
        outer_ket_train_states = [np.kron(i, j) for i,j in zip(outer_ket_train_states, initial_label_train_qubits)]
        outer_ket_test_states = [np.kron(i, j) for i,j in zip(outer_ket_test_states, initial_label_test_qubits)]

    """
    Perform Overlaps
    """
    #We want qubit formation:
    #|l_0^0>|l_1^0>|l_0^1>|l_1^1> |l_2^0>|l_3^0>|l_2^1>|l_3^1>...
    #I.e. act only on first 2 qubits on all copies.
    #Since unitary is contructed on first 2 qubits of each copy.
    #So we want U @ SWAP @ |copy_preds>
    #print('Performing Overlaps!')
    train_preds_U = np.array([abs(U.dot(i)) for i in verb(outer_ket_train_states)])
    test_preds_U = np.array([abs(U.dot(i)) for i in verb(outer_ket_test_states)])

    """
    Trace out other qubits/copies
    """

    #print('Performing Partial Trace!')
    train_preds_U = np.array([np.diag(partial_trace(i, [0])) for i in verb(train_preds_U)])
    test_preds_U = np.array([np.diag(partial_trace(i, [0])) for i in verb(test_preds_U)])

    training_predictions = evaluate_classifier_top_k_accuracy(train_preds_U, y_train, 1)
    test_predictions = evaluate_classifier_top_k_accuracy(test_preds_U, y_test, 1)

    #print('Training accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_train_qubits, y_train, 1))
    #print('Test accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_test_qubits, y_test, 1))
    print('Training accuracy U:', training_predictions)
    print('Test accuracy U:', test_predictions)
    print()

    return training_predictions, test_predictions

"""
SVM classification
"""

def svms():
    from sklearn.svm import SVC, LinearSVC
    from sklearn.metrics import classification_report

    (theta_phi_train, y_train), (theta_phi_test, y_test) = create_dataset()
    #SVM cannot handle complex numbers
    #Convert to spherical coords instead
    x_train = spherical_coords(theta_phi_train)
    x_test = spherical_coords(theta_phi_test)

    x_train_initial = create_states(theta_phi_train)
    x_test_initial = create_states(theta_phi_test)
    print('Initial training accuracy: ', evaluate_classifier_top_k_accuracy(x_train_initial,y_train,1))
    print('Initial test accuracy: ', evaluate_classifier_top_k_accuracy(x_test_initial,y_test,1))
    print()

    linear_classifier = SVC(kernel = 'linear', verbose = 0)
    linear_classifier.fit(x_train,y_train)
    linear_train_preds = linear_classifier.predict(x_train)
    linear_test_preds = linear_classifier.predict(x_test)
    print('Linear svm train accuracy: ', classification_report(linear_train_preds, y_train, output_dict = True)['accuracy'])
    print('Linear svm test accuracy: ', classification_report(linear_test_preds, y_test, output_dict = True)['accuracy'])
    print()

    another_linear_classifier = LinearSVC(max_iter=10000)
    another_linear_classifier.fit(x_train,y_train)
    another_linear_train_preds = another_linear_classifier.predict(x_train)
    another_linear_test_preds = another_linear_classifier.predict(x_test)
    print('Another linear svm train accuracy: ', classification_report(another_linear_train_preds, y_train, output_dict = True)['accuracy'])
    print('Another linear svm test accuracy: ', classification_report(another_linear_test_preds, y_test, output_dict = True)['accuracy'])
    print()

    for i in ['scale', 'auto']:
        gaussian_linear_classifier = SVC(kernel = 'rbf', gamma = i, verbose = 0)
        gaussian_linear_classifier.fit(x_train,y_train)
        gaussian_linear_train_preds = gaussian_linear_classifier.predict(x_train)
        gaussian_linear_test_preds = gaussian_linear_classifier.predict(x_test)
        print(f'Gaussian svm {i} train accuracy: ', classification_report(gaussian_linear_train_preds, y_train, output_dict = True)['accuracy'])
        print(f'Gaussian svm {i} test accuracy: ', classification_report(gaussian_linear_test_preds, y_test, output_dict = True)['accuracy'])
    print()

    for j in range(1,5):
        poly_linear_classifier = SVC(kernel = 'poly', degree = j, gamma = 'scale', verbose = 0)
        poly_linear_classifier.fit(x_train,y_train)
        poly_linear_train_preds = poly_linear_classifier.predict(x_train)
        poly_linear_test_preds = poly_linear_classifier.predict(x_test)
        print(f'Poly-{j} svm scale train accuracy: ', classification_report(poly_linear_train_preds, y_train, output_dict = True)['accuracy'])
        print(f'Poly-{j} svm scale test accuracy: ', classification_report(poly_linear_test_preds, y_test, output_dict = True)['accuracy'])
    print()

    for j in range(1,5):
        poly_linear_classifier = SVC(kernel = 'poly', degree = j, gamma = 'auto', verbose = 0)
        poly_linear_classifier.fit(x_train,y_train)
        poly_linear_train_preds = poly_linear_classifier.predict(x_train)
        poly_linear_test_preds = poly_linear_classifier.predict(x_test)
        print(f'Poly-{j} svm auto train accuracy: ', classification_report(poly_linear_train_preds, y_train, output_dict = True)['accuracy'])
        print(f'Poly-{j} svm auto test accuracy: ', classification_report(poly_linear_test_preds, y_test, output_dict = True)['accuracy'])
    print()

    for i in ['scale', 'auto']:
        sigmoid_linear_classifier = SVC(kernel = 'sigmoid', gamma = i, verbose = 0)
        sigmoid_linear_classifier.fit(x_train,y_train)
        sigmoid_linear_train_preds = sigmoid_linear_classifier.predict(x_train)
        sigmoid_linear_test_preds = sigmoid_linear_classifier.predict(x_test)
        print(f'Sigmoid svm {i} train accuracy: ', classification_report(sigmoid_linear_train_preds, y_train, output_dict = True)['accuracy'])
        print(f'Sigmoid svm {i} test accuracy: ', classification_report(sigmoid_linear_test_preds, y_test, output_dict = True)['accuracy'])
    print()
    print('######################################')
    print()

if __name__ == '__main__':
    #plot_sphere()
    #plot_results()

    svms()
    training_acc = []
    test_acc = []
    for n in range(10):
        print(f"n_copies = {n}")
        U = initialise_V(n, verbose = 0)
        train_result, test_result = evaluate_stacking_unitary(U, verbose = 0)

        training_acc.append(train_result)
        test_acc.append(test_result)

    #np.save('toy_stacking_results/training_accuracy_n_total_1000_n_test_02_sigma_06_seed_42_mu_00_pi20', training_acc)
    #np.save('toy_stacking_results/test_accuracy_n_total_1000_n_test_02_sigma_06_seed_42_mu_00_pi20', test_acc)
