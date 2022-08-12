from xmps.svd_robust import svd
from scipy.linalg import polar
import numpy as np
from plot_results import produce_psuedo_sum_states, load_brute_force_permutations
from scipy import sparse
import qutip
import matplotlib.pyplot as plt
from functools import reduce

from tqdm import tqdm

from tools import load_qtn_classifier, data_to_QTN, arrange_data, load_data, bitstring_data_to_QTN
from fMPO import fMPO
from experiments import adding_centre_batches, label_last_site_to_centre, create_experiment_bitstrings, centred_bitstring_to_qtn, prepare_centred_batched_classifier, adding_centre_batches
from deterministic_mpo_classifier import unitary_qtn
from variational_mpo_classifiers import evaluate_classifier_top_k_accuracy, classifier_predictions, mps_encoding, create_hairy_bitstrings_data


import quimb.tensor as qtn
from quimb.tensor.tensor_core import rand_uuid
from oset import oset

from scipy.stats import unitary_group


def classical_stacking(dataset):
    """
    #Ancillae start in state |00...>
    #n=0
    #ancillae_qubits = np.eye(2**n)[0]
    #Tensor product ancillae with predicition qubits
    #Amount of ancillae equal to amount of predicition qubits
    #training_predictions = np.array([np.kron(ancillae_qubits, (mps_image.H @ classifier).squeeze().data) for mps_image in tqdm(mps_images)])

    #Create predictions
    training_predictions = np.array([abs((mps_image.H @ classifier).squeeze().data) for mps_image in mps_images])
    np.save('ortho_big_training_predictions_D_32',training_predictions)
    training_predictions = np.load('ortho_big_training_predictions_D_32.npy')
    labels = np.load('big_labels.npy')
    training_acc = evaluate_classifier_top_k_accuracy(training_predictions, labels, 1)
    print('Training Accuracy:', training_acc)
    """

    """
    x_train, y_train, x_test, y_test = load_data(
        100,10000, shuffle=False, equal_numbers=True
    )
    D_test = 32
    mps_test = mps_encoding(x_test, D_test)
    test_predictions = np.array(classifier_predictions(classifier, mps_test, bitstrings))
    accuracy = evaluate_classifier_top_k_accuracy(test_predictions, y_test, 1)
    print('Test Accuracy:', accuracy)
    """
    """
    inputs = tf.keras.Input(shape=(28,28,1))
    x = tf.keras.layers.AveragePooling2D(pool_size = (2,2))(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras  .layers.Dense(10, activation = 'relu')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    """
    #training_predictions = np.load('Classifiers/fashion_mnist/initial_training_predictions_ortho_mpo_classifier.npy')
    #y_train = np.load('Classifiers/fashion_mnist/big_dataset_training_labels.npy')
    import tensorflow as tf

    training_predictions = np.load('data/' + dataset + '/new_ortho_d_final_vs_training_predictions.npy')[15]
    training_predictions = np.array([np.kron(i,i) for i in training_predictions])
    y_train = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_labels.npy')
    y_train = np.array([np.kron(i,i) for i in y_train])

    inputs = tf.keras.Input(shape=(training_predictions.shape[1],))
    #inputs = tf.keras.Input(shape=(qtn_prediction_and_ancillae_qubits.shape[1],))
    #x = tf.keras.layers.Dense(1000, activation = 'sigmoid')(inputs)
    #x = tf.keras.layers.Dense(1000, activation = 'sigmoid')(x)
    outputs = tf.keras.layers.Dense(256, activation = None)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()


    model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    #earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)

    history = model.fit(
        training_predictions,
        y_train,
        epochs=300,
        batch_size = 32,
        verbose = 1
    )
    #model.save('models/ortho_big_dataset_D_32')
    test_preds = np.load('data/' + dataset + '/new_ortho_d_final_vs_test_predictions.npy')[15]
    y_test = np.load('data/' + dataset + '/ortho_d_final_vs_test_predictions_labels.npy')

    trained_test_predictions = model.predict(test_preds)
    #np.save('final_label_qubit_states_4',trained_training_predictions)

    #np.save('trained_predicitions_1000_classifier_32_1000_train_images', trained_training_predictions)
    accuracy = evaluate_classifier_top_k_accuracy(trained_test_predictions, y_test, 1)
    print(accuracy)

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
    if num_qubit == 4:
        return np.outer(rho,rho.conj())
    else:
        rho = qutip.Qobj(rho, dims = [[2] * num_qubit ,[1]])
        return rho.ptrace(qubit_2_keep)

def evaluate_stacking_unitary(U, partial = False, dataset = 'fashion_mnist', training = False):
    """
    Evaluate Performance
    """
    n_copies = int(np.log2(U.shape[0])//4)-1

    if training:
        """
        Load Training Data
        """

        initial_label_qubits = np.load('data/' + dataset + '/new_ortho_d_final_vs_training_predictions.npy')[15].astype(np.float32)
        y_train = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_labels.npy').astype(np.float32)

        """
        Rearrange test data to match new bitstring assignment
        """
        if partial:
            possible_labels = [5,7,8,9,4,0,6,1,2,3]
            assignment = possible_labels + list(range(10,16))
            reassigned_preds = np.array([i[assignment] for i in initial_label_qubits])
            reassigned_preds = np.array([i / np.sqrt(i.conj().T @ i) for i in reassigned_preds])
        else:
            reassigned_preds = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])

        outer_ket_states = reassigned_preds
        #.shape = n_train, dim_l**n_copies+1
        for k in range(n_copies):
            outer_ket_states = [np.kron(i, j) for i,j in zip(outer_ket_states, reassigned_preds)]

        """
        Perform Overlaps
        """
        #We want qubit formation:
        #|l_0^0>|l_1^0>|l_0^1>|l_1^1> |l_2^0>|l_3^0>|l_2^1>|l_3^1>...
        #I.e. act only on first 2 qubits on all copies.
        #Since unitary is contructed on first 2 qubits of each copy.
        #So we want U @ SWAP @ |copy_preds>
        print('Performing Overlaps!')
        preds_U = np.array([abs(U.dot(i)) for i in tqdm(outer_ket_states)])

        """
        Trace out other qubits/copies
        """

        print('Performing Partial Trace!')
        preds_U = np.array([np.diag(partial_trace(i, [0,1,2,3])) for i in tqdm(preds_U)])

        #Rearrange to 0,1,2,3,.. formation. This is req. for evaluate_classifier
        if partial:
            preds_U = np.array([i[assignment] for i in preds_U])
        training_predictions = evaluate_classifier_top_k_accuracy(preds_U, y_train, 1)
        print()
        print('Training accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_train, 1))
        print('Training accuracy U:', training_predictions)
        print()

    """
    Load Test Data
    """
    initial_label_qubits = np.load('data/' + dataset + '/new_ortho_d_final_vs_test_predictions.npy')[15]#.astype(np.float32)
    y_test = np.load('data/' + dataset + '/ortho_d_final_vs_test_predictions_labels.npy')#.astype(np.float32)

    """
    Rearrange test data to match new bitstring assignment
    """
    if partial:
        possible_labels = [5,7,8,9,4,0,6,1,2,3]
        assignment = possible_labels + list(range(10,16))
        reassigned_preds = np.array([i[assignment] for i in initial_label_qubits])
        reassigned_preds = np.array([i / np.sqrt(i.conj().T @ i) for i in reassigned_preds])
    else:
        reassigned_preds = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])

    outer_ket_states = reassigned_preds
    #.shape = n_train, dim_l**n_copies+1
    for k in range(n_copies):
        outer_ket_states = [np.kron(i, j) for i,j in zip(outer_ket_states, reassigned_preds)]

    """
    Perform Overlaps
    """
    #We want qubit formation:
    #|l_0^0>|l_1^0>|l_0^1>|l_1^1> |l_2^0>|l_3^0>|l_2^1>|l_3^1>...
    #I.e. act only on first 2 qubits on all copies.
    #Since unitary is contructed on first 2 qubits of each copy.
    #So we want U @ SWAP @ |copy_preds>
    print('Performing Overlaps!')
    preds_U = np.array([abs(U.dot(i)) for i in tqdm(outer_ket_states)])

    """
    Trace out other qubits/copies
    """
    print('Performing Partial Trace!')
    preds_U = np.array([np.diag(partial_trace(i, [0,1,2,3])) for i in tqdm(preds_U)])

    #Rearrange to 0,1,2,3,.. formation. This is req. for evaluate_classifier
    if partial:
        preds_U = np.array([i[assignment] for i in preds_U])

    test_predictions = evaluate_classifier_top_k_accuracy(preds_U, y_test, 1)
    print()
    print('Test accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_test, 1))
    print('Test accuracy U:', test_predictions)
    print()

    return None, test_predictions

def delta_efficent_deterministic_quantum_stacking(n_copies, v_col = True, dataset = 'fashion_mnist'):
    from numpy import linalg as LA
    print('Dataset: ', dataset)

    #initial_label_qubits = np.load('Classifiers/' + dataset + '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle = True)['arr_0'][15]
    #y_train = np.load('Classifiers/' + dataset + '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_labels.npy')
    #initial_label_qubits = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle = True)['arr_0'][15].astype(np.float32)
    initial_label_qubits = np.load('data/' + dataset + '/new_ortho_d_final_vs_training_predictions.npy')[15].astype(np.float32)
    y_train = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_labels.npy').astype(np.float32)

    initial_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])
    possible_labels = list(set(y_train))

    dim_l = initial_label_qubits.shape[1]
    outer_ket_states = initial_label_qubits

    dim_lc = dim_l ** (1 + n_copies)

    #.shape = n_train, dim_l**n_copies+1
    for k in range(n_copies):
        outer_ket_states = np.array([np.kron(i, j) for i,j in zip(outer_ket_states, initial_label_qubits)])

    V = []
    for l in tqdm(possible_labels):
        weighted_outer_states = np.zeros((dim_lc, dim_lc))
        for i in tqdm(initial_label_qubits[y_train == l]):
            ket = i

            for k in range(n_copies):
                ket = np.kron(ket, i)

            outer = np.outer(ket.conj(), ket)
            weighted_outer_states += outer

        #print('Performing SVD!')
        U, S = svd(weighted_outer_states)[:2]
        #print(U.shape)
        #print(S.shape)
        #assert()
        if v_col:
            #a = b = 16**n (using andrew's defn)
            a, b = U.shape
            p = int(np.log10(b)) - 1
            D_trunc = 16
            Vl = np.array(U[:, :b//16] @ np.sqrt(np.diag(S)[:b//16, :b//16]))
            #print(Vl.shape)
            #assert()
            #Vl = np.array(U[:, :10**p] @ np.sqrt(np.diag(S)[:10**p, :10**p]))
            #Vl = np.array(U[:, :D_trunc] @ np.sqrt(np.diag(S)[:D_trunc, :D_trunc]))
        else:
            Vl = np.array(U[:, :1] @ np.sqrt(np.diag(S)[:1, :1])).squeeze()

        V.append(Vl)

    V = np.array(V)
    if v_col:
        c, d, e = V.shape
        #V = np.pad(V, ((0,dim_l - c), (0,0), (0,dim_l**p - D_trunc))).transpose(0, 2, 1).reshape(d , -1)
        V = np.pad(V, ((0,dim_l - c), (0,0), (0,0))).transpose(0, 2, 1).reshape(dim_l*e, d)

    else:
        a, b = V.shape
        V = np.pad(V, ((0,dim_l - a), (0,0)))
    #np.save('V', V)
    print('Performing Polar Decomposition!')
    U = polar(V)[0]
    print('Finished Computing Stacking Unitary!')
    return U.astype(np.float32)

def sum_state_deterministic_quantum_stacking(n_copies, v_col = True, dataset = 'fashion_mnist'):
    from numpy import linalg as LA
    print('Dataset: ', dataset)

    initial_label_qubits = produce_psuedo_sum_states(dataset)
    y_train = range(10)
    initial_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])

    possible_labels = list(set(y_train))

    dim_l = initial_label_qubits.shape[1]
    outer_ket_states = initial_label_qubits

    dim_lc = dim_l ** (1 + n_copies)

    weighted_outer_states = np.zeros((dim_lc, dim_lc))
    for l in tqdm(possible_labels):
        for i in tqdm(initial_label_qubits[y_train == l]):
            ket = i

            for k in range(n_copies):
                ket = np.kron(ket, i)

            outer = np.outer(np.kron(bitstrings[l].squeeze().tensors[5].data, np.eye(dim_lc//16)[0]), ket)
            weighted_outer_states += outer
    print('Performing Polar Decomposition!')
    U = polar(weighted_outer_states)[0]
    print('Finished Computing Stacking Unitary!')
    return U.astype(np.float32)

def specific_quantum_stacking(n_copies, dataset, v_col = False):
    from numpy import linalg as LA

    """
    Load Data
    """
    #initial_label_qubits = np.load('Classifiers/fashion_mnist_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle = True)['arr_0'][15].astype(np.float32)
    #y_train = np.load('Classifiers/fashion_mnist_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_labels.npy').astype(np.float32)

    initial_label_qubits = np.load('data/' + dataset + '/new_ortho_d_final_vs_training_predictions.npy')[15].astype(np.float32)
    y_train = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_labels.npy').astype(np.float32)
    initial_label_qubits = [i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits]

    """
    Rearrange labels such that 5,7,8,9 are on the bitstrings:
    |00>(|00> + |01> + |10> + |11>)
    Also post-select (slice) such that stacking unitary is constructed only from bitstrings corresponding to labels 5,7,8,9
    Normalisation req. after post-selection
    """
    possible_labels = [5,7,8,9,4,0,6,1,2,3]
    assignment = possible_labels + list(range(10,16))
    reassigned_preds = np.array([i[assignment][:4] for i in initial_label_qubits])
    initial_label_qubits = reassigned_preds
    initial_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])

    """
    Construct V: Non-unitary stacking operator
    """
    dim_l = initial_label_qubits.shape[1]
    dim_lc = dim_l ** (1 + n_copies)

    #.shape = n_train, dim_l**n_copies+1
    outer_ket_states = initial_label_qubits
    for k in range(n_copies):
        outer_ket_states = np.array([np.kron(i, j) for i,j in zip(outer_ket_states, initial_label_qubits)])

    print('Computing Stacking Matrix!')

    V = []
    for l in tqdm(possible_labels[:4]):
        weighted_outer_states = np.zeros((dim_lc, dim_lc))
        for i in tqdm(initial_label_qubits[y_train == l]):
            ket = i

            for k in range(n_copies):
                ket = np.kron(ket, i)

            outer = np.outer(ket, ket.conj())
            weighted_outer_states += outer

        #print('Performing SVD!')
        U, S = svd(weighted_outer_states)[:2]

        if v_col:
            a, b = U.shape
            #Truncated to this amount to ensure V is square for polar decomp.
            Vl = np.array(U[:, :b//4] @ np.sqrt(np.diag(S)[:b//4, :b//4]))
        else:
            Vl = np.array(U[:, :1] @ np.sqrt(np.diag(S)[:1, :1])).squeeze()

        V.append(Vl)

    V = np.array(V)

    if v_col:
        c, d, e = V.shape
        V = V.transpose(0, 2, 1).reshape(c*e, d)

    print('Performing Polar Decomposition!')
    U = polar(V)[0]
    U = U.astype(np.float32)


    """
    Add conditionals and apply unitary to correct qubits via cirq (qiskit is slow af)
    """
    """
    print('Obtaining Circuit Unitary!')
    import cirq
    #stacking_qubits = list(np.array([[i,i+1] for i in range(2, (n_copies + 1) * 4, 4)]).flatten())
    stacking_qubits = [2,3,6,7,10,11]
    control_qubits = [i for i in range((n_copies + 1) * 4) if i not in stacking_qubits]

    sq = [cirq.LineQubit(i) for i in stacking_qubits]
    #sq = [cirq.LineQubit(i) for i in [0,1,4,5,8,9]]
    cq = [cirq.LineQubit(i) for i in control_qubits]
    #cq = [cirq.LineQubit(i) for i in [2,3,6,7,10,11]]

    U_cirq = cirq.MatrixGate(U).controlled(len(cq))
    circuit = cirq.Circuit()

    # You can create a circuit by appending to it
    circuit.append(cirq.X(q) for q in cq)
    circuit.append(U_cirq(*cq, *sq))
    circuit.append(cirq.X(q) for q in cq)
    print(circuit)
    U_circ = cirq.unitary(circuit)
    """
    def swap_gate(a,b,n):
        M = [np.eye(2, dtype = U.dtype) for _ in range(n)]
        result = sparse.eye(2**n, dtype = U.dtype) - sparse.eye(2**n, dtype = U.dtype)

        #Same as qiskit convention
        #a = n-a-1
        #b = n-b-1

        for i in [[1,0],[0,1]]:
            for j in [[1,0],[0,1]]:
                M[a] = np.outer(i,j) #|i><j|
                M[b] = np.outer(j,i) #|j><i|
                swap_gate = sparse.csr_matrix(M[0])
                for m in M[1:]:
                    swap_gate = sparse.kron(swap_gate, sparse.csr_matrix(m))
                result += swap_gate
        return result

    I = np.eye(4**(n_copies + 1), dtype = U.dtype)
    U_circ = sparse.kron(np.outer(I[0],I[0]), U)
    for i in tqdm(I[1:]):
        U_circ += sparse.kron(sparse.csr_matrix(np.outer(i, i)),sparse.csr_matrix(I))

    for i in tqdm(range(1, n_copies+1)):
        s_sparse = sparse.csr_matrix(swap_gate(2+4*(i-1), 2+4*(i-1) + 2*n_copies - 2*(i-1), 4 * (n_copies + 1)))
        U_circ = s_sparse @ U_circ @ s_sparse

        s_sparse = sparse.csr_matrix(swap_gate(2+4*(i-1)+1, 2+4*(i-1) + 2*n_copies - 2*(i-1) + 1, 4 * (n_copies + 1)))
        U_circ = s_sparse @ U_circ @ s_sparse

    return U_circ.astype(np.float32)

def stacking_on_confusion_matrix(max_copies, dataset = 'fashion_mnist'):

    initial_label_qubits = produce_psuedo_sum_states(dataset)
    permutation = load_brute_force_permutations(10,dataset)[1]

    rearranged_results = []
    for row in initial_label_qubits[permutation]:
        rearranged_results.append(row[permutation])
    rearranged_results = np.array(rearranged_results)

    total_results = []
    total_results.append(rearranged_results)
    f, axarr = plt.subplots(1,max_copies + 2)
    axarr[0].imshow(rearranged_results, cmap = "Greys")
    axarr[0].set_title('MNIST' + '\n No Stacking')
    axarr[0].set_xticks(range(10))
    axarr[0].set_yticks(range(10))
    axarr[0].set_xticklabels(permutation)
    axarr[0].set_yticklabels(permutation)
    color = ['black','black','white','black','black']
    for i in range(len(rearranged_results)):
        for k, j in enumerate(range(-2,3)):
            if i + j > -1 and i + j < 10:
                axarr[0].text(i,i+j,np.round(rearranged_results[i,i+j],3), color = color[k], ha="center", va="center", fontsize = 6)

    initial_label_qubits = np.pad(initial_label_qubits, ((0,0), (0,6)))
    for n_copies in range(max_copies + 1):
        U = delta_efficent_deterministic_quantum_stacking(n_copies, True, dataset)

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
        print('Performing Overlaps!')
        preds_U = np.array([abs(U.dot(i)) for i in tqdm(outer_ket_states)])

        """
        Trace out other qubits/copies
        """

        print('Performing Partial Trace!')
        preds_U = np.array([np.diag(partial_trace(i, [0,1,2,3])) for i in tqdm(preds_U)])
        preds_U = np.array([i / np.sqrt(i.conj().T @ i) for i in preds_U])

        rearranged_results = []
        for row in preds_U[permutation]:
            rearranged_results.append(row[permutation])
        rearranged_results = np.array(rearranged_results, dtype = np.float32)
        total_results.append(rearranged_results.T)

        axarr[n_copies+1].imshow(rearranged_results.T, cmap = "Greys")
        axarr[n_copies+1].set_title('MNIST' + f'\n Stacking: Copies = {n_copies+1}')
        axarr[n_copies+1].set_xticks(range(10))
        axarr[n_copies+1].set_yticks(range(10))
        axarr[n_copies+1].set_xticklabels(permutation)
        axarr[n_copies+1].set_yticklabels(permutation)
        for i in range(len(rearranged_results)):
            for k, j in enumerate(range(-2,3)):
                if i + j > -1 and i + j < 10:
                    axarr[n_copies+1].text(i,i+j,np.round(rearranged_results[i,i+j],3), color = color[k], ha="center", va="center", fontsize = 6)

        #np.save(dataset + '_stacked_confusion_matrix', total_results)
    #f.tight_layout()
    #plt.savefig(dataaset + '_stacking_confusion_matrix_results.pdf')
    plt.show()

def plot_confusion_matrix(dataset = 'fashion_mnist'):
    total_results = np.load('Classifiers/' + dataset + '_stacked_confusion_matrix(copy).npy')
    permutation = load_brute_force_permutations(10,dataset)[1]

    if dataset == 'fashion_mnist':
        dataset = 'FASHION MNIST'
    else:
        dataset = 'MNIST'
    f, axarr = plt.subplots(1,4)
    axarr[0].imshow(total_results[0], cmap = "Greys")
    axarr[0].set_title('\n No Stacking')
    axarr[0].set_xticks(range(10))
    axarr[0].set_yticks(range(10))
    axarr[0].set_xticklabels(permutation)
    axarr[0].set_yticklabels(permutation)
    color = ['black','black','white','black','black']
    for i in range(len(total_results[0])):
        axarr[0].text(i,i,f'{total_results[0][i,i]:.2f}', color = 'white', ha="center", va="center", fontsize = 7, fontweight = 'bold')

    for n_copies in range(2 + 1):

        axarr[n_copies+1].imshow(total_results[n_copies+1], cmap = "Greys")
        axarr[n_copies+1].set_title(f'Stacking:\n Copies = {n_copies+1}')
        axarr[n_copies+1].set_xticks(range(10))
        axarr[n_copies+1].set_yticks(range(10))
        axarr[n_copies+1].set_xticklabels(permutation)
        axarr[n_copies+1].set_yticklabels(permutation)
        for i in range(len(total_results[0])):
            axarr[n_copies+1].text(i,i,f'{total_results[n_copies+1].T[i,i]:.2f}', color = 'white', ha="center", va="center", fontsize = 7, fontweight = 'bold')
        #for i in range(len(total_results[n_copies+1])):
        #    for k, j in enumerate(range(-2,3)):
        #        if i + j > -1 and i + j < 10:
        #            axarr[n_copies+1].text(i,i+j,np.round(total_results[n_copies+1].T[i,i+j],3), color = color[k], ha="center", va="center", fontsize = 6)

        #np.save(dataset + '_stacked_confusion_matrix', total_results)
    f.tight_layout()
    plt.suptitle(dataset, y = 0.9)
    f.set_size_inches(12.5, 4.5)

    plt.savefig(dataset + '_stacking_confusion_matrix_results.pdf')
    #plt.savefig('test.pdf')
    plt.show()

def mps_stacking(dataset, n_copies):

    def generate_copy_state(QTN, n_copies):
        initial_QTN = QTN
        for _ in range(n_copies):
            QTN = QTN | initial_QTN
        return relabel_QTN(QTN)

    def relabel_QTN(QTN):
        qtn_data = []
        previous_ind = rand_uuid()
        for j, site in enumerate(QTN.tensors):
            next_ind = rand_uuid()
            tensor = qtn.Tensor(
                site.data, inds=(f"k{j}", previous_ind, next_ind), tags=oset([f"{j}"])
            )
            previous_ind = next_ind
            qtn_data.append(tensor)
        return qtn.TensorNetwork(qtn_data)

    #Upload Data
    initial_label_qubits = np.load('Classifiers/' + dataset + '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle = True)['arr_0'][15].astype(np.float32)
    y_train = np.load('Classifiers/' + dataset + '_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_labels.npy').astype(np.float32)

    trunc_label_qubits = initial_label_qubits[:100]
    trunc_labels = y_train[:100]

    #Convert predictions to MPS
    mps_predictions = mps_encoding(trunc_label_qubits, 4)

    #Add n_copies of same prediction state
    copied_predictions = [generate_copy_state(pred,n_copies) for pred in mps_predictions]

    #Add bitstring onto copied states

def peel_stacking(dataset):

    def create_density_predictions(dataset):

        def create_density_matrix(pred):
            tensor1 = qtn.Tensor(
                pred.data.reshape(16,-1), inds=("l0", "p")
            )
            tensor2 = qtn.Tensor(
                pred.data.reshape(16,-1), inds=("l1", "p")
            )
            return (tensor1 | tensor2) ^ all

        """
        Create Classifier
        """
        #load sum states. Assuming D=32 classifier for now.
        path = dataset + "_mixed_sum_states/D_total/" + f"sum_states_D_total_32/"

        sum_states = [load_qtn_classifier(path + f"digit_{i}") for i in range(10)]
        sum_states_data = [fMPO([site.data for site in sum_state.tensors]) for sum_state in sum_states]

        #Add sum states together and compress
        ortho_classifier_data = adding_centre_batches(sum_states_data, 32, 10, orthogonalise = True)[0]
        ortho_mpo_classifier = data_to_QTN(ortho_classifier_data.data)

        """
        Turn MPO isometries to unitaries
        """
        u_classifier = unitary_qtn(ortho_mpo_classifier)

        """
        Create training predictions
        """
        x_train, y_train, x_test, y_test = load_data(
            1000,1, shuffle=False, equal_numbers=True, dataset = dataset
        )
        x_train, y_train = arrange_data(x_train, y_train, arrangement='one class')
        mps_train = mps_encoding(x_train, 32)

        preds = [mps_image.H.squeeze() @ u_classifier.squeeze() for mps_image in tqdm(mps_train)]
        """
        Create prediction density matrices
        """

        density_preds = [create_density_matrix(pred) for pred in preds]
        return density_preds, y_train

    def create_copy_density_predictions(density_preds, n_copies):

        def generate_copy_state(QTN, n_copies):
            initial_QTN = QTN
            for _ in range(n_copies):
                QTN = QTN | initial_QTN
            return relabel_QTN(QTN)

        def relabel_QTN(QTN):
            qtn_data = []
            previous_ind = rand_uuid()
            for j, site in enumerate(QTN.tensors):
                next_ind = rand_uuid()
                tensor = qtn.Tensor(
                    np.expand_dims(np.expand_dims(site.data,-1),-1), inds=(f"l{j}",f"p{j}", previous_ind, next_ind), tags=oset([f"{j}"])
                )
                previous_ind = next_ind
                qtn_data.append(tensor)
            return qtn.TensorNetwork(qtn_data)
        """
        Generate copy states
        """
        training_copied_predictions = [generate_copy_state(pred,n_copies) for pred in density_preds]

        return training_copied_predictions

    n_copies = 1

    #d_preds, y_train = create_density_predictions(dataset)

    training_rhos = np.load('data/' + dataset + "/padded_ortho_d_final_32_vs_training_predictions.npy")
    y_train = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_labels.npy')

    #reduced_training_rhos = np.array(reduce(list.__add__, [list(training_rhos[i*5421 : i * 5421 + 5408]) for i in range(10)]))
    reduced_training_rhos = np.array(reduce(list.__add__, [list(training_rhos[i*5421 : i * 5421 + 100]) for i in range(10)]))
    reduced_y_train = np.array(reduce(list.__add__, [list(y_train[i*5421 : i * 5421 + 100]) for i in range(10)]))

    #training_preds = [abs(np.diag(i)) for i in reduced_training_rhos]
    #print('Training acc:', evaluate_classifier_top_k_accuracy(training_preds, reduced_y_train, 1))
    qtn_reduced_training_rhos = [qtn.Tensor(i, inds = ('d', 's')) for i in reduced_training_rhos]
    copied_rhos = create_copy_density_predictions(qtn_reduced_training_rhos,n_copies)
    fMPO_rhos = np.array([fMPO([site.data for site in QTN.tensors]) for QTN in copied_rhos])

    unitaries = []
    for l in tqdm(list(set(y_train))):
        class_rhos = fMPO_rhos[reduced_y_train == l]

        #Add predictions of same class together
        added_rhos = class_rhos[0]
        for mpo in class_rhos[1:]:
            added_rhos = added_rhos.add(mpo)

        qtn_added_rhos = data_to_QTN(added_rhos.data)
        #qtn_added_rhos.compress_all(max_bond = 32, inplace = True)

        #first round of SVDs
        class_unitaries = []
        class_unitaries_dag = []

        for num, site in enumerate(qtn_added_rhos.tensors):

            d,s,i,j = site.shape
            reshaped_site = site.data.reshape(d,s*i*j)

            U, _, __ = svd(reshaped_site)
            qtn_U = qtn.Tensor(U, inds = (f's{num}', f'q{num}'), tags = f'U{num}')
            qtn_U_dag = qtn.Tensor(U.conj().T, inds = (f'k{num}', f'w{num}'), tags = f'Ud{num}')

            class_unitaries.append(qtn_U)
            class_unitaries_dag.append(qtn_U_dag)

        #second round of SVDs
        #performs the SVDs on two tensors- not one.

        print(qtn_added_rhos)
        assert()
        #contract Us with TN
        for num, (U, U_dag) in enumerate(zip(class_unitaries, class_unitaries_dag)):
            qtn_added_rhos = (U_dag | (qtn_added_rhos | U)).contract_tags((f'U{num}', f'Ud{num}', f'{num}'))

        if qtn_added_rhos.num_tensors == 2:
            #fuse together indices
            qtn_added_rhos = (qtn_added_rhos ^ all).squeeze()
            qtn_added_rhos.fuse({'q': [ind for ind in qtn_added_rhos.inds if 'q' in ind]}, inplace = True)
            qtn_added_rhos.fuse({'w': [ind for ind in qtn_added_rhos.inds if 'w' in ind]}, inplace = True)

            U, _, __ = svd(qtn_added_rhos.data)
            qtn_U = qtn.Tensor(U, inds = (f's{num}', f'q{num}'), tags = f'U{2*n_copies}')
            qtn_U_dag = qtn.Tensor(U.conj().T, inds = (f'k{num}', f'w{num}'), tags = f'Ud{2*n_copies}')

            class_unitaries.append(qtn_U)
            class_unitaries_dag.append(qtn_U_dag)

        unitaries.append(class_unitaries)

    U0s = []
    U1s = []
    U2s = []

    for l in unitaries:
        U0s.append(l[0].data)
        U1s.append(l[1].data)
        U2s.append(l[2].data)

    print(np.array(U0s).shape)
    print(np.array(U1s).shape)
    print(np.array(U2s).shape)

    #shape: (label_side, other_side)
    #Want label side to be out-going side i.e. not cotracted over.

    U0s = qtn.Tensor( polar(np.pad(U0s, ((0,6) ,(0,0), (0,0)))[:,:,:1].transpose(0,2,1).reshape(16,16))[0], inds = ('q0','r0'), tags = 'U0')
    U0s_dag = qtn.Tensor(U0s.data.conj().T, inds = ('n0','m0'), tags = 'U0_dag') #U0s.H.reindex({'q0':'m0', 'p0':'l0'})

    U1s = qtn.Tensor( polar(np.pad(U1s, ((0,6) ,(0,0), (0,0)))[:,:,:1].transpose(0,2,1).reshape(16,16))[0], inds = ('q1','r1'), tags = 'U1')
    U1s_dag = qtn.Tensor(U1s.data.conj().T, inds = ('n1','m1'), tags = 'U1_dag') #U1s.H.reindex({'q1':'m1', 'p1':'l1'})

    U2s = qtn.Tensor(polar(np.pad(U2s, ((0,6) ,(0,0), (0,0)))[:,:,:16].transpose(0,2,1).reshape(256,256))[0].reshape(16,16,16,16), inds = ('r0', 'r1', 'p0','p1'), tags = 'U2')
    U2s_dag = qtn.Tensor(U2s.data.conj().transpose(3,2,1,0), inds = ('l0','l1','n0','n1'), tags = 'U2_dag')

    initial_partial_copied_rhos = [(crho.reindex({f'l{k}':f'p{k}' for k in range(1,n_copies+1)})^all).squeeze().data for crho in copied_rhos]
    initial_training = np.array([abs(np.diag(i)) for i in initial_partial_copied_rhos])
    print('Initial training acc:', evaluate_classifier_top_k_accuracy(initial_training, reduced_y_train, 1))

    stacked_copied_rhos = [(U0s | (U1s | (U2s | crho | U2s_dag) | U1s_dag) | U0s_dag)^all for crho in copied_rhos]

    stacked_partial_copied_rhos = [crho.reindex({f'm{k}':f'q{k}' for k in range(1,n_copies+1)}).contract().squeeze().data for crho in stacked_copied_rhos]


    stacked_training = np.array([abs(np.diag(i)) for i in stacked_partial_copied_rhos])
    result = evaluate_classifier_top_k_accuracy(stacked_training, reduced_y_train, 1)
    print('Stacked training acc:', result)

    assert()

def peel_stacking_fixed(dataset):

    def create_copy_density_predictions(density_preds, n_copies):

        def generate_copy_state(QTN, n_copies):
            initial_QTN = QTN
            for _ in range(n_copies):
                QTN = QTN | initial_QTN
            return relabel_QTN(QTN)

        def relabel_QTN(QTN):
            qtn_data = []
            previous_ind = rand_uuid()
            for j, site in enumerate(QTN.tensors):
                next_ind = rand_uuid()
                tensor = qtn.Tensor(
                    np.expand_dims(np.expand_dims(site.data,-1),-1), inds=(f"k{j}",f"s{j}", previous_ind, next_ind), tags=oset([f"{j}"])
                )
                previous_ind = next_ind
                qtn_data.append(tensor)
            return qtn.TensorNetwork(qtn_data)
        """
        Generate copy states
        """
        training_copied_predictions = [generate_copy_state(pred,n_copies) for pred in density_preds]

        return training_copied_predictions

    n_copies = 1

    training_rhos = np.load('data/' + dataset + "/padded_ortho_d_final_32_vs_training_predictions.npy")
    y_train = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_labels.npy')

    #reduced_training_rhos = np.array(reduce(list.__add__, [list(training_rhos[i*5421 : i * 5421 + 5408]) for i in range(10)]))
    reduced_training_rhos = np.array(reduce(list.__add__, [list(training_rhos[i*5421 : i * 5421 + 100]) for i in range(10)]))
    reduced_y_train = np.array(reduce(list.__add__, [list(y_train[i*5421 : i * 5421 + 100]) for i in range(10)]))

    qtn_reduced_training_rhos = [qtn.Tensor(i, inds = ('d', 's')) for i in reduced_training_rhos]
    copied_rhos = create_copy_density_predictions(qtn_reduced_training_rhos,n_copies)
    fMPO_rhos = np.array([fMPO([site.data for site in QTN.tensors]) for QTN in copied_rhos])

    V0s = []
    V1s = []
    for l in tqdm(list(set(y_train))):
        class_rhos = fMPO_rhos[reduced_y_train == l]

        #Add predictions of same class together
        added_rhos = class_rhos[0]
        for mpo in class_rhos[1:]:
            added_rhos = added_rhos.add(mpo)

        qtn_added_rhos = data_to_QTN(added_rhos.data)
        #qtn_added_rhos.compress_all(max_bond = 32, inplace = True)

        #first round of SVDs
        for num, site in enumerate(qtn_added_rhos.tensors):

            d,s,i,j = site.shape
            reshaped_site = site.data.reshape(d,s*i*j)

            U, _, __ = svd(reshaped_site)

            if num == 0:
                V0s.append(U[:,:1])
            if num == 1:
                V1s.append(U[:,:1])


    V0 = np.pad(V0s, ((0,6) ,(0,0), (0,0))).transpose(0,2,1).reshape(16,16)
    V1 = np.pad(V1s, ((0,6) ,(0,0), (0,0))).transpose(0,2,1).reshape(16,16)

    U0_final = qtn.Tensor(polar(V0)[0], inds = ('u0', 'a0'))
    U0_final_dag = qtn.Tensor(polar(V0)[0].conj().T, inds = ('m0', 'b0'))
    U1_final = qtn.Tensor(polar(V1)[0], inds = ('u1', 'a1'))
    U1_final_dag = qtn.Tensor(polar(V1)[0].conj().T, inds = ('m1', 'b1'))

    #U0_trunc = qtn.Tensor(svd(V0)[0], inds = ('s0', 't0'),tags = 'U0')
    U0_trunc = qtn.Tensor(svd(V0)[0], inds = ('t0', 's0'),tags = 'U0')
    #U0_trunc_dag = qtn.Tensor(svd(V0)[0].conj().T, inds = ('k0', 'l0'),tags = 'Ud0')
    U0_trunc_dag = qtn.Tensor(svd(V0)[0].conj().T, inds = ('l0', 'k0'),tags = 'Ud0')
    #U1_trunc = qtn.Tensor(svd(V1)[0], inds = ('s1', 't1'),tags = 'U1')
    U1_trunc = qtn.Tensor(svd(V1)[0], inds = ('t1', 's1'),tags = 'U1')
    #U1_trunc_dag = qtn.Tensor(svd(V1)[0].conj().T, inds = ('k1', 'l1'),tags = 'Ud1')
    U1_trunc_dag = qtn.Tensor(svd(V1)[0].conj().T, inds = ('l1', 'k1'),tags = 'Ud1')

    V2s = []
    for l in tqdm(list(set(y_train))):
        class_rhos = fMPO_rhos[reduced_y_train == l]

        #Add predictions of same class together
        added_rhos = class_rhos[0]
        for mpo in class_rhos[1:]:
            added_rhos = added_rhos.add(mpo)

        qtn_added_rhos = data_to_QTN(added_rhos.data)
        qtn_added_rhos = (U0_trunc_dag | qtn_added_rhos | U0_trunc).contract_tags(('U0', 'Ud0', '0'))
        qtn_added_rhos = (U1_trunc_dag | qtn_added_rhos | U1_trunc).contract_tags(('U1', 'Ud1', '1'))

        qtn_added_rhos = (qtn_added_rhos ^ all).squeeze()
        qtn_added_rhos.fuse({'l': [ind for ind in qtn_added_rhos.inds if 'l' in ind]}, inplace = True)
        qtn_added_rhos.fuse({'t': [ind for ind in qtn_added_rhos.inds if 't' in ind]}, inplace = True)


        U, S = svd(qtn_added_rhos.data)[:2]
        V2s.append(U[:,:16] @ np.sqrt(np.diag(S)[:16,:16]))

    V2 = np.pad(V2s, ((0,6) ,(0,0), (0,0))).transpose(0,2,1).reshape(256,256)
    #U2_final = qtn.Tensor(polar(V2)[0].reshape(16,16,16,16), inds = ('s0','s1','t0','t1'))
    U2_final = qtn.Tensor(polar(V2)[0].reshape(16,16,16,16), inds = ('a0','a1','s0','s1'))
    #U2_final_dag = qtn.Tensor(polar(V2)[0].conj().T.reshape(16,16,16,16), inds = ('k0','k1','l0','l1'))
    U2_final_dag = qtn.Tensor(polar(V2)[0].conj().T.reshape(16,16,16,16), inds = ('b0','b1','k0','k1'))


    initial_partial_copied_rhos = [(crho.reindex({f'k{k}':f's{k}' for k in range(1,n_copies+1)})^all).squeeze().data for crho in copied_rhos]
    initial_training = np.array([abs(np.diag(i)) for i in initial_partial_copied_rhos])
    print('Initial training acc:', evaluate_classifier_top_k_accuracy(initial_training, reduced_y_train, 1))

    stacked_copied_rhos = [(U0_final | (U1_final | (U2_final | crho | U2_final_dag) | U1_final_dag) | U0_final_dag)^all for crho in copied_rhos]
    stacked_partial_copied_rhos = [crho.reindex({f'u{k}':f'm{k}' for k in range(1,n_copies+1)}).contract().squeeze().data for crho in stacked_copied_rhos]

    stacked_training = np.array([abs(np.diag(i)) for i in stacked_partial_copied_rhos])
    result = evaluate_classifier_top_k_accuracy(stacked_training, reduced_y_train, 1)
    print('Stacked training acc:', result)

    assert()

def mpo_stacking(dataset):

    def create_copy_density_predictions(density_preds, n_copies):

        def generate_copy_state(QTN, n_copies):
            initial_QTN = QTN
            for _ in range(n_copies):
                QTN = QTN | initial_QTN
            return relabel_QTN(QTN)

        def relabel_QTN(QTN):
            qtn_data = []
            previous_ind = rand_uuid()
            for j, site in enumerate(QTN.tensors):
                next_ind = rand_uuid()
                tensor = qtn.Tensor(
                    np.expand_dims(np.expand_dims(site.data,-1),-1), inds=(f"k{j}",f"s{j}", previous_ind, next_ind), tags=oset([f"{j}"])
                )
                previous_ind = next_ind
                qtn_data.append(tensor)
            return qtn.TensorNetwork(qtn_data)
        """
        Generate copy states
        """
        training_copied_predictions = [generate_copy_state(pred,n_copies) for pred in density_preds]

        return training_copied_predictions

    def relabel_density_qtn(QTN, label):
        qtn_data = []
        previous_ind = rand_uuid()
        for j, site in enumerate(QTN.tensors):
            next_ind = rand_uuid()
            if j == QTN.num_tensors//2 :
                labelled_site = [site.data * i for i in np.eye(16)[label]]
                tensor = qtn.Tensor(
                    labelled_site, inds=("l",f"k{j}",f"s{j}", previous_ind, next_ind), tags=oset([f"{j}"])
                )
            else:
                tensor = qtn.Tensor(
                    site.data, inds=(f"k{j}",f"s{j}", previous_ind, next_ind), tags=oset([f"{j}"])
                )
            previous_ind = next_ind
            qtn_data.append(tensor)
        return qtn.TensorNetwork(qtn_data)

    def density_data_to_qtn(data):
        qtn_data = []
        previous_ind = rand_uuid()
        for j, site in enumerate(data):
            next_ind = rand_uuid()
            if j == len(data)//2 :
                tensor = qtn.Tensor(
                    site, inds=("l",f"k{j}",f"s{j}", previous_ind, next_ind), tags=oset([f"{j}"])
                )
            else:
                tensor = qtn.Tensor(
                    site, inds=(f"k{j}",f"s{j}", previous_ind, next_ind), tags=oset([f"{j}"])
                )
            previous_ind = next_ind
            qtn_data.append(tensor)
        return qtn.TensorNetwork(qtn_data)

    def add_labelled_density_states(a, b):
        new_data_1 = [
            1j
            * np.zeros(
                (
                    a.tensors[i].shape[0],
                    a.tensors[i].shape[1],
                    a.tensors[i].shape[2] + b.tensors[i].shape[2],
                    a.tensors[i].shape[3] + b.tensors[i].shape[3],
                )
            )
            for i in range(a.num_tensors//2)
        ]

        new_data_2 = [1j* np.zeros(
                (
                    a.tensors[a.num_tensors//2].shape[0],
                    a.tensors[a.num_tensors//2].shape[1],
                    a.tensors[a.num_tensors//2].shape[2],
                    a.tensors[a.num_tensors//2].shape[3] + b.tensors[a.num_tensors//2].shape[3],
                    a.tensors[a.num_tensors//2].shape[4] + b.tensors[a.num_tensors//2].shape[4],
                )
            )]

        new_data_3 = [
            1j
            * np.zeros(
                (
                    a.tensors[i].shape[0],
                    a.tensors[i].shape[1],
                    a.tensors[i].shape[2] + b.tensors[i].shape[2],
                    a.tensors[i].shape[3] + b.tensors[i].shape[3],
                )
            )
            for i in range(a.num_tensors//2 + 1, a.num_tensors)
        ]

        new_data = new_data_1 + new_data_2 + new_data_3

        for i in range(a.num_tensors):
            if i == 0:
                new_data[i] = np.concatenate([a.tensors[i].data, b.tensors[i].data], 3)
            elif i == a.num_tensors - 1:
                new_data[i] = np.concatenate([a.tensors[i].data, b.tensors[i].data], 2)
            elif i == a.num_tensors//2:
                new_data[i][:, :, :, : a.tensors[i].data.shape[3], : a.tensors[i].data.shape[4]] = a.tensors[i].data
                new_data[i][:, :, :, a.tensors[i].data.shape[3] :, a.tensors[i].data.shape[4] :] = b.tensors[i].data
            else:
                new_data[i][:, :, : a.tensors[i].data.shape[2], : a.tensors[i].data.shape[3]] = a.tensors[i].data
                new_data[i][:, :, a.tensors[i].data.shape[2] :, a.tensors[i].data.shape[3] :] = b.tensors[i].data

        return density_data_to_qtn(new_data)

    n_copies = 2

    training_rhos = np.load('data/' + dataset + "/padded_ortho_d_final_32_vs_training_predictions.npy")
    y_train = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_labels.npy')

    #reduced_training_rhos = np.array(reduce(list.__add__, [list(training_rhos[i*5421 : i * 5421 + 5408]) for i in range(10)]))
    reduced_training_rhos = np.array(reduce(list.__add__, [list(training_rhos[i*5421 : i * 5421 + 10]) for i in range(10)]))
    reduced_y_train = np.array(reduce(list.__add__, [list(y_train[i*5421 : i * 5421 + 10]) for i in range(10)]))

    qtn_reduced_training_rhos = [qtn.Tensor(i, inds = ('d', 's')) for i in reduced_training_rhos]
    copied_rhos = create_copy_density_predictions(qtn_reduced_training_rhos,n_copies)
    fMPO_rhos = np.array([fMPO([site.data for site in QTN.tensors]) for QTN in copied_rhos])


    #Add mixed states of the same class together & label them
    mixed_sum_states = []
    for l in tqdm(list(set(y_train))):
        class_rhos = fMPO_rhos[reduced_y_train == l]

        #Add predictions of same class together
        added_rhos = class_rhos[0]
        for mpo in class_rhos[1:]:
            added_rhos = added_rhos.add(mpo)

        qtn_added_rhos = data_to_QTN(added_rhos.data)
        labelled_qtn_added_rhos = relabel_density_qtn(qtn_added_rhos,l)
        mixed_sum_states.append(labelled_qtn_added_rhos)

    #Add different mixed labeled states together
    mixed_added_sum_states = mixed_sum_states[0]
    for i in mixed_sum_states[1:]:
        mixed_added_sum_states = add_labelled_density_states(mixed_added_sum_states, i)

    #Compression required
    mixed_added_sum_states.compress_all(max_bond = 32, inplace = True)

    #Orthogonalise added sumstates
    ortho_mixed_sum_states = []
    for i, site in enumerate(mixed_added_sum_states.tensors):
        if i == mixed_added_sum_states.num_tensors//2:
            l,d,s,i,j = site.shape
            U_site = polar(site.data.transpose(0,2,1,3,4).reshape(l*s,d*i*j))[0].reshape(l,s,d,i,j).transpose(0,2,1,3,4)
            ortho_mixed_sum_states.append(U_site)
        else:
            d,s,i,j = site.shape
            U_site = polar(site.data.transpose(0,2,1,3).reshape(d*i,s*j))[0].reshape(d,i,s,j).transpose(0,2,1,3)
            ortho_mixed_sum_states.append(U_site)

    U = density_data_to_qtn(ortho_mixed_sum_states)
    U_dag_data = []
    for k, i in enumerate(U.tensors):
        if k == U.num_tensors//2:
            U_dag_data.append(i.data.transpose(0,2,1,3,4).conj())
        else:
            U_dag_data.append(i.data.transpose(1,0,2,3).conj())

    U_dag = density_data_to_qtn(U_dag_data)

    initial_partial_copied_rhos = [(crho.reindex({f'k{k}':f's{k}' for k in range(1,n_copies+1)})^all).squeeze().data for crho in copied_rhos]
    initial_training = np.array([abs(np.diag(i)) for i in initial_partial_copied_rhos])
    print('Initial training acc:', evaluate_classifier_top_k_accuracy(initial_training, reduced_y_train, 1))

    U.reindex({f's{k}':f'a{k}' for k in range(U.num_tensors)}, inplace = True)
    U_dag.reindex({f'k{k}':f'a{k}' for k in range(U.num_tensors)}, inplace = True)
    U_dag.reindex({'l':'m'}, inplace = True)


    stacked_copied_rhos = [(U @ (crho @ U_dag)).squeeze().data for crho in tqdm(copied_rhos)]
    stacked_training = np.array([abs(np.diag(i)) for i in stacked_copied_rhos])
    print('Stacked training acc:', evaluate_classifier_top_k_accuracy(stacked_training, reduced_y_train, 1))

def update_V(initial_V, n_steps, dataset = 'mnist'):

    initial_label_qubits = np.load('data/' + dataset + '/new_ortho_d_final_vs_training_predictions.npy')[15].astype(np.float32)
    y_train = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_labels.npy').astype(np.float32)

    #evaluate_stacking_unitary(initial_V, dataset = 'mnist', training = False)

    initial_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in initial_label_qubits])
    possible_labels = list(set(y_train))
    n_copies = int(np.log2(initial_V.shape[0])/4)-1
    pI = np.eye(int(initial_V.shape[0]/16))


    alpha_results1 = []
    alpha_results2 = []
    #alpha_results3 = []
    #Update Step
    for alpha in np.logspace(-4,-1,num=4):
    #for alpha in [1, 0.001, 0.0001]:
    #for alpha in [0.0001]:
        V1 = initial_V
        V2 = initial_V
        #V3 = initial_V

        results1 = []
        results2 = []
        #results3 = []

        for n in tqdm(range(n_steps)):

            dV1 = np.zeros((initial_V.shape[0], initial_V.shape[0]),dtype = np.complex128)
            dV2 = np.zeros((initial_V.shape[0], initial_V.shape[0]),dtype = np.complex128)
            #dV3 = np.zeros((initial_V.shape[0], initial_V.shape[0]),dtype = np.complex128)

            for l in possible_labels:
                weighted_outer_states = np.zeros((initial_V.shape[0], initial_V.shape[0]))
                pL = np.outer(np.eye(16)[int(l)], np.eye(16)[int(l)])

                for i in initial_label_qubits[y_train == l]:
                    ket = i

                    for k in range(n_copies):
                        ket = np.kron(ket, i)

                    outer = np.outer(ket.conj(), ket)
                    weighted_outer_states += outer

                dV1 += np.kron(pL, pI) @ V1 @ weighted_outer_states
                dV2 += np.kron(pL, pI) @ V2 @ weighted_outer_states
                #dV3 += np.kron(pI, pL) @ V3 @ weighted_outer_states

            V1 = polar(V1 + alpha*polar(dV1.conj().T)[0])[0]
            V2 = polar(V2 + alpha*dV2.conj().T)[0]
            #V3 = polar(dV3.conj().T)[0]

            _, test_results1 = evaluate_stacking_unitary(V1, dataset = 'mnist', training = False)
            _, test_results2 = evaluate_stacking_unitary(V2, dataset = 'mnist', training = False)
            #_, test_results3 = evaluate_stacking_unitary(V3, dataset = 'mnist', training = False)

            results1.append(test_results1)
            results2.append(test_results2)
            #results3.append(test_results3)

        alpha_results1.append(results1)
        alpha_results2.append(results2)
        #alpha_results3.append(results3)

        np.save(f'update_V_results/rearranged_gradient_update_results_1_n_copies_{n_copies}', alpha_results1)
        np.save(f'update_V_results/rearranged_gradient_update_results_2_n_copies_{n_copies}', alpha_results2)
        #np.save(f'update_V_results/gradient_update_results_3_n_copies_{n_copies}', alpha_results3)

def update_mpo_V(initial_mpo_V, n_steps, dataset = 'mnist'):
    pass

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


    x = range(1,11)
    alphas = np.logspace(-4,1,num=6)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    #fig, ax1 = plt.subplots(1, 1)


    ax1.plot(x,best_V1_results, label = fr'$V = Polar(V + \alpha * Polar[dV]) \ ,\alpha:{np.round(alphas[np.argmax(V1_results_max)],5)}$')
    ax1.plot(x,best_V2_results,linestyle = 'dashed', label = fr'$V = Polar(V + \alpha * dV) \ ,\alpha:{np.round(alphas[np.argmax(V2_results_max)],5)}$')
    ax1.plot(x,best_V3_results,linestyle = 'dotted', label = fr'$V = Polar(dV) \ ,\alpha:{np.round(alphas[np.argmax(V3_results_max)],5)}$')
    ax1.axhline(0.8565,c = 'k', alpha = 0.4, label = 'Full Stacking 2 Copy Result')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('2 Copy Case. Best result plotted from '+ r'$\alpha = [0.0001,10]$')
    ax1.legend()

    alphas = np.logspace(-4,0,num=5)
    x = range(1,6)
    ax2.plot(x,best_two_copy_1_results, label = fr'$V = Polar(V + \alpha * Polar[dV]) \ ,\alpha:{np.round(alphas[np.argmax(two_copy_1_results_max)],5)}$')
    ax2.plot(x,best_two_copy_2_results,linestyle = 'dashed', label = fr'$V = Polar(V + \alpha * dV) \ ,\alpha:{np.round(alphas[np.argmax(two_copy_2_results_max)],5)}$')
    #ax2.plot(x,two_copy3,linestyle = 'dotted', label = fr'$V = Polar(dV)$')
    ax2.axhline(0.8703,c = 'k', alpha = 0.4, label = 'Full Stacking 3 Copy Result')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('3 Copy Case. Plotted for '+ r'$\alpha = [0.0001, 1]$')
    ax2.legend()
    plt.tight_layout()
    #plt.savefig('update_V_results/new_updating_V.pdf')
    plt.show()


    assert()



def mps_stacking(training_mps_predictions, test_mps_predictions, n_copies, bond_order, y_train, y_test):

    def generate_copy_state(QTN, n_copies):
        initial_QTN = QTN
        for _ in range(n_copies):
            QTN = QTN | initial_QTN
        return relabel_QTN(QTN)

    def relabel_QTN(QTN):
        qtn_data = []
        previous_ind = rand_uuid()
        for j, site in enumerate(QTN.tensors):
            next_ind = rand_uuid()
            tensor = qtn.Tensor(
                site.data, inds=(f"k{j}", previous_ind, next_ind), tags=oset([f"{j}"])
            )
            previous_ind = next_ind
            qtn_data.append(tensor)
        return qtn.TensorNetwork(qtn_data)

    #Add n_copies of same prediction state
    training_copied_predictions = [generate_copy_state(pred,n_copies) for pred in training_mps_predictions]
    #training_copied_predictions = training_mps_predictions

    #Create bitstrings to add onto copied states
    possible_labels = list(set(y_train))
    n_sites = training_copied_predictions[0].num_tensors

    hairy_bitstrings_data = create_hairy_bitstrings_data(
        possible_labels, n_sites
    )
    q_hairy_bitstrings = bitstring_data_to_QTN(
        hairy_bitstrings_data, n_sites
    )
    hairy_bitstrings_data = [label_last_site_to_centre(b) for b in q_hairy_bitstrings]
    q_hairy_bitstrings = centred_bitstring_to_qtn(hairy_bitstrings_data)


    #Parameters for batch adding label predictions
    D_batch = bond_order
    batch_nums = [4, 8, 13, 13]
    #batch_nums = [3, 13, 39, 139]
    #batch_nums = [2, 3, 5, 2, 5, 2, 5, 2, 10]
    #batch_nums = [10]

    #Batch adding copied predictions to create sum states
    print('Batch adding predictions...')
    sum_states = prepare_centred_batched_classifier(training_copied_predictions, y_train, q_hairy_bitstrings, D_batch, batch_nums)

    #Batch adding sum states to create stacking unitary
    D_final = bond_order
    batch_final = 10
    ortho_at_end = True
    classifier_data = adding_centre_batches(sum_states, D_final, batch_final, orthogonalise = ortho_at_end)[0]
    stacking_unitary = data_to_QTN(classifier_data.data)#.squeeze()

    #Evaluate mps stacking unitary
    test_copied_predictions = [generate_copy_state(pred,n_copies) for pred in test_mps_predictions]

    #Perform overlaps
    print('Performing overlaps...')
    stacked_predictions = np.array([np.abs((mps_image.H.squeeze() @ stacking_unitary.squeeze()).data) for mps_image in tqdm(test_copied_predictions)])
    result = evaluate_classifier_top_k_accuracy(stacked_predictions, y_test, 1)
    print()
    print('Test accuracy U:', result)
    print()

    return result

def tensor_network_stacking_experiment(dataset, max_n_copies, bond_order):

    #Upload Data
    training_label_qubits = np.load('data/' + dataset + '/new_ortho_d_final_vs_training_predictions.npy')[15].astype(np.float32)
    y_train = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_labels.npy')
    training_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in training_label_qubits])

    #training_label_qubits = np.array(reduce(list.__add__, [list(training_label_qubits[i*5421 : i * 5421 + 5408]) for i in range(10)]))
    #y_train = np.array(reduce(list.__add__, [list(y_train[i*5421 : i * 5421 + 5408]) for i in range(10)]))
    training_label_qubits = np.array(reduce(list.__add__, [list(training_label_qubits[i*5421 : i * 5421 + 10]) for i in range(10)]))
    y_train = np.array(reduce(list.__add__, [list(y_train[i*5421 : i * 5421 + 10]) for i in range(10)]))

    #Convert predictions to MPS
    print('Encoding predictions...')
    training_mps_predictions = mps_encoding(training_label_qubits, None)

    test_label_qubits = np.load('data/' + dataset + '/new_ortho_d_final_vs_test_predictions.npy')[15]
    y_test = np.load('data/' + dataset + '/ortho_d_final_vs_test_predictions_labels.npy')
    test_label_qubits = np.array([i / np.sqrt(i.conj().T @ i) for i in test_label_qubits])

    print('Test accuracy before:', evaluate_classifier_top_k_accuracy(test_label_qubits, y_test, 1))


    test_mps_predictions = mps_encoding(test_label_qubits, None)
    for i in range(max_n_copies):
        result = mps_stacking(training_mps_predictions, test_mps_predictions, i, bond_order, y_train, y_test)
        np.save(f'tensor_network_stacking_results/max_copies_{max_n_copies}_D_{bond_order}', result)




if __name__ == '__main__':

    tensor_network_stacking_experiment('mnist',10, 32)
    assert()




    n_copies = 1
    #plot_update_V()
    initial_V = delta_efficent_deterministic_quantum_stacking(n_copies, v_col = True, dataset = 'mnist')
    #initial_V = polar(np.random.randn(16**n_copies,16**n_copies) + 1.0j*np.random.randn(16**n_copies,16**n_copies))[0]
    update_V(initial_V, 10)
    assert()

    evaluate_stacking_unitary(initial_V, dataset = 'mnist', training = False)
    #mpo_stacking('mnist')
    #classical_stacking('fashion_mnist')
    assert()

    #mps_stacking('mnist',1)
    #for i in range(3):
    #    U = delta_efficent_deterministic_quantum_stacking(i, v_col = True, dataset = 'mnist')
    #    evaluate_stacking_unitary(U, dataset = 'mnist', training = True)
    #assert()
    #U = test(1, dataset = 'fashion_mnist')
    #classical_stacking()
    #assert()
    #plot_confusion_matrix('mnist')
    #results = []
    #for i in range(1,10):
        #print('NUMBER OF COPIES: ',i)
        #U = specific_quantum_stacking(i,'fashion_mnist', True)
        #training_predictions, test_predictions = evaluate_stacking_unitary(U, True, dataset = 'fashion_mnist', training = False)
        #results.append([training_predictions, test_predictions])
        #np.save('partial_stacking_results_new', results)
    #stacking_on_confusion_matrix(0, dataset = 'mnist')
    #U = delta_efficent_deterministic_quantum_stacking(1, dataset = 'mnist')
    #U = sum_state_deterministic_quantum_stacking(2, dataset = 'mnist')
