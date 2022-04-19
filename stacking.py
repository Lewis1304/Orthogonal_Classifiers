import quimb as qu
import quimb.tensor as qtn
from xmps.svd_robust import svd
from scipy.linalg import polar
import numpy as np
from variational_mpo_classifiers import evaluate_classifier_top_k_accuracy, classifier_predictions
from deterministic_mpo_classifier import unitary_extension
import autograd.numpy as anp
import tensorflow as tf
from scipy import sparse
import qutip


from tools import load_data, arrange_data


from tqdm import tqdm

def classical_stacking():
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
    training_predictions = np.load('Classifiers/fashion_mnist/initial_training_predictions_ortho_mpo_classifier.npy')
    y_train = np.load('Classifiers/fashion_mnist/big_dataset_training_labels.npy')


    inputs = tf.keras.Input(shape=(training_predictions.shape[1],))
    #inputs = tf.keras.Input(shape=(qtn_prediction_and_ancillae_qubits.shape[1],))
    outputs = tf.keras.layers.Dense(10, activation = 'sigmoid')(inputs)
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
        epochs=2000,
        batch_size = 32,
        verbose = 1
    )
    #model.save('models/ortho_big_dataset_D_32')
    test_preds = np.load('Classifiers/fashion_mnist/initial_test_predictions_ortho_mpo_classifier.npy')
    y_test = np.load('Classifiers/fashion_mnist/big_dataset_test_labels.npy')
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

def evaluate_stacking_unitary(U, partial = False):
    """
    Evaluate Performance
    """
    n_copies = int(np.log2(U.shape[0])//4)-1

    """
    Load Training Data
    """
    initial_label_qubits = np.load('Classifiers/fashion_mnist_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle = True)['arr_0'][15].astype(np.float32)
    y_train = np.load('Classifiers/fashion_mnist_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_labels.npy').astype(np.float32)
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
        outer_ket_states = np.array([np.kron(i, j) for i,j in zip(outer_ket_states, reassigned_preds)])

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
    initial_label_qubits = np.load('Classifiers/fashion_mnist_mixed_sum_states/D_total/ortho_d_final_vs_test_predictions.npy')[15].astype(np.float32)
    y_test = np.load('Classifiers/fashion_mnist_mixed_sum_states/D_total/ortho_d_final_vs_test_predictions_labels.npy').astype(np.float32)

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
        outer_ket_states = np.array([np.kron(i, j) for i,j in zip(outer_ket_states, reassigned_preds)])

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

    return training_predictions, test_predictions

def delta_efficent_deterministic_quantum_stacking(n_copies, v_col = False):
    from numpy import linalg as LA

    initial_label_qubits = np.load('Classifiers/fashion_mnist_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle = True)['arr_0'][15]
    y_train = np.load('Classifiers/fashion_mnist_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_labels.npy')
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

        if v_col:
            #a = b = 16**n (using andrew's defn)
            a, b = U.shape
            p = int(np.log10(b)) - 1
            D_trunc = 16
            Vl = np.array(U[:, :b//16] @ np.sqrt(np.diag(S)[:b//16, :b//16]))
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
    return U.astype(np.float32)

def specific_quantum_stacking(n_copies, v_col = False):
    from numpy import linalg as LA

    """
    Load Data
    """
    initial_label_qubits = np.load('Classifiers/fashion_mnist_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_compressed.npz', allow_pickle = True)['arr_0'][15]
    y_train = np.load('Classifiers/fashion_mnist_mixed_sum_states/D_total/ortho_d_final_vs_training_predictions_labels.npy')
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

    U = U.astype(np.float32)
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


if __name__ == '__main__':
    #classical_stacking()
    #assert()
    #U = delta_efficent_deterministic_quantum_stacking(1, True)
    results = []
    for i in range(10):
        U = specific_quantum_stacking(i, True)
        training_predictions, test_predictions = evaluate_stacking_unitary(U, True)
        results.append([training_predictions, test_predictions])
        np.save('partial_stacking_results', results)
