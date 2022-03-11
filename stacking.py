import quimb as qu
import quimb.tensor as qtn
from xmps.svd_robust import svd
from scipy.linalg import polar
import numpy as np
from variational_mpo_classifiers import evaluate_classifier_top_k_accuracy, classifier_predictions
from deterministic_mpo_classifier import unitary_extension
import autograd.numpy as anp
import tensorflow as tf

import pennylane as qml
from pennylane import numpy as p_np

from pennylane.templates.state_preparations import MottonenStatePreparation
from pennylane.templates.layers import StronglyEntanglingLayers


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
    qubit_axis = [(i, num_qubit + i) for i in range(num_qubit)
                  if i not in qubit_2_keep]
    minus_factor = [(i, 2 * i) for i in range(len(qubit_axis))]
    minus_qubit_axis = [(q[0] - m[0], q[1] - m[1])
                        for q, m in zip(qubit_axis, minus_factor)]
    rho_res = np.reshape(rho, [2, 2] * num_qubit)
    qubit_left = num_qubit - len(qubit_axis)
    for i, j in minus_qubit_axis:
        rho_res = np.trace(rho_res, axis1=i, axis2=j)
    if qubit_left > 1:
        rho_res = np.reshape(rho_res, [2 ** qubit_left] * 2)

    return rho_res

def delta_efficent_deterministic_quantum_stacking(n_copies, v_col = False):
    from numpy import linalg as LA
    #Shape: n_train,2**label_qubits
    #initial_label_qubits = np.array(np.load('results/stacking/initial_label_qubit_states_4.npy'), dtype = np.float64)
    #initial_label_qubits = np.load('models/initial_training_predictions_ortho_mpo_classifier.npy')
    initial_label_qubits = np.load('Classifiers/fashion_mnist/initial_training_predictions_ortho_mpo_classifier.npy')

    #y_train = np.load('models/big_dataset_train_labels.npy')
    y_train = np.load('Classifiers/fashion_mnist/big_dataset_training_labels.npy')
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

            outer = np.outer(ket, ket)
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
    """
    print('Performing Contractions!')
    #np.save('U', U)

    preds_U = np.array([abs(U @ i) for i in outer_ket_states])
    preds_U = np.array([i / np.sqrt(i @ i) for i in preds_U])
    #preds_V = np.array([abs(V @ i) for i in outer_ket_states])
    #preds_V = np.array([i / np.sqrt(i @ i) for i in preds_V])

    if v_col:
        print('Performing Partial Trace!')
        preds_U = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_U])
        #preds_V = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_V])

    #print('Accuracy V:', evaluate_classifier_top_k_accuracy(preds_V, y_train, 1))
    variational_label_qubits = np.load('models/trained_training_predictions_ortho_mpo_classifier.npy')
    print()
    print('Training accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_train, 1))
    print('Training accuracy after:', evaluate_classifier_top_k_accuracy(variational_label_qubits, y_train, 1))
    print('Training accuracy U:', evaluate_classifier_top_k_accuracy(preds_U, y_train, 1))
    """

    #initial_label_qubits = np.load('models/initial_test_predictions_ortho_mpo_classifier.npy')
    initial_label_qubits = np.load('Classifiers/fashion_mnist/initial_test_predictions_ortho_mpo_classifier.npy')
    #variational_label_qubits = np.load('models/trained_test_predictions_ortho_mpo_classifier.npy')
    outer_ket_states = initial_label_qubits
    #.shape = n_train, dim_l**n_copies+1
    for k in range(n_copies):
        outer_ket_states = np.array([np.kron(i, j) for i,j in zip(outer_ket_states, initial_label_qubits)])

    preds_U = np.array([abs(U @ i) for i in outer_ket_states])
    preds_U = np.array([i / np.sqrt(i @ i) for i in preds_U])
    #preds_V = np.array([abs(V @ i) for i in outer_ket_states])
    #preds_V = np.array([i / np.sqrt(i @ i) for i in preds_V])
    if v_col:
        print('Performing Partial Trace!')
        preds_U = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_U])
        #preds_V = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_V])

    y_test = np.load('Classifiers/fashion_mnist/big_dataset_test_labels.npy')
    #y_test = np.load('models/big_dataset_test_labels.npy')
    print()
    print('Test accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_test, 1))
    print('Test accuracy after:', evaluate_classifier_top_k_accuracy(variational_label_qubits, y_test, 1))
    print('Test accuracy U:', evaluate_classifier_top_k_accuracy(preds_U, y_test, 1))
    print()

def paralell_deterministic_quantum_stacking(n_copies, v_col = False):
    """
    Code in order to get 3 copies datapoint on rosalind machine.
    First simulation: Generate U & S.
    Second Simulation: Generate V, perform polar decomp.
    """



    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size() #Number of Cores
    rank = comm.Get_rank() #Core number

    numDataPerRank = 1

    sendbuf = np.zeros(numDataPerRank, dtype = np.float32)
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty([size,numDataPerRank], dtype = np.float32)

    print('Rank: ',rank,'online!')



    #Shape: n_train,2**label_qubits
    #initial_label_qubits = np.array(np.load('results/stacking/initial_label_qubit_states_4.npy'), dtype = np.float64)
    initial_label_qubits = np.load('models/initial_training_predictions_ortho_mpo_classifier.npy')

    #Shape: n_train, n_classes
    # No Need to project onto bitstring states, as evaluate picks highest
    # Which corresponds to bitstring state anyway
    #variational_label_qubits = np.array(np.load('results/stacking/final_label_qubit_states_4.npy'), dtype = np.float64)
    variational_label_qubits = np.load('models/trained_training_predictions_ortho_mpo_classifier.npy')

    y_train = np.load('models/big_dataset_train_labels.npy')
    #y_train = np.load('training_labels.npy')


    dim_l = initial_label_qubits.shape[1]
    outer_ket_states = initial_label_qubits

    #copy_qubits = np.ones(dim_l) / np.sqrt(dim_l)
    #copy_qubits = np.eye(dim_l)[0]

    #n_copies = 2
    dim_lc = dim_l ** (1 + n_copies)


    #.shape = n_train, dim_l**n_copies+1
    #for k in range(n_copies):
    #    outer_ket_states = np.array([np.kron(i, j) for i,j in zip(outer_ket_states, initial_label_qubits)])

    V = []
    #Each core does 1 label
    for L in variational_label_qubits.T[rank:rank+1]:
        weighted_outer_states = np.zeros((dim_lc, dim_lc))
        for i, fl  in enumerate(tqdm(L)):
            ket = initial_label_qubits[i]

            for k in range(n_copies):
                ket = np.kron(ket, initial_label_qubits[i])

            outer = np.outer(fl * ket, ket)
            weighted_outer_states += outer

        #print('Performing SVD!')
        U, S = svd(weighted_outer_states)[:2]

        np.save(f'U_rank_{rank}', U[:, dim_l])
        np.save(f'S_rank_{rank}', np.diag(S)[:dim_l, :dim_l])
        print('Rank: ',rank,'finished!')
    """

        if v_col:
            Vl = np.array(U[:, :dim_l] @ np.sqrt(np.diag(S)[:dim_l, :dim_l]))
        else:
            Vl = np.array(U[:, :1] @ np.sqrt(np.diag(S)[:1, :1])).squeeze()

        V.append(Vl)

    V = np.array(V)
    if v_col:
        a, b, c = V.shape
        V = np.pad(V, ((0,dim_l - a), (0,0), (0,0))).transpose(0, 2, 1).reshape(dim_l*c, b)
    else:
        a, b = V.shape
        V = np.pad(V, ((0,dim_l - a), (0,0)))

    print('Performing Polar Decomposition!')
    U = polar(V)[0]
    print('Performing Contractions!')


    preds_U = np.array([abs(U @ i) for i in outer_ket_states])
    preds_U = np.array([i / np.sqrt(i @ i) for i in preds_U])
    #preds_V = np.array([abs(V @ i) for i in outer_ket_states])
    #preds_V = np.array([i / np.sqrt(i @ i) for i in preds_V])

    if v_col:
        print('Performing Partial Trace!')
        preds_U = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_U])
        #preds_V = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_V])

    #print('Accuracy V:', evaluate_classifier_top_k_accuracy(preds_V, y_train, 1))
    print()
    print('Training accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_train, 1))
    print('Training accuracy after:', evaluate_classifier_top_k_accuracy(variational_label_qubits, y_train, 1))
    print('Training accuracy U:', evaluate_classifier_top_k_accuracy(preds_U, y_train, 1))

    initial_label_qubits = np.load('models/initial_test_predictions_ortho_mpo_classifier.npy')
    variational_label_qubits = np.load('models/trained_test_predictions_ortho_mpo_classifier.npy')
    outer_ket_states = initial_label_qubits
    #.shape = n_train, dim_l**n_copies+1
    for k in range(n_copies):
        outer_ket_states = np.array([np.kron(i, j) for i,j in zip(outer_ket_states, initial_label_qubits)])

    preds_U = np.array([abs(U @ i) for i in outer_ket_states])
    preds_U = np.array([i / np.sqrt(i @ i) for i in preds_U])
    #preds_V = np.array([abs(V @ i) for i in outer_ket_states])
    #preds_V = np.array([i / np.sqrt(i @ i) for i in preds_V])

    if v_col:
        print('Performing Partial Trace!')
        preds_U = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_U])
        #preds_V = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_V])

    #print('Accuracy V:', evaluate_classifier_top_k_accuracy(preds_V, y_train, 1))
    y_test = np.load('models/big_dataset_test_labels.npy')
    print()
    print('Test accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_test, 1))
    print('Test accuracy after:', evaluate_classifier_top_k_accuracy(variational_label_qubits, y_test, 1))
    print('Test accuracy U:', evaluate_classifier_top_k_accuracy(preds_U, y_test, 1))
    """




if __name__ == '__main__':
    classical_stacking()
    assert()
    delta_efficent_deterministic_quantum_stacking(2, True)
