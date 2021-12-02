import quimb as qu
import quimb.tensor as qtn
from xmps.svd_robust import svd
from scipy.linalg import polar
import numpy as np
from variational_mpo_classifiers import evaluate_classifier_top_k_accuracy, classifier_predictions
from deterministic_mpo_classifier import unitary_extension
import autograd.numpy as anp
import tensorflow as tf

from tqdm import tqdm

def classical_stacking(mps_images, labels, classifier, bitstrings):

    #Ancillae start in state |00...>
    #n=0
    #ancillae_qubits = np.eye(2**n)[0]
    #Tensor product ancillae with predicition qubits
    #Amount of ancillae equal to amount of predicition qubits
    #training_predictions = np.array([np.kron(ancillae_qubits, (mps_image.H @ classifier).squeeze().data) for mps_image in tqdm(mps_images)])
    training_predictions = np.array([abs((mps_image.H @ classifier).squeeze().data) for mps_image in mps_images])
    np.save('initial_label_qubit_states_4',training_predictions)

    #Create predictions
    test_training_predictions = np.array(classifier_predictions(classifier, mps_images, bitstrings))
    test_accuracy = evaluate_classifier_top_k_accuracy(test_training_predictions, labels, 1)
    print(test_accuracy)
    """
    x_train, y_train, x_test, y_test = load_data(
        100,10000, shuffle=False, equal_numbers=True
    )
    D_test = 32
    mps_test = mps_encoding(x_test, D_test)
    test_predictions = np.array(classifier_predictions(classifier, mps_test, bitstrings))
    accuracy = evaluate_classifier_top_k_accuracy(test_predictions, y_test, 1)
    print(accuracy)
    """
    """
    inputs = tf.keras.Input(shape=(28,28,1))
    x = tf.keras.layers.AveragePooling2D(pool_size = (2,2))(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras  .layers.Dense(10, activation = 'relu')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    """
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
        labels,
        epochs=10000,
        batch_size = 32,
        verbose = 0
    )

    trained_training_predictions = model.predict(training_predictions)
    np.save('final_label_qubit_states_4',trained_training_predictions)

    #np.save('trained_predicitions_1000_classifier_32_1000_train_images', trained_training_predictions)
    accuracy = evaluate_classifier_top_k_accuracy(trained_training_predictions, labels, 1)
    print(accuracy)

def quantum_stacking_with_ancillae(classifier, bitstrings, mps_images, labels, loss_func, loss_name):
    import quimb as qu
    import quimb.tensor as qtn

    def ansatz_circuit_U3(n, depth, gate2='CX', **kwargs):
        """Construct a circuit of single qubit and entangling layers.
        """
        def single_qubit_layer(circ, gate_round=None):
            """Apply a parametrizable layer of single qubit ``U3`` gates.
            """
            for i in range(circ.N):
                # initialize with random parameters
                #params = qu.randn(3, dist='uniform')
                params = np.zeros(3)
                circ.apply_gate(
                    'U3', *params, i,
                    gate_round=gate_round, parametrize=True)

        def two_qubit_layer(circ, gate2='CX', reverse=False, gate_round=None):
            """Apply a layer of constant entangling gates.
            """
            regs = range(0, circ.N - 1)
            if reverse:
                regs = reversed(regs)

            for i in regs:
                circ.apply_gate(
                    gate2, i, i + 1, gate_round=gate_round)

        circ = qtn.Circuit(n, **kwargs)

        for r in range(depth):
            # single qubit gate layer
            single_qubit_layer(circ, gate_round=r)

            # alternate between forward and backward CZ layers
            two_qubit_layer(
                circ, gate2=gate2, gate_round=r, reverse=r % 2 == 0)

        # add a final single qubit layer
        single_qubit_layer(circ, gate_round=r + 1)

        return circ

    def ansatz_circuit_SU4(n, depth, **kwargs):
        """Construct a circuit of SU4s
        """

        def single_qubit_layer(circ, gate_round=None):
            """Apply a parametrizable layer of single qubit ``U3`` gates.
            """
            for i in range(circ.N):
                # initialize with random parameters
                #params = np.zeros(3)
                params = qu.randn(3, dist='uniform')
                circ.apply_gate(
                    'U3', *params, i,
                    gate_round=gate_round, parametrize=True)


        def two_qubit_layer(circ, gate_round=None):
            """Apply a layer of constant entangling gates.
            """
            regs = range(0, circ.N - 1)
            params = qu.randn(15, dist='uniform')
            #params = np.zeros(15)

            for i in regs:
                circ.apply_gate(
                    'su4',*params, i, i + 1, gate_round=gate_round, parametrize = True, contract = False)

        circ = qtn.Circuit(n, **kwargs)

        for r in range(depth):
            # single qubit gate layer
            #single_qubit_layer(circ, gate_round=r)

            # alternate between forward and backward CZ layers
            two_qubit_layer(
                circ, gate_round=r)

        # add a final single qubit layer
        single_qubit_layer(circ, gate_round=r + 1)

        return circ

    def overlap(predicition, V, bitstring):
        return (predicition.squeeze().H @ (V @ bitstring.squeeze())).norm()#**2

    def optimiser(circ):
        optmzr = qtn.TNOptimizer(
                circ,  # our initial input, the tensors of which to optimize
                loss_fn = lambda V: loss(V, qtn_prediction_and_ancillae_qubits, qtn_bitstrings, labels),
                norm_fn=normalize_tn,
                autodiff_backend="autograd",  # {'jax', 'tensorflow', 'autograd'}
                optimizer="nadam",  # supplied to scipy.minimize
                )
        return optmzr

    #Reshape label site into qubit states
    n_label_qubits = 4
    n_ancillae =  n_label_qubits
    n_total = n_ancillae + n_label_qubits

    bitstrings_qubits = [bitstring.squeeze().tensors[-1].data.reshape(2,2,2,2) for bitstring in bitstrings]
    qtn_bitstrings = [qtn.Tensor(bitstring_qubits, inds = [f'b{n_total-4}', f'b{n_total-3}', f'b{n_total-2}', f'b{n_total-1}']) for bitstring_qubits in bitstrings_qubits]


    #Ancillae start in state |00...>
    ancillae_qubits = np.eye(2**n_ancillae)[0]
    #ancillae_qubits = np.ones(2**n_ancillae) / (2**(n_ancillae/2))
    #Tensor product ancillae with predicition qubits
    #Amount of ancillae equal to amount of predicition qubits
    qtn_prediction_and_ancillae_qubits = [qtn.Tensor(np.kron(ancillae_qubits, (mps_image.H @ classifier).squeeze().data).reshape(*[2]*n_total), inds = [f'k{i}' for i in range(n_total)]) for mps_image in mps_images]
    #Normalise predictions
    qtn_prediction_and_ancillae_qubits = [i/(i.H @ i)**0.5 for i in qtn_prediction_and_ancillae_qubits]


    circ = ansatz_circuit_U3(n_total, n_total)
    V = circ.uni
    #V.draw(color=[f'ROUND_{i}' for i in range(n_total + 1)], show_inds=True, show_tags = False)
    #V.draw(color=['U3', 'SU4'], show_inds=True)

    #test = (V.contract(all) & qtn_bitstrings[0]) & qtn_prediction_and_ancillae_qubits[0]
    #print(test.contract(all).data)
    #test.draw(show_tags = False)
    #print(qtn_prediction_and_ancillae_qubits[0].H @ qtn_prediction_and_ancillae_qubits[0])
    #print(test.H @ test)
    print(f'Loss function: {loss_name}')

    predictions = [[overlap(p, V, b) for b in qtn_bitstrings] for p in qtn_prediction_and_ancillae_qubits]
    #print(predictions)
    accuracies = [evaluate_classifier_top_k_accuracy(predictions, labels, 1)]
    losses = [loss_func(V, qtn_prediction_and_ancillae_qubits, qtn_bitstrings, labels)]
    print(f'Accuracy before: {accuracies[0]}')
    assert()
    for _ in tqdm(range(1)):

        tnopt = qtn.TNOptimizer(
        V,                        # the tensor network we want to optimize
        loss_func,                     # the function we want to minimize
        loss_kwargs = {'mps_train': qtn_prediction_and_ancillae_qubits,
                        'q_hairy_bitstrings': qtn_bitstrings,
                        'y_train': labels},
        tags=['U3'],              # only optimize U3 tensors
        autodiff_backend='autograd',   # use 'autograd' for non-compiled optimization
        optimizer='nadam',     # the optimization algorithm
        )

        V = tnopt.optimize_basinhopping(n=500, nhop=10)

        losses.append(tnopt.loss)
        predictions = [[overlap(p, V, b) for b in qtn_bitstrings] for p in qtn_prediction_and_ancillae_qubits]
        accuracies.append(evaluate_classifier_top_k_accuracy(predictions, labels, 1))
        np.save(f'losses_{loss_name}_su4', losses)
        np.save(f'accuracies_{loss_name}_su4', accuracies)


    #predictions = [[overlap(p, V_opt, b) for b in qtn_bitstrings] for p in qtn_prediction_and_ancillae_qubits]
    #accuracies = [evaluate_classifier_top_k_accuracy(predictions, labels, 1)]
    #print(f'Accuracy after: {accuracies[0]}')

def quantum_stacking_with_copy_qubits(classifier, bitstrings, mps_images, labels, n_copies, U_param = None):
    import quimb as qu
    import quimb.tensor as qtn

    """
    Alternating U3 and CX Gates
    Depth inc. one layer of each
    """
    def ansatz_circuit_U3(n, depth, gate2='CX', **kwargs):
        """Construct a circuit of single qubit and entangling layers.
        """
        def single_qubit_layer(circ, gate_round=None):
            """Apply a parametrizable layer of single qubit ``U3`` gates.
            """
            for i in range(circ.N):
                # initialize with random parameters
                #params = qu.randn(3, dist='uniform')
                params = np.zeros(3)
                circ.apply_gate(
                    'U3', *params, i,
                    gate_round=gate_round, parametrize=True)

        def two_qubit_layer(circ, gate2='CX', reverse=False, gate_round=None):
            """Apply a layer of constant entangling gates.
            """
            regs = range(0, circ.N - 1)
            if reverse:
                regs = reversed(regs)

            for i in regs:
                circ.apply_gate(
                    gate2, i, i + 1, gate_round=gate_round)

        circ = qtn.Circuit(n, **kwargs)

        for r in range(depth):
            # single qubit gate layer
            single_qubit_layer(circ, gate_round=r)

            # alternate between forward and backward CZ layers
            two_qubit_layer(
                circ, gate2=gate2, gate_round=r, reverse=r % 2 == 0)

        # add a final single qubit layer
        single_qubit_layer(circ, gate_round=r + 1)

        return circ

    """
    General SU4 gate layers. W/ U3 layer at end
    """
    def ansatz_circuit_SU4(n, depth, **kwargs):
        """Construct a circuit of SU4s
        """

        def single_qubit_layer(circ, gate_round=None):
            """Apply a parametrizable layer of single qubit ``U3`` gates.
            """
            for i in range(circ.N):
                # initialize with random parameters
                #params = np.zeros(3)
                params = qu.randn(3, dist='uniform')
                circ.apply_gate(
                    'U3', *params, i,
                    gate_round=gate_round, parametrize=True)


        def two_qubit_layer(circ, gate_round=None):
            """Apply a layer of constant entangling gates.
            """
            regs = range(0, circ.N - 1)
            params = qu.randn(15, dist='uniform')
            #params = np.zeros(15)

            for i in regs:
                circ.apply_gate(
                    'su4',*params, i, i + 1, gate_round=gate_round, parametrize = True, contract = False)

        circ = qtn.Circuit(n, **kwargs)

        for r in range(depth):
            two_qubit_layer(
                circ, gate_round=r)

        # add a final single qubit layer
        single_qubit_layer(circ, gate_round=r + 1)

        return circ

    """
    Overlap between:
    label qubits and copies (predicition)
    Circuit (V)
    Bitstring, with post-selection of copies on |000..>.
    If tracing over copy states, need to include norm()
    """
    def overlap(predicition, V, bitstring):
        return (predicition.squeeze().H @ (V @ bitstring.squeeze())).norm()#**2

    """
    Cross Entropy Loss function.
    Predictions are not normalized
    """
    def loss_func(classifier, mps_train, q_hairy_bitstrings, y_train):
        overlaps = [
            anp.log(overlap(mps_train[i], classifier, q_hairy_bitstrings[y_train[i]]) ) for i in range(len(mps_train))]
        return -np.sum(overlaps) / len(mps_train)
        """
        #im_ov = [(p.squeeze().H @ classifier) for p in mps_train]
        #predictions = [[(pv @ b.squeeze()).norm() for b in q_hairy_bitstrings] for pv in im_ov]
        predictions = [[overlap(p, classifier, b) for b in q_hairy_bitstrings] for p in mps_train]
        n_predicitions = [i/anp.sqrt(np.dot(i,i)) for i in predictions]
        overlaps = [anp.log(p[i]) for p, i in zip(predictions, y_train)]
        return -np.sum(overlaps) / len(mps_train)
        """

    """
    Parameters for experiment.
    n_copies = total number of copy states.
    """
    if U_param is not None:
        n_copies = 1
    n_label_qubits = int(np.log2(bitstrings[0].squeeze().tensors[-1].data.shape)[0])
    n_total = (n_copies * n_label_qubits) + n_label_qubits

    """
    Post select copy qubits in |000..> state.
    """
    #ancillae_qubits = np.eye(2**(n_copies * n_label_qubits))[0]
    #bitstrings_qubits = [np.kron(ancillae_qubits, bitstring.squeeze().tensors[-1].data).reshape(*[2]*n_total) for bitstring in bitstrings]
    #qtn_bitstrings = [qtn.Tensor(bitstring_qubits, inds = [f'b{i}' for i in range(n_total)]) for bitstring_qubits in bitstrings_qubits]

    """
    Trace over all copy qubits state
    """
    bitstrings_qubits = [bitstring.squeeze().tensors[-1].data.reshape(*[2]*n_label_qubits) for bitstring in bitstrings]
    #qtn_bitstrings = [qtn.Tensor(bitstring_qubits, inds = [f'b{n_total-4}', f'b{n_total-3}', f'b{n_total-2}', f'b{n_total-1}']) for bitstring_qubits in bitstrings_qubits]
    qtn_bitstrings = [qtn.Tensor(bitstring_qubits, inds = [f'b{n_total-4}', f'b{n_total-3}', f'b{n_total-2}', f'b{n_total-1}']) for bitstring_qubits in bitstrings_qubits]

    """
    Generate prediction states
    """
    prediction_qubits = [(mps_image.H @ classifier).squeeze().data for mps_image in mps_images]
    """
    Normalise predictions
    """
    prediction_qubits = [i/np.sqrt(np.dot(i,i)) for i in prediction_qubits]


    initial_incorrect_predictions = prediction_qubits#[prediction_qubits[6], prediction_qubits[8], prediction_qubits[9]]
    incorrect_labels = labels#[labels[6], labels[8], labels[9]]

    np.save('results/stacking/initial_incorrect_predictions', initial_incorrect_predictions)
    np.save('results/stacking/incorrect_labels', incorrect_labels)


    """
    Construct input state.
    I.e. tensor product label qubit with copies
    """
    prediction_and_copy_qubits = prediction_qubits
    for k in range(n_copies):
        prediction_and_copy_qubits = [np.kron(i, j) for i,j in zip(prediction_and_copy_qubits, prediction_qubits)]
    qtn_prediction_and_copy_qubits = [qtn.Tensor(j.reshape(*[2]*n_total), inds = [f'k{i}' for i in range(n_total)]) for j in prediction_and_copy_qubits]

    """
    #Log depth. circuit
    """
    depth = int(np.ceil(np.log2(n_total)))
    """
    #Round to nearest power of 2. (means identity gives original result otherwise bitflips)
    """
    if depth % 2 != 0:
        depth += 1

    if U_param is not None:
        V = U_param
    else:
        circ = ansatz_circuit_U3(n_total, depth)
        V = circ.get_uni(transposed=True)
    #V.draw(color=[f'ROUND_{i}' for i in range(n_total + 1)], show_inds=True, show_tags = False)
    #V.draw(color=['U3', 'CX'], show_inds=True)

    """
    Collect predictions from circuit.
    Should be same as initial, since V is identity
    """
    predictions = [[overlap(p, V, b) for b in qtn_bitstrings] for p in tqdm(qtn_prediction_and_copy_qubits)]
    """
    Normalise predicitions (after measurement). Doesn't affect accuracy
    """
    predictions = [i/np.sqrt(np.dot(i,i)) for i in predictions]

    #print(np.round(predictions[6],5))
    #print(predictions)
    """
    Initial accuracy & loss
    """
    accuracies = [evaluate_classifier_top_k_accuracy(predictions, labels, 1)]
    losses = [loss_func(V, qtn_prediction_and_copy_qubits, qtn_bitstrings, labels)]
    print(f'Accuracy before: {accuracies[0]}')

    for _ in tqdm(range(1)):

        tnopt = qtn.TNOptimizer(
        V,                        # the tensor network we want to optimize
        loss_func,                     # the function we want to minimize
        loss_kwargs = {'mps_train': qtn_prediction_and_copy_qubits,
                        'q_hairy_bitstrings': qtn_bitstrings,
                        'y_train': labels},
        tags=['SU4'],              # only optimize U3 tensors
        autodiff_backend='autograd',   # use 'autograd' for non-compiled optimization
        optimizer='nadam',     # the optimization algorithm
        )

        V = tnopt.optimize(100)#_basinhopping(n=10, nhop=10)

        losses.append(tnopt.losses)
        predictions = [[overlap(p, V, b) for b in qtn_bitstrings] for p in qtn_prediction_and_copy_qubits]
        predictions = [i/np.sqrt(np.dot(i,i)) for i in predictions]
        accuracies.append(evaluate_classifier_top_k_accuracy(predictions, labels, 1))

        np.save('results/stacking/variational_incorrect_predictions_test_3', predictions)
        np.save(f'results/stacking/quantum_stacking_2_losses_test_3', losses)
        np.save(f'results/stacking/quantum_stacking_2_accuracies_test_3', accuracies)


    #predictions = [[overlap(p, V_opt, b) for b in qtn_bitstrings] for p in qtn_prediction_and_ancillae_qubits]
    #accuracies = [evaluate_classifier_top_k_accuracy(predictions, labels, 1)]
    #print(f'Accuracy after: {accuracies[0]}')

def parameterise_deterministic_U(U):

    """
    General SU4 gate layers.
    """
    def ansatz_circuit_SU4(n, depth, **kwargs):
        """Construct a circuit of SU4s
        """

        def two_qubit_layer(circ, gate_round=None):
            """Apply a layer of constant entangling gates.
            """
            regs = range(0, circ.N - 1)
            params = qu.randn(15, dist='uniform')
            #params = np.zeros(15)

            for i in regs:
                circ.apply_gate(
                    'su4',*params, i, i + 1, gate_round=gate_round, parametrize = True, contract = False)

        circ = qtn.Circuit(n, **kwargs)

        for r in range(depth):
            two_qubit_layer(
                circ, gate_round=r)

        return circ

    """
    #Log depth. circuit
    """
    n_total = int(np.ceil(np.log2(U.shape[0])))
    depth = int(np.ceil(np.log2(n_total)))
    U = qtn.Tensor(data = unitary_extension(U).reshape(* [2] * (2 * n_total)), inds=[f'k{i}' for i in range(n_total)] + [f'b{i}' for i in range(n_total)], tags={'U_TARGET'})


    """
    #Round to nearest power of 2. (means identity gives original result otherwise bitflips)
    """
    if depth % 2 != 0:
        depth += 1

    circ = ansatz_circuit_SU4(n_total, depth)
    V = circ.get_uni(transposed=True)
    #V.draw(color=[f'ROUND_{i}' for i in range(n_total + 1)], show_inds=True, show_tags = False)
    #V.draw(color=['U3', 'CX'], show_inds=True)

    def loss(V, U):
        return 1 - abs((V.H & U).contract(all, optimize='auto-hq')) / 2**n_total

    print(f'Loss before: {loss(V, U)}')
    tnopt = qtn.TNOptimizer(
    V,                        # the tensor network we want to optimize
    loss,                     # the function we want to minimize
    loss_constants={'U': U},  # supply U to the loss function as a constant TN
    tags=['SU4'],              # only optimize su4 tensors
    autodiff_backend='autograd',   # use 'autograd' for non-compiled optimization
    optimizer='L-BFGS-B',     # the optimization algorithm
    )

    V_opt = tnopt.optimize_basinhopping(n=100, nhop=10)

    return V_opt

def deterministic_quantum_stacking(y_train, bitstrings, n_copies, classifier, v_col = False):

    #Shape: n_train,2**label_qubits
    initial_label_qubits = np.load('results/stacking/initial_label_qubit_states_4.npy')

    #Shape: n_train, n_classes
    # No Need to project onto bitstring states, as evaluate picks highest
    # Which corresponds to bitstring state anyway
    variational_label_qubits = np.load('results/stacking/final_label_qubit_states_4.npy')

    #print('Accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_train, 1))
    #print('Accuracy After:', evaluate_classifier_top_k_accuracy(variational_label_qubits, y_train, 1))

    dim_l = initial_label_qubits.shape[1]
    outer_ket_states = initial_label_qubits

    #copy_qubits = np.ones(dim_l) / np.sqrt(dim_l)
    #copy_qubits = np.eye(dim_l)[0]

    #n_copies = 2
    dim_lc = dim_l ** (1 + n_copies)

    #.shape = n_train, dim_l**n_copies+1
    for k in range(n_copies):
        outer_ket_states = np.array([np.kron(i, j) for i,j in zip(outer_ket_states, initial_label_qubits)])

    """
    #.shape = n_train, dim_l**n_copies+1, dim_l**n_copies+1
    outer_states = np.array([np.outer(i.conj().T, i) for i in outer_ket_states])
    print('Outer label qubits shape:', outer_states.shape)

    #.shape = n_classes, dim_l**n_copies+1
    weighted_summed_states = np.zeros((len(set(y_train)), dim_lc, dim_lc), dtype = np.complex128)
    for i in tqdm(range(len(outer_states))):
        weighted_outer_states = []
        for fl in variational_label_qubits[i]:
            weighted_outer_states.append(outer_states[i] * fl)
        weighted_summed_states += weighted_outer_states
    #weighted_summed_states_inefficent = np.sum([[ outer_states[i] * fl for fl in variational_label_qubits[i]] for i in range(len(outer_states))], axis = 0)
    """

    weighted_summed_states = np.zeros((len(set(y_train)), dim_lc, dim_lc))#, dtype = np.complex128)
    #Loop over all predictions
    #for i in tqdm(range(len(variational_label_qubits))):
    for i in range(len(variational_label_qubits)):
        weighted_outer_states = []

        #stack (append) different labels
        for fl in variational_label_qubits[i]:
            #weighted_proto_sum_states = np.zeros((dim_lc, dim_lc), dtype = np.complex128)
            ket = initial_label_qubits[i]

            #Construct label qubit + copies state
            for k in range(n_copies):
                ket = np.kron(ket, initial_label_qubits[i])
            #    ket = np.kron(ket, copy_qubits)

            #Outer product total state to convert into matrixs
            outer = np.outer(ket, ket)

            #Weight each element via output of NN
            weighted_outer_states.append(fl * outer)

        #Sum over all image predictions
        weighted_summed_states += weighted_outer_states


    #U and S.shape = dim_l**n_copies+1, dim_l**n_copies+1
    #print('Performing SVDs!')
    #USs = [svd(i)[:2] for i in tqdm(weighted_summed_states)]
    USs = [svd(i)[:2] for i in weighted_summed_states]

    if v_col:
        D = dim_l
        V = np.array([i[0][:, :D] @ np.sqrt(np.diag(i[1])[:D, :D]) for i in USs]).squeeze()
        a, b, c = V.shape
        V = np.pad(V, ((0,D-a), (0,0), (0,0))).transpose(0,2,1).reshape(D*c,b)

    #V.shape = dim_l**n_copies+1 , dim_l
    else:
        V = np.array([i[0][:, :1] @ np.sqrt(np.diag(i[1])[:1, :1]) for i in USs]).squeeze()#.conj().T


    #print('Performing Polar Decomposition!')
    U = polar(V)[0]

    #print('Performing Contractions!')

    preds_U = np.array([abs(U @ i) for i in outer_ket_states])
    preds_U = np.array([i / np.sqrt(i @ i) for i in preds_U])
    preds_V = np.array([abs(V @ i) for i in outer_ket_states])
    preds_V = np.array([i / np.sqrt(i @ i) for i in preds_V])

    if v_col:
        preds_U = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_U])
        preds_V = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_V])

    #print(preds_U[0,:30].shape)
    #print(preds_U[0,:30])
    #assert()

    print('Accuracy V:', evaluate_classifier_top_k_accuracy(preds_V, y_train, 1))
    print('Accuracy U:', evaluate_classifier_top_k_accuracy(preds_U, y_train, 1))
    """
    for i in range(10):
        plt.bar(range(10), preds_U[i], color = 'tab:blue', label = 'ortho')
        plt.bar(range(10), preds_V[i], fill = False, edgecolor = 'tab:orange', linewidth = 1.5, label = 'non_ortho')
        plt.legend()
        plt.show()
    """

def deterministic_quantum_stacking_with_sum_states(y_train, bitstrings, sum_states, classifier):

    #Shape: n_train,2**label_qubits
    #initial_label_qubits = np.load('initial_label_qubit_states_3.npy')
    #Prototypical sum states
    initial_label_qubits = np.array([(classifier @ s).squeeze().data.reshape(-1) for s in sum_states])

    #Shape: n_train, n_classes
    # No Need to project onto bitstring states, as evaluate picks highest
    # Which corresponds to bitstring state anyway
    variational_label_qubits = np.load('results/stacking/final_label_qubit_states_3.npy')

    print('Accuracy before:', evaluate_classifier_top_k_accuracy(np.load('results/stacking/initial_label_qubit_states_3.npy'), y_train, 1))
    print('Accuracy After:', evaluate_classifier_top_k_accuracy(variational_label_qubits, y_train, 1))

    dim_l = initial_label_qubits.shape[1]
    outer_ket_states = initial_label_qubits

    #copy_qubits = np.ones(dim_l) / np.sqrt(dim_l)
    #copy_qubits = np.eye(dim_l)[0]

    #.shape = n_train, dim_l**n_copies+1

    weighted_summed_states = np.zeros((len(set(y_train)), dim_l, dim_l), dtype = np.complex128)
    #Loop over all predictions
    for i in tqdm(range(len(variational_label_qubits))):
        weighted_outer_states = []

        #stack (append) different labels
        for fl in variational_label_qubits[i]:
            weighted_proto_sum_states = np.zeros((dim_lc, dim_lc), dtype = np.complex128)
            #Sum over proto sum states
            for s in initial_label_qubits:
                ket = s

            #Outer product total state to convert into matrixs
                outer = np.outer(ket, ket)
                weighted_proto_sum_states += (fl * outer)

            #Weight each element via output of NN
            weighted_outer_states.append(weighted_proto_sum_states)


        #Sum over all image predictions
        weighted_summed_states += weighted_outer_states


    #U and S.shape = dim_l, dim_l
    print('Performing SVDs!')
    USs = [svd(i)[:2] for i in tqdm(weighted_summed_states)]

    #V.shape = dim_l , dim_l
    V = np.array([i[0][:, :1] @ np.sqrt(np.diag(i[1])[:1, :1]) for i in USs]).squeeze().conj().T#, axis = 0)

    print('Performing Polar Decomposition!')
    U = polar(V)[0]

    print('Performing Contractions!')
    #Copy qubits as initial states
    outer_ket_states = np.array([np.kron(i, j) for i,j in zip(np.load('results/stacking/initial_label_qubit_states_3.npy'), np.load('results/stacking/initial_label_qubit_states_3.npy'))])

    preds_U = np.array([abs(i @ U) for i in outer_ket_states])
    preds_U = np.array([i / np.sqrt(i @ i) for i in preds_U])
    preds_V = np.array([abs(i @ V) for i in outer_ket_states])
    preds_V = np.array([i / np.sqrt(i @ i) for i in preds_V])

    print('Accuracy V:', evaluate_classifier_top_k_accuracy(preds_V, y_train, 1))
    print('Accuracy U:', evaluate_classifier_top_k_accuracy(preds_U, y_train, 1))

def deterministic_quantum_stacking_with_ortho_normal_states(y_train, bitstrings, n_copies, classifier, v_col = False):

    #Shape: n_train,2**label_qubits
    initial_label_qubits = np.load('results/stacking/initial_label_qubit_states_3.npy')

    #Shape: n_train, n_classes
    # No Need to project onto bitstring states, as evaluate picks highest
    # Which corresponds to bitstring state anyway
    variational_label_qubits = np.load('results/stacking/final_label_qubit_states_3.npy')

    print('Accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_train, 1))
    print('Accuracy After:', evaluate_classifier_top_k_accuracy(variational_label_qubits, y_train, 1))

    dim_l = initial_label_qubits.shape[1]
    outer_ket_states = initial_label_qubits

    #copy_qubits = np.ones(dim_l) / np.sqrt(dim_l)
    #copy_qubits = np.eye(dim_l)[0]

    #n_copies = 2
    dim_lc = dim_l ** (1 + n_copies)

    #.shape = n_train, dim_l**n_copies+1
    for k in range(n_copies):
        outer_ket_states = np.array([np.kron(i, j) for i,j in zip(outer_ket_states, initial_label_qubits)])

    """
    #.shape = n_train, dim_l**n_copies+1, dim_l**n_copies+1
    outer_states = np.array([np.outer(i.conj().T, i) for i in outer_ket_states])
    print('Outer label qubits shape:', outer_states.shape)

    #.shape = n_classes, dim_l**n_copies+1
    weighted_summed_states = np.zeros((len(set(y_train)), dim_lc, dim_lc), dtype = np.complex128)
    for i in tqdm(range(len(outer_states))):
        weighted_outer_states = []
        for fl in variational_label_qubits[i]:
            weighted_outer_states.append(outer_states[i] * fl)
        weighted_summed_states += weighted_outer_states
    #weighted_summed_states_inefficent = np.sum([[ outer_states[i] * fl for fl in variational_label_qubits[i]] for i in range(len(outer_states))], axis = 0)
    """

    weighted_summed_states = np.zeros((len(set(y_train)), dim_lc, dim_lc), dtype = np.complex128)
    from scipy.stats import unitary_group
    ons = unitary_group.rvs(dim_lc)
    #Loop over all predictions
    for i in tqdm(range(len(variational_label_qubits))):
        weighted_outer_states = []

        #stack (append) different labels
        for fl in variational_label_qubits[i]:

            weighted_ortho_norm_states = np.zeros((dim_lc, dim_lc), dtype = np.complex128)
            for o in ons:
                #Outer product total state to convert into matrixs
                outer = np.outer(o, o)
                weighted_ortho_norm_states += (fl * outer)

            #Weight each element via output of NN
            weighted_outer_states.append(fl * outer)

        #Sum over all image predictions
        weighted_summed_states += weighted_outer_states

    from xmps.svd_robust import svd
    from scipy.linalg import polar
    #U and S.shape = dim_l**n_copies+1, dim_l**n_copies+1
    print('Performing SVDs!')
    USs = [svd(i)[:2] for i in tqdm(weighted_summed_states)]

    if v_col:
        D = dim_l
        V = np.array([i[0][:, :D] @ np.sqrt(np.diag(i[1])[:D, :D]) for i in USs]).squeeze()
        a, b, c = V.shape
        V = np.pad(V, ((0,D-a), (0,0), (0,0))).transpose(0,2,1).reshape(D*c,b)

    #V.shape = dim_l**n_copies+1 , dim_l
    else:
        V = np.array([i[0][:, :1] @ np.sqrt(np.diag(i[1])[:1, :1]) for i in USs]).squeeze()#.conj().T


    print('Performing Polar Decomposition!')
    U = polar(V)[0]

    print('Performing Contractions!')

    preds_U = np.array([abs(U @ i) for i in outer_ket_states])
    preds_U = np.array([i / np.sqrt(i @ i) for i in preds_U])
    preds_V = np.array([abs(V @ i) for i in outer_ket_states])
    preds_V = np.array([i / np.sqrt(i @ i) for i in preds_V])

    if v_col:
        preds_U = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_U])
        preds_V = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_V])

    #print(preds_U[0,:30].shape)
    #print(preds_U[0,:30])
    #assert()

    print('Accuracy V:', evaluate_classifier_top_k_accuracy(preds_V, y_train, 1))
    print('Accuracy U:', evaluate_classifier_top_k_accuracy(preds_U, y_train, 1))
    """
    for i in range(10):
        plt.bar(range(10), preds_U[i], color = 'tab:blue', label = 'ortho')
        plt.bar(range(10), preds_V[i], fill = False, edgecolor = 'tab:orange', linewidth = 1.5, label = 'non_ortho')
        plt.legend()
        plt.show()
    """

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

def pennylane_example():
    import numpy as np
    import tensorflow as tf

    import pennylane as qml
    from pennylane import numpy as p_np

    from pennylane.templates.state_preparations import MottonenStatePreparation
    from pennylane.templates.layers import StronglyEntanglingLayers


    # Get the MNIST Data
    mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    """
    Settings
    """
    n_qubits = 6                  # Number of qubits
    num_layers = 8                # Number of layers

    is_data_reduced = True        # Data is reduced to n classes
    reduced_classes = [1,2,3,7]   # Selected (and sorted) classes

    # Number of reduced classes
    reduced_num_classes = len(reduced_classes)

    """
    Filtering Data
    """
    # All indexes
    train_index_f = (y_train == -1)
    tests_index_f = (y_test  == -1)

    if is_data_reduced:
      # Filter indexes
      for n_class in reduced_classes:
        train_index_f   |= (y_train == n_class)
        tests_index_f   |= (y_test == n_class)

      num_classes_q = reduced_num_classes

    # New databases
    X_ends_pre = x_train[train_index_f]
    Y_ends_pre = y_train[train_index_f]

    X_tests_pre = x_test[tests_index_f]
    Y_tests_pre = y_test[tests_index_f]


    if is_data_reduced:
      # Change categories to their new range.
      # E.g. {0,...,9} -> {0,...,4}
      for i, k in enumerate(reduced_classes):
        Y_ends_pre[Y_ends_pre == k] = i
        Y_tests_pre[Y_tests_pre == k] = i

    latent_dim = 2 ** n_qubits    # Selected latent dimensions

    """
    Pre-process data.
    I.e. Reduce dimensionality
    """
    class Autoencoder(tf.keras.models.Model):
      def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
          tf.keras.layers.Flatten(name = "faltten_1"),
          tf.keras.layers.Dense(196, activation='relu', name = "dense_1"),
          tf.keras.layers.Dense(64, activation='relu', name = "dense_2"),
          tf.keras.layers.Dense(latent_dim, activation='sigmoid', name = "dense_3"),
        ])
        self.decoder = tf.keras.Sequential([
          tf.keras.layers.Dense(64, activation='relu', name = "dense_4"),
          tf.keras.layers.Dense(196, activation='relu', name = "dense_5"),
          tf.keras.layers.Dense(784, activation='relu', name = "dense_6"),
          tf.keras.layers.Reshape((28, 28), name = "reshape_1")
        ])

      def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    # Prepare and compile the model
    autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(optimizer='adam', loss='mae', metrics=["accuracy"])

    # Train the model with the filtered data
    autoencoder.fit(X_ends_pre, X_ends_pre, epochs=1, shuffle=True, validation_data=(X_tests_pre, X_tests_pre))

    """
    Encode data
    """
    # Encode data with our new autoencoder
    QX_train = autoencoder.encoder(X_ends_pre).numpy()
    QX_test = autoencoder.encoder(X_tests_pre).numpy()

    # Change Y values to categorical
    QY_train = tf.keras.utils.to_categorical(Y_ends_pre, num_classes_q)
    QY_test = tf.keras.utils.to_categorical(Y_tests_pre, num_classes_q)

    """
    Define VQC
    """

    dev = qml.device("default.qubit", wires = n_qubits)

    @qml.qnode(dev, diff_method='adjoint')
    def circuit(weights, inputs=None):
      ''' Quantum QVC Circuit'''

      # Splits need to be done through the tensorflow interface
      weights_each_layer = weights#tf.split(weights, num_or_size_splits=num_layers, axis=0)

      # Input normalization
      inputs_1 = inputs / p_np.sqrt(max(p_np.sum(inputs ** 2, axis=-1), 0.001))

      for i, W in enumerate(weights):
        # Data re-uploading technique
        if i % 2 == 0:
          MottonenStatePreparation(inputs_1, wires = range(n_qubits))

        # Neural network layer
        StronglyEntanglingLayers(weights_each_layer[i], wires=range(n_qubits))

      # Measurement return
      return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (num_layers,n_qubits,3)}

    # Model
    input_m = tf.keras.layers.Input(shape=(2 ** n_qubits,), name = "input_0")
    keras_1 = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=n_qubits, name = "keras_1")(input_m)
    output = tf.keras.layers.Dense(num_classes_q, activation='softmax', name = "dense_1")(keras_1)

    # Model creation
    model = tf.keras.Model(inputs=input_m, outputs=output, name="mnist_quantum_model")

    # Model compilation
    model.compile(
      loss='categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.01) ,
      metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    # Train the model
    model.fit(QX_train, QY_train, epochs=1, batch_size=8, shuffle=True)

    results = model.evaluate(QX_test, QY_test, batch_size=16)

def quantum_stacking_with_pennylane(mps_images, labels, classifier, bitstrings, n_copies):
    import numpy as np
    import tensorflow as tf

    import pennylane as qml
    from pennylane import numpy as p_np

    from pennylane.templates.state_preparations import MottonenStatePreparation
    from pennylane.templates.layers import StronglyEntanglingLayers


    """
    Get Data
    """
    initial_label_qubits = np.load('results/stacking/initial_label_qubit_states_4.npy')#np.array([abs((mps_image.H @ classifier).squeeze().data) for mps_image in mps_images])

    training_predictions = initial_label_qubits
    n_label_qubits = int(np.log2(training_predictions.shape[1]))
    for k in range(n_copies):
        training_predictions = np.array([np.kron(i, j) for i,j in zip(training_predictions, initial_label_qubits)])

    labels = np.load('training_labels.npy')

    r_train = np.arange(len(training_predictions))
    np.random.shuffle(r_train)
    training_predictions = training_predictions[r_train]
    labels = labels[r_train]

    """
    Settings
    """
    n_qubits = int(np.log2(training_predictions.shape[1])) # Number of qubits
    num_layers = 14 # Number of layers

    #One Hot encode labels
    #And pad to make shape 16
    labels = np.array([np.pad(i ,(0,6)) for i in tf.keras.utils.to_categorical(labels, len(set(labels)))])

    """
    Define VQC
    """

    #dev = qml.device("lightning.qubit", wires = n_qubits)
    dev = qml.device("default.qubit.tf", wires = n_qubits)

    #@qml.qnode(dev, diff_method='adjoint')
    @qml.qnode(dev)
    def circuit(weights, inputs=None):
      ''' Quantum QVC Circuit'''

      # Splits need to be done through the tensorflow interface
      weights_each_layer = tf.split(weights, num_or_size_splits=num_layers, axis=0)

      # Input normalization
      inputs_1 = inputs / p_np.sqrt(max(p_np.sum(inputs ** 2, axis=-1), 0.001))

      for i, W in enumerate(weights):
        # Data re-uploading technique
        if i % 2 == 0:
          MottonenStatePreparation(inputs_1, wires = range(n_qubits))

        # Neural network layer
        StronglyEntanglingLayers(weights_each_layer[i], wires=range(n_qubits))

      # Return full state of label qubits (trace over everything else)
      return qml.probs(wires=list(range(n_label_qubits)))

    weight_shapes = {"weights": (num_layers,n_qubits,3)}

    #drawer = qml.draw(circuit)
    #print(drawer(, ))
    #print(qml.draw(circuit()))
    #rho = circuit(np.random.randn(num_layers,n_qubits,3), training_predictions[0])

    # Model
    input_m = tf.keras.layers.Input(shape=(2**n_qubits,), name = "input")
    output = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim= 2**n_label_qubits, name = "V")(input_m)
    #output = tf.keras.layers.Dense(10, activation='softmax', name = "dense_1")(keras_1)

    # Model creation
    model = tf.keras.Model(inputs=input_m, outputs=output, name="Quantum_Stacking")
    # Model compilation
    model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
      optimizer=tf.keras.optimizers.Adam(learning_rate=0.01) ,
      metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    model(training_predictions[:1])
    model.summary()
    # Train the model
    history = model.fit(training_predictions, labels, epochs=100, batch_size=8, shuffle=True)

    loss = history.history['loss']
    acc = history.history['categorical_accuracy']
    np.save('loss', loss)
    np.save('accuracy', acc)
    results = model.evaluate(training_predictions, labels, batch_size=32)



def efficent_deterministic_quantum_stacking(y_train, bitstrings, n_copies, classifier, v_col = False):

    #Shape: n_train,2**label_qubits
    initial_label_qubits = np.array(np.load('results/stacking/initial_label_qubit_states_4.npy'), dtype = np.float32)

    #Shape: n_train, n_classes
    # No Need to project onto bitstring states, as evaluate picks highest
    # Which corresponds to bitstring state anyway
    variational_label_qubits = np.array(np.load('results/stacking/final_label_qubit_states_4.npy'), dtype = np.float32)

    print('Accuracy before:', evaluate_classifier_top_k_accuracy(initial_label_qubits, y_train, 1))
    print('Accuracy after:', evaluate_classifier_top_k_accuracy(variational_label_qubits, y_train, 1))

    dim_l = initial_label_qubits.shape[1]
    outer_ket_states = initial_label_qubits

    #copy_qubits = np.ones(dim_l) / np.sqrt(dim_l)
    #copy_qubits = np.eye(dim_l)[0]

    #n_copies = 2
    dim_lc = dim_l ** (1 + n_copies)

    #.shape = n_train, dim_l**n_copies+1
    for k in range(n_copies):
        outer_ket_states = np.array([np.kron(i, j) for i,j in zip(outer_ket_states, initial_label_qubits)])

    V = []
    for L in tqdm(variational_label_qubits.T):
        weighted_outer_states = np.zeros((dim_lc, dim_lc), dtype = np.float32)
        for i, fl  in tqdm(enumerate(L)):
            ket = initial_label_qubits[i]

            for k in range(n_copies):
                ket = np.kron(ket, initial_label_qubits[i])

            outer = np.outer(fl * ket, ket)
            weighted_outer_states += outer

        print('Performing SVD!')
        U, S = svd(weighted_outer_states)[:2]

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
        preds_U = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_U])
        #preds_V = np.array([np.diag(partial_trace(np.outer(i, i.conj()), [0,1,2,3]))[:10] for i in preds_V])

    #print('Accuracy V:', evaluate_classifier_top_k_accuracy(preds_V, y_train, 1))
    print('Accuracy U:', evaluate_classifier_top_k_accuracy(preds_U, y_train, 1))


if __name__ == '__main__':
    pennylane_example()
