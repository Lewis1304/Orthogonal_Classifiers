import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from variational_mpo_classifiers import evaluate_classifier_top_k_accuracy
import matplotlib.pyplot as plt

tf.random.set_seed(42)


def load_predictions(dataset):
    from functools import reduce
    training_predictions = np.load('data/' + dataset + '/new_ortho_d_final_vs_training_predictions.npy')[15]
    training_predictions = np.array([i / np.sqrt(i.T @ i) for i in training_predictions])
    y_train = np.load('data/' + dataset + '/ortho_d_final_vs_training_predictions_labels.npy')

    #training_predictions = np.array(reduce(list.__add__, [list(training_predictions[i*5421 : i * 5421 + 10]) for i in range(10)]))
    #y_train = np.array(reduce(list.__add__, [list(y_train[i*5421 : i * 5421 + 10]) for i in range(10)]))

    return training_predictions, y_train

def plot_predictions(dataset):
    import pandas as pd
    from pandas.plotting import scatter_matrix
    from functools import reduce

    x,y = load_predictions(dataset)
    reduced_x = np.array(reduce(list.__add__, [list(x[i*5421 : i * 5421 + 100]) for i in range(10)]))
    reduced_y = np.array(reduce(list.__add__, [list(y[i*5421 : i * 5421 + 100]) for i in range(10)]))

    data = pd.DataFrame(reduced_x[:,:10])
    scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='hist')
    plt.savefig('predictions_scatter_plot.pdf')

def nonlinear_predictions(dataset, plot = False):

    if plot:
        train_acc = np.load('non_linear_training_accuracy.npy')
        test_acc = np.load('non_linear_test_accuracy.npy')

        plt.plot(train_acc)
        plt.plot(test_acc)
        plt.axhline(1, color = 'r', alpha = 0.4, label = f'Max training accuracy: {1}')
        plt.axhline(max(test_acc), color = 'b', alpha = 0.4, label = f'Max test accuracy: {np.round(max(test_acc),5)}')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('MNIST')
        plt.savefig('nonlinear_performance.pdf')
        plt.show()
        assert()


    training_predictions, y_train = load_predictions(dataset)
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    #x_train = x_train / 255.
    #x_test = x_test / 255.
    inputs = tf.keras.Input(shape=(training_predictions.shape[1],))
    #inputs = tf.keras.Input(shape=(28,28,1))
    x = tf.keras.layers.Flatten()(inputs)
    #inputs = tf.keras.Input(shape=(qtn_prediction_and_ancillae_qubits.shape[1],))
    x = tf.keras.layers.Dense(1000, activation = 'sigmoid')(inputs)
    #x = tf.keras.layers.Dense(1000, activation = 'sigmoid')(x)
    outputs = tf.keras.layers.Dense(10, activation = None)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()


    model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


    test_preds = np.load('data/' + dataset + '/new_ortho_d_final_vs_test_predictions.npy')[15]
    y_test = np.load('data/' + dataset + '/ortho_d_final_vs_test_predictions_labels.npy')

    history = model.fit(
        training_predictions,
        #x_train,
        y_train,
        #epochs=200,
        epochs=3000,
        batch_size = 32,
        verbose = 1,
        validation_data = (test_preds, y_test),
        #validation_data = (x_test, y_test),
    )

    #np.save('non_linear_training_accuracy_on_images', history.history['sparse_categorical_accuracy'])
    #np.save('non_linear_test_accuracy_on_images', history.history['val_sparse_categorical_accuracy'])

    #model.save('models/ortho_big_dataset_D_32')

    trained_test_predictions = model.predict(test_preds)
    #np.save('final_label_qubit_states_4',trained_training_predictions)

    #np.save('trained_predicitions_1000_classifier_32_1000_train_images', trained_training_predictions)
    accuracy = evaluate_classifier_top_k_accuracy(trained_test_predictions, y_test, 1)
    print(accuracy)

def svms(dataset):
    from sklearn.svm import SVC, LinearSVC
    from sklearn.metrics import classification_report

    x,y = load_predictions('fashion_mnist')
    print('Initial training accuracy: ', evaluate_classifier_top_k_accuracy(x,y,1))

    x_test = np.load('data/' + dataset + '/new_ortho_d_final_vs_test_predictions.npy')[15]
    y_test = np.load('data/' + dataset + '/ortho_d_final_vs_test_predictions_labels.npy')
    print('Initial test accuracy: ', evaluate_classifier_top_k_accuracy(x_test,y_test,1))
    print()

    linear_classifier = SVC(kernel = 'linear', verbose = 0)
    linear_classifier.fit(x,y)
    linear_preds = linear_classifier.predict(x)
    linear_test_preds = linear_classifier.predict(x_test)
    print('Linear svm accuracy: ', classification_report(linear_preds, y, output_dict = True)['accuracy'])
    print('Linear test svm accuracy: ', classification_report(linear_test_preds, y_test, output_dict = True)['accuracy'])
    print()

    another_linear_classifier = LinearSVC(max_iter=10000)
    another_linear_classifier.fit(x,y)
    another_linear_preds = another_linear_classifier.predict(x)
    another_linear_test_preds = another_linear_classifier.predict(x_test)
    print('Another linear svm accuracy: ', classification_report(another_linear_preds, y, output_dict = True)['accuracy'])
    print('Another linear test svm accuracy: ', classification_report(another_linear_test_preds, y_test, output_dict = True)['accuracy'])
    print()

    train_results = []
    test_results = []
    for i in np.linspace(0.1,1,10):
        gaussian_linear_classifier = SVC(kernel = 'rbf', gamma = i, verbose = 0)
        gaussian_linear_classifier.fit(x,y)
        gaussian_linear_preds = gaussian_linear_classifier.predict(x)
        gaussian_linear_test_preds = gaussian_linear_classifier.predict(x_test)
        train_results.append(classification_report(gaussian_linear_preds, y, output_dict = True)['accuracy'])
        test_results.append(classification_report(gaussian_linear_test_preds, y_test, output_dict = True)['accuracy'])
    print('Gaussian svm accuracy: ', max(train_results))
    print('Gaussian test svm accuracy: ', max(test_results))
    print()

    train_results = []
    test_results = []
    for i in np.linspace(0.1,1,10):
        for j in range(1,11):
            poly_linear_classifier = SVC(kernel = 'poly', degree = j, gamma = i, verbose = 0)
            poly_linear_classifier.fit(x,y)
            poly_linear_preds = poly_linear_classifier.predict(x)
            poly_linear_test_preds = poly_linear_classifier.predict(x_test)
            train_results.append(classification_report(poly_linear_preds, y, output_dict = True)['accuracy'])
            test_results.append(classification_report(poly_linear_test_preds, y_test, output_dict = True)['accuracy'])
    print('Poly svm accuracy: ', max(train_results))
    print('Poly test svm accuracy: ', max(test_results))
    print()

    train_results = []
    test_results = []
    for i in np.linspace(0.1,1,10):
        sigmoid_linear_classifier = SVC(kernel = 'sigmoid', gamma = i, verbose = 0)
        sigmoid_linear_classifier.fit(x,y)
        sigmoid_linear_preds = sigmoid_linear_classifier.predict(x)
        sigmoid_linear_test_preds = sigmoid_linear_classifier.predict(x_test)
        train_results.append(classification_report(sigmoid_linear_preds, y, output_dict = True)['accuracy'])
        test_results.append(classification_report(sigmoid_linear_test_preds, y_test, output_dict = True)['accuracy'])
    print('Sigmoid svm accuracy: ', max(train_results))
    print('Sigmoid test svm accuracy: ', max(test_results))
    print()

def gaussian_svm(dataset):
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    from tqdm import tqdm

    x,y = load_predictions(dataset)
    #(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #x = x.reshape(x.shape[0], -1)
    #x_test = x_test.reshape(x_test.shape[0], -1)
    print(f'DATASET: {dataset}')
    print()
    print('Initial training accuracy: ', evaluate_classifier_top_k_accuracy(x,y,1))

    x_test = np.load('data/' + dataset + '/new_ortho_d_final_vs_test_predictions.npy')[15]
    y_test = np.load('data/' + dataset + '/ortho_d_final_vs_test_predictions_labels.npy')
    print('Initial test accuracy: ', evaluate_classifier_top_k_accuracy(x_test,y_test,1))
    print()
    train_results = []
    test_results = []
    for i in tqdm(np.linspace(0.001,1,10)):
    #for i in tqdm(['scale', 'auto']):
        gaussian_linear_classifier = SVC(kernel = 'rbf', gamma = i, verbose = 0, C = 1)
        gaussian_linear_classifier.fit(x,y)
        gaussian_linear_preds = gaussian_linear_classifier.predict(x)
        gaussian_linear_test_preds = gaussian_linear_classifier.predict(x_test)

        #print('Gaussian svm accuracy: ', classification_report(gaussian_linear_preds, y, output_dict = True)['accuracy'])
        #print('Gaussian test svm accuracy: ', classification_report(gaussian_linear_test_preds, y_test, output_dict = True)['accuracy'])
        #print()
        train_results.append(classification_report(gaussian_linear_preds, y, output_dict = True)['accuracy'])
        test_results.append(classification_report(gaussian_linear_test_preds, y_test, output_dict = True)['accuracy'])
    print('Gaussian svm accuracy: ', max(train_results))
    print('Gaussian test svm accuracy: ', max(test_results))
    #print()

def fitting_stacking():
    from scipy.optimize import curve_fit

    #test_results = [100-80.33,100-85.65,100-87.03]
    test_results = [100-80.33,100-87.63,100-89.80]
    x = range(1,4)


    #def func(x, a, b, c):
    #    return a * np.log(b * x) + c

    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, pcov = curve_fit(func, x, test_results)

    plt.plot(x, test_results, 'b-', label='data', linewidth = 0, markersize = 12, marker = '.')
    plt.plot(range(1,10), func(range(1,10), *popt), 'g--', label='fit: ''$a * e^{-b * x} + c$')
    plt.axhline(100-90.46, color = 'r', alpha = 0.4, label = '"Best" linear classifier')
    plt.axhline(100 - 91.07, color = 'orange', alpha = 0.4, label = '"Best" non-linear classifier')
    plt.legend()
    plt.ylabel('Test Error')
    plt.xlabel('Number of Copies')
    plt.savefig('fitting_curve.pdf')
    plt.show()

if __name__ == '__main__':
    fitting_stacking()
    assert()
    nonlinear_predictions('fashion_mnist')
    #gaussian_svm('fashion_mnist')
    #svms('fashion_mnist')
