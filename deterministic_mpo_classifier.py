import numpy as np
from functools import reduce
from xmps.fMPS import fMPS
from tqdm import tqdm
from fMPO_reduced import fMPO

"""
Load Data
"""

def gather_data_mnist(n_train=10,n_test=2,size=1,shuffle=True,equal_classes = False):

    def load_mnist():
        try:
            import idx2numpy as idn
        except ImportError:
            print('idx2numpy not installed. Try pip install idx2numpy')
        data = idn.convert_from_file(open('data/MNIST/t10k-images-idx3-ubyte', 'rb'))
        labels = idn.convert_from_file(
            open('data/MNIST/t10k-labels-idx1-ubyte', 'rb'))
        return data/255, labels  # (10000, 28, 28), (10000,)

    def average_image(image, window_width, window_height=None):
        """average_image: block average an image

        :param image: image data
        :param window_width: how many horizonal pixels to average
        :param window_height: optional - if None, use window_width
        """
        window_height = window_height if window_height is not None else window_width
        width, height = image.shape
        new_width, new_height = int(width/window_width), int(height/window_height)
        new_image = np.zeros((new_width, new_height))
        for j in range(new_height):
            for i in range(new_width):
                window = image[window_width*i:window_width *
                               (i+1), window_height*j: window_height*(j+1)]
                new_image[i, j] = np.mean(window)
        return new_image

    # n_train: Number of training images per digit
    # n_test: Number of test images per digit
    data, labels = load_mnist()
    possible_labels = list(set(labels))

    if shuffle:
        randomize = np.arange(len(data))
        np.random.shuffle(randomize)
        data = data[randomize]
        labels = labels[randomize]


    data = np.array([data[labels == label][:n_train+n_test] for label in possible_labels]).reshape(-1,28,28)

    N,x,y = data.shape
    labels = np.array([[label]*(n_train+n_test) for label in possible_labels]).reshape(-1)
    #List of images
    train_data = np.array([data[labels == label][:n_train] for label in possible_labels]).reshape(-1,x,y)

    #Blockwise average
    train_data = np.array([average_image(i,size) for i in train_data])

    #Flatten as snake
    train_data = np.array([np.array([i[::(-1)**k] for k,i in enumerate(j)]).reshape(-1) for j in train_data])
    train_labels = np.array([[label]*n_train for label in possible_labels]).reshape(-1)

    test_data =  np.array([data[labels == label][n_train:n_train + n_test] for label in possible_labels]).reshape(-1,x,y)
    test_data = np.array([average_image(i,size) for i in test_data])
    test_data = np.array([np.array([i[::(-1)**k] for k,i in enumerate(j)]).reshape(-1) for j in test_data])
    test_labels = np.array([[label]*n_test for label in possible_labels]).reshape(-1)

    if equal_classes:
        sorted_images = [[train_data[i] for i in range(len(train_data)) if train_labels[i] == label] for label in possible_labels]
        sorted_labels = [[train_labels[i] for i in range(len(train_labels)) if train_labels[i] == label] for label in possible_labels]

        equalised_data = []
        equalised_labels = []

        for j in range(n_train):
            group_data = []
            group_labels = []
            for i in range(len(sorted_images)):
                group_data.append(sorted_images[i][j])
                group_labels.append(sorted_labels[i][j])

            equalised_data.append(group_data)
            equalised_labels.append(group_labels)

        equalised_data = np.array([item for sublist in equalised_data for item in sublist])
        equalised_labels = np.array([item for sublist in equalised_labels for item in sublist])

        return equalised_data, equalised_labels, test_data, test_labels

    return train_data,train_labels,test_data,test_labels

"""
Encode Data
"""

def mps_encoding(images, D=2):

    def image_to_mps(f_image, D):
        num_pixels = f_image.shape[0]
        L = int(np.ceil(np.log2(num_pixels)))
        padded_image = np.pad(f_image, (0, 2**L - num_pixels)).reshape(*[2] * L)
        return fMPS().left_from_state(padded_image).left_canonicalise(D=D)

    #Embed as MPSs
    images_mps = [image_to_mps(image, D) for image in images]

    return images_mps

def spread_ditstring(n_hairy_sites, possible_labels, n_features, truncate = True, one_site = False):

    if one_site:
        num_qubits = int(np.log2(len(possible_labels))) + 1
        hairy_sites = np.expand_dims(
            [i for i in np.eye(2 ** num_qubits)][: len(possible_labels)], 1
        )

        other_sites = np.array(
            [
                [np.eye(2 ** num_qubits)[0] for pixel in range(n_features - 1)]
                for _ in possible_labels
            ]
        )

    else:
        """
        Iterates over bitstrings in chunks. i.e i = bitstring[chunksize*i:chunksize*(i+1)]
        different basis vector for each chunk. i.e. 00 = [1,0,0,0], 01 = [0,1,0,0]
        """
        bitstrings = [bin(label)[2:].zfill(n_hairy_sites*2) for label in possible_labels]

        hairy_sites = np.array([[[1,0,0,0]*(1-int(bitstring[i:i + 2][0]))*(1-int(bitstring[i:i + 2][1])) +
              [0,1,0,0]*(1-int(bitstring[i:i + 2][0]))*(int(bitstring[i:i + 2][1])) +
              [0,0,1,0]*(int(bitstring[i:i + 2][0]))*(1-int(bitstring[i:i + 2][1])) +
              [0,0,0,1]*(int(bitstring[i:i + 2][0]))*(int(bitstring[i:i + 2][1]))
              for i in range(0, len(bitstring), 2)]
              for bitstring in bitstrings])

        other_sites = np.array([[[1,0,0,0] for pixel in range(n_features - n_hairy_sites)] for label in possible_labels])

    spread_ditstrings = np.append(other_sites,hairy_sites,axis=1)


    if truncate:
        spread_ditstrings = [[ spread_ditstrings[label][pixel][:1] if pixel < (n_features-n_hairy_sites) else spread_ditstrings[label][pixel][:] for pixel in range(n_features)] for label in possible_labels]
        return spread_ditstrings

    return spread_ditstrings

def mpo_data_to_mpo(possible_labels,mpo_data):
    return {label: [fMPO(mpo_prod_image) for mpo_prod_image in mpo_data[label]] for label in possible_labels}

def mpo_encoded_data(train_data, train_labels, D, one_site = True):
    possible_labels = list(set(train_labels))
    n_train = int(len(train_data)/len(possible_labels))
    n_features = int(np.log2(len(train_data[0]))) + 1

    if not one_site:
        n_hairy_sites = int((int(np.log2(len(possible_labels)))+1)/2)
    else:
        n_hairy_sites = 1

    train_mps = [mps_encoding(train_data[train_labels == label], D=D) for label in possible_labels]

    b = spread_ditstring(n_hairy_sites, possible_labels, n_features, truncate=False, one_site = one_site)

    mpo_data_train = [[[np.array([train_mps[label][image][pixel] * i for i in j]).transpose(1,0,2,3)
            for pixel,j in enumerate(b[label])]
            for image in range(n_train)]
            for label in possible_labels]

    mpo_data_train_trunc = [[[ mpo_data_train[label][image][pixel][:,:1,:,:] if pixel < (n_features-n_hairy_sites) else mpo_data_train[label][image][pixel][:,:,:,:] for pixel in range(n_features)]  for image in range(n_train)] for label in possible_labels]

    MPOs_train = mpo_data_to_mpo(possible_labels, mpo_data_train_trunc)
    return MPOs_train

"""
Adding Images
"""

def add_sublist(*args):
    """
    :param args: tuple of B_D and MPOs to be added together
    """

    B_D = args[0][0]
    ortho = args[0][1]
    sub_list_mpos = args[1]
    N = len(sub_list_mpos)


    c = sub_list_mpos[0]


    for i in range(1,N):
        c = c.add(sub_list_mpos[i])

    if c.data[-2].shape[1] == 1:
        return c.compress_one_site(B_D, orthogonalise=ortho)
    return c.compress(B_D, orthogonalise=ortho)

def adding_batches(list,D,batch_num=2,truncate=True, orthogonalise = False):
    # if batches are not of equal size, the remainder is added
    # at the end- this is a MAJOR problem with equal weightings!

    if len(list) % batch_num != 0:
        if not truncate:
            raise Exception('Batches are not of equal size!')
        else:
            trun_expo = int(np.log(len(list)) / np.log(batch_num))
            list = list[:batch_num**trun_expo]
    result = []

    for i in range(int(len(list)/batch_num)+1):
        sub_list = list[batch_num*i:batch_num*i+batch_num]
        if len(sub_list) > 0:
            result.append(reduce(add_sublist,((D, orthogonalise),sub_list)))
    return result

"""
Evaluate classifier
"""

def classify(classifiers, test_data, test_labels, D, one_site = False):
    possible_labels = list(set(test_labels))
    n_features = int(np.log2(len(test_data[0]))) + 1

    if not one_site:
        n_hairy_sites = int((int(np.log2(len(possible_labels)))+1)/2)
    else:
        n_hairy_sites = 1

    def mpo_overlap(classifiers, bitstring, test_datum, n_train = 10):
        return np.abs(classifiers.apply_mpo_from_bottom(bitstring).overlap(test_datum))

    labels = np.array(possible_labels)

    list_spread_ditstrings = [fMPS().from_product_state(i) for i in spread_ditstring(n_hairy_sites, possible_labels, n_features, one_site = one_site)]
    list_test_MPSs = mps_encoding(test_data, D=D)

    return labels.take(np.argmax(np.array([[mpo_overlap(classifiers,i,j) for i in list_spread_ditstrings] for j in (list_test_MPSs)]), axis = 1))

def evaluate_classifiers(classifiers, test_data, test_labels,  D=2, one_site = False):

    def success_fraction(ar1, ar2):
        assert len(ar1) == len(ar2)
        average_result = np.sum((ar1 == ar2).astype(int))/len(ar1)
        return average_result

    out_labels = classify(classifiers,test_data, test_labels, D, one_site)
    sf = success_fraction(test_labels,out_labels)
    return sf

"""
Experiment
"""

def sequential_mpo_classifier_experiment():
    n_samples = 10
    possible_labels = list(range(10))
    D_total = 32
    batch_num = 2
    one_site = False

    train_data, train_labels, test_data, test_labels = gather_data_mnist(n_samples, 1, 1, shuffle=True, equal_classes=True)

    train_spread_mpo_product_states = mpo_encoded_data(train_data, train_labels, D_total, one_site = one_site)
    train_spread_mpo_product_states = [image for label in train_spread_mpo_product_states.values() for image in label]

    MPOs = train_spread_mpo_product_states
    while len(MPOs) > 1:
        MPOs = adding_batches(MPOs, D_total, batch_num, orthogonalise=False)

    #one_site = True
    #test1 = fMPO(MPOs[0].data)
    #test2 = fMPO(MPOs[0].data)
    for ortho_at_end in [False, True]:
        if one_site:
            classifier = MPOs[0].compress_one_site(D=D_total, orthogonalise=ortho_at_end)
        else:
            classifier = MPOs[0].compress(D=D_total, orthogonalise=ortho_at_end)

        result = evaluate_classifiers(classifier, train_data, train_labels, D=D_total, one_site = one_site)
        #result2 = evaluate_classifiers(classifier2, train_data, train_labels, D=D_total, one_site = False)
        #print(result1)
        #print(result2)
        print(result)
        
    return classifier, result

if __name__ == '__main__':
    sequential_mpo_classifier_experiment()
