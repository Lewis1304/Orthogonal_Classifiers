import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from scipy.linalg import polar


model = tf.keras.models.load_model('results/deterministic_initialisation_non_ortho/my_model')
model.summary()
print(model.evaluate(np.array([np.ones(10)]),np.array([1])))
assert()
weights = model.weights[0].numpy()
bias = model.weights[1].numpy()

U = polar(weights)[0]
