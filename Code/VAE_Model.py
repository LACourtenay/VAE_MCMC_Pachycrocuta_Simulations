
# load libraries

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Flatten, Dense, Activation
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Lambda
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
import sklearn
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

print("Libraries\n------------------\n")
print(f"Tensorflow: {tf.__version__}")
print(f"Scikit-Learn: {sklearn.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Numpy: {np.__version__}")

# define cyclic learning rate class

class CyclicLR(Callback):
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
        
# define the swish activation function

def swish(x, beta = 1):
    return(x * K.sigmoid(beta * x))
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
get_custom_objects().update({"swish":Activation(swish)})

# load flattened landmark data that was prepared in R

landmarks_data = np.loadtxt("./data/fn3_reference_landmarks.txt", delimiter = ",")

with open('./data/fn3_reference_labels.txt', 'r') as f:
    labels_data = [line.strip() for line in f.readlines()]

landmarks_array = np.array(landmarks_data)

n_individuals = 823
n_points = 25

# reshape the flattened landmarks to their original tensor shape

reshaped_landmarks = landmarks_array.reshape((n_individuals, n_points, 3))

X = reshaped_landmarks
y = np.array(labels_data)
X = X.reshape(-1, n_points, 3, 1)

# define train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

val_size = np.round(X_train.shape[0] * 0.2)
print(f"Train set: {X_train.shape[0] - val_size:.0f} individuals ({((X_train.shape[0] - val_size) / X.shape[0]) * 100:.0f}%)")
print(f"Test set: {X_test.shape[0]} individuals ({(X_test.shape[0] / X.shape[0]) * 100:.0f}%)")
print(f"Validation set: {val_size:.0f} individuals ({val_size / X.shape[0] * 100:.0f}%)")

# define VAE architecture

latent_dims = 15

x = Input(shape = (n_points, 3, 1))
h = Conv2D(64, (3, 3), (1, 3), activation = "swish", padding = "same",
         kernel_regularizer = l2(l = 0.0001),
         kernel_initializer = "random_normal") (x)
h = Flatten() (h)

z_mean = Dense(latent_dims) (h)
z_log_var = Dense(latent_dims) (h)

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape = (batch, dim), mean = 0.0, stddev = 0.01)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape = (latent_dims, )) ([z_mean, z_log_var])

h_decoder = Dense(n_points * 3 * 64, activation = "swish",
         kernel_regularizer = l2(l = 0.0001),
         kernel_initializer = "random_normal") (z)
h_decoder = Reshape((n_points, 3, 64)) (h_decoder)
x_bar = Conv2DTranspose(1, (3, 3), (1, 1), activation = "linear", padding = "same") (h_decoder)

vae = Model(x, x_bar)

# define loss functions

reconstruction_loss = tf.sqrt(tf.reduce_mean(tf.square(K.flatten(x) - K.flatten(x_bar))))
kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis = 1)
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)

vae.compile(optimizer = "adam")

clr = CyclicLR(base_lr = 0.0001,
              max_lr = 0.01,
              step_size = 16,
              mode = "triangular2")

# fit the model

history = vae.fit(
    X_train, X_train, shuffle = True,
    epochs = 100, batch_size = 32,
    validation_split = 0.2, # define validation set
    callbacks = [clr]
)

# plot loss curves

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# calculate test loss

x_prima = vae.predict(X_test, verbose = False)
error = np.square(x_prima.reshape(X_test.shape[0], n_points * 3) - X_test.reshape(X_test.shape[0], n_points * 3))
np.sqrt(np.mean(error))

# extract weights of different components of the algorithm

input_generator = Input(shape = (latent_dims, ))
h_generator = Dense(n_points * 3 * 64, activation = "swish",
         kernel_regularizer = l2(l = 0.0001),
         kernel_initializer = "random_normal") (input_generator)
h_generator = Reshape((n_points, 3, 64)) (h_generator)
x_bar = Conv2DTranspose(1, (3, 3), (1, 1), activation = "linear", padding = "same") (h_generator)
generator = Model(input_generator, x_bar)
generator.set_weights(vae.get_weights()[6:10])
encoder = Model(x, z_mean)

central_tendency_encoder = Model(x, z_mean)
deviation_encoder = Model(x, z_log_var)

# save the model weights

os.makedirs("./new_trained_weights", exist_ok = True)

central_tendency_encoder.save_weights("./new_trained_weights/central_tendency_encoder.h5")
deviation_encoder.save_weights("./new_trained_weights/deviation_encoder.h5")
vae.save_weights("./new_trained_weights/vae.h5")
generator.save_weights("./new_trained_weights/generator.h5")
encoder.save_weights("./new_trained_weights/encoder.h5")