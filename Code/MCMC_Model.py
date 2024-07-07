
# we have to redefine the architecture and load the trained weights from before

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Flatten, Dense, Activation
from tensorflow.keras.layers import Reshape, Conv2DTranspose, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
import sklearn
import matplotlib
import numpy as np

print("Libraries\n------------------\n")
print(f"Tensorflow: {tf.__version__}")
print(f"Scikit-Learn: {sklearn.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Numpy: {np.__version__}")

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
        
        
def swish(x, beta = 1):
    return(x * K.sigmoid(beta * x))
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
get_custom_objects().update({"swish":Activation(swish)})

latent_dims = 15
n_individuals = 823
n_points = 25

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

reconstruction_loss = tf.sqrt(tf.reduce_mean(tf.square(K.flatten(x) - K.flatten(x_bar))))
kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis = 1)
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)

vae.compile(optimizer = "adam")

clr = CyclicLR(base_lr = 0.0001,
              max_lr = 0.01,
              step_size = 16,
              mode = "triangular2")

custom_objects = {'swish': Activation('swish')}

from tensorflow.keras.layers import Activation
get_custom_objects().update({"swish": Activation(swish)})

input_generator = Input(shape = (latent_dims, ))
h_generator = Dense(n_points * 3 * 64, activation = "swish",
         kernel_regularizer = l2(l = 0.0001),
         kernel_initializer = "random_normal") (input_generator)
h_generator = Reshape((n_points, 3, 64)) (h_generator)
x_bar = Conv2DTranspose(1, (3, 3), (1, 1), activation = "linear", padding = "same") (h_generator)
generator = Model(input_generator, x_bar)
generator.set_weights(vae.get_weights()[6:10])

central_tendency_encoder = Model(x, z_mean)
deviation_encoder = Model(x, z_log_var)

encoder = Model(x, z_mean)

# LOAD WEIGHTS

central_tendency_encoder.load_weights("/new_trained_weights/central_tendency_encoder.h5")
deviation_encoder.load_weights("/new_trained_weights/deviation_encoder.h5")
vae.load_weights("/new_trained_weights/vae.h5")
generator.load_weights("/new_trained_weights/generator.h5")
encoder.load_weights("/new_trained_weights/encoder.h5")

# MCMC ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# define the sampling function

def sample_distribution(mu_vector, sd_vector, n_samples=10000, step_size=0.001):
    
    starting_point = np.array([np.random.normal(mu, sd) for mu, sd in zip(mu_vector, sd_vector)])

    latent_samples = np.empty((0, len(mu_vector)))

    current_step = starting_point
    latent_samples = np.vstack((latent_samples, current_step))

    for _ in range(n_samples):
        pdf_current = np.prod(np.exp(-0.5 * ((current_step - mu_vector) / sd_vector) ** 2) / (sd_vector * np.sqrt(2 * np.pi)))

        perturbation = np.random.normal(0, step_size, len(mu_vector))
        proposed_step = current_step + perturbation

        pdf_proposed = np.prod(np.exp(-0.5 * ((proposed_step - mu_vector) / sd_vector) ** 2) / (sd_vector * np.sqrt(2 * np.pi)))

        acceptance_ratio = pdf_proposed / pdf_current

        if np.random.uniform() < acceptance_ratio:
            current_step = proposed_step

        latent_samples = np.vstack((latent_samples, current_step))

    latent_samples = np.unique(latent_samples, axis=0)

    return latent_samples

# load the coordiante values for the pachycrocuta individuals

pachycrocuta = np.loadtxt("./data/fn3_pachycrocuta.txt", delimiter = ",")
pachycrocuta = pachycrocuta.reshape(pachycrocuta.shape[0], 25, 3, 1)

# define the properties of the latent dimension for the pachycrocuta individual

latent_mean = central_tendency_encoder(pachycrocuta).numpy()
latent_sd = deviation_encoder(pachycrocuta).numpy()
latent_sd = np.sqrt(np.exp(latent_sd)) * 0.01 # scale to the gaussian distribution from reparametrization trick

# simulate pachycrocuta pits based on the first pit from FN3

latent_individuals = sample_distribution(latent_mean[0], np.abs(latent_sd[0]), step_size = 0.005)
latent_individuals = latent_individuals[np.random.choice(latent_individuals.shape[0], size = 1000, replace=False)]
pred_individuals = generator.predict(latent_individuals, verbose = False)
pred_individuals = pred_individuals.reshape(latent_individuals.shape[0], n_points * 3)
np.savetxt("./data/reconstructed_pachycrocuta1.txt", pred_individuals, delimiter=",", fmt='%f')

# simulate pachycrocuta pits based on the second pit from FN3

latent_individuals = sample_distribution(latent_mean[1], np.abs(latent_sd[1]), step_size = 0.005)
latent_individuals = latent_individuals[np.random.choice(latent_individuals.shape[0], size = 1000, replace=False)]
pred_individuals = generator.predict(latent_individuals, verbose = False)
pred_individuals = pred_individuals.reshape(latent_individuals.shape[0], n_points * 3)
np.savetxt("./data/reconstructed_pachycrocuta2.txt", pred_individuals, delimiter=",", fmt='%f')

# simulate pachycrocuta pits based on the third pit from FN3

latent_individuals = sample_distribution(latent_mean[2], np.abs(latent_sd[2]), step_size = 0.005)
latent_individuals = latent_individuals[np.random.choice(latent_individuals.shape[0], size = 1000, replace=False)]
pred_individuals = generator.predict(latent_individuals, verbose = False)
pred_individuals = pred_individuals.reshape(latent_individuals.shape[0], n_points * 3)
np.savetxt("./data/reconstructed_pachycrocuta3.txt", pred_individuals, delimiter=",", fmt='%f')

# simulate pachycrocuta pits based on the fourth pit from FN3

latent_individuals = sample_distribution(latent_mean[3], np.abs(latent_sd[3]), step_size = 0.005)
latent_individuals = latent_individuals[np.random.choice(latent_individuals.shape[0], size = 1000, replace=False)]
pred_individuals = generator.predict(latent_individuals, verbose = False)
pred_individuals = pred_individuals.reshape(latent_individuals.shape[0], n_points * 3)
np.savetxt("./data/reconstructed_pachycrocuta4.txt", pred_individuals, delimiter=",", fmt='%f')