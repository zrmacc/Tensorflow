#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import tempfile
import tensorflow as tf
import tensorflow.keras as keras

import utils

# ----------------------------------------------------------------------------

def load_mnist_data():
    """Loads and formats MNIST data.
    
    Returns:
        Dictionary containing x_train, y_train, x_val, y_val, x_test, y_test.
    """
    
    # Load data.
    mnist_data = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist_data.load_data()
    
    # Expand dimensions.
    x_train = np.expand_dims(x_train, axis=3)
    x_test  = np.expand_dims(x_test , axis=3)
    
    # Convert y from 0-9 label to categorical.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test  = tf.keras.utils.to_categorical(y_test , num_classes=10)
    
    # Normalize.
    x_train = x_train / 255.0
    x_test  = x_test  / 255.0
    
    # Partition training data into training and evaluation.
    data = utils.two_way_split(x_train, y_train)
    data['x_test'] = x_test
    data['y_test'] = y_test  
    
    return data

# ----------------------------------------------------------------------------    

class DenseBlock(keras.layers.Layer):
  """Denses layer followed by dropout."""

  def __init__(self, nodes, dropout=0.2):
    super(DenseBlock, self).__init__()
    self.dense = keras.layers.Dense(nodes, activation='relu')
    self.drop = keras.layers.Dropout(dropout)
  
  def call(self, input):
    h = self.dense(input)
    h = self.drop(h)
    return h


# ----------------------------------------------------------------------------

def compile_model():
    """Simple convolutional model for classifying MNIST images."""
    
    # Model layers.
    inputs = keras.layers.Input(shape=(28, 28, 1), name="Input")
    conv1 = keras.layers.Conv2D(
        filters=10, 
        kernel_size=(3, 3),
        name="Conv1")(inputs)
    conv2 = keras.layers.Conv2D(
        filters=10,
        kernel_size=(3, 3),
        activation='relu',
        name="Conv2")(conv1)
    maxp = keras.layers.MaxPooling2D(name="MaxPool")(conv2)
    flat = keras.layers.Flatten(name="Flat")(maxp)
    dens1 = DenseBlock(nodes=10)(flat)
    outputs = keras.layers.Dense(
        units=10, 
        activation='softmax',
        name="Outputs")(dens1)
    
    # Specify model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model.
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()]
    )
    
    return model

# ----------------------------------------------------------------------------

def plot_random_prediction(x, model):
    """Plot random image with predicted label."""
    n = x.shape[0]
    idx = np.random.choice(n)
    probs = model.predict(np.expand_dims(x[idx,:], axis=0))[0,:]
    yhat = np.argmax(probs)
    yprob = probs[yhat]
    
    plt.imshow(x[idx,:,:,0] * 255.0, cmap='gray')
    plt.title(f'Prediction: {yhat}. Probability: {yprob:.2f}')
    plt.show()

    return None


# ----------------------------------------------------------------------------

# Load data.
data = load_mnist_data()

# Tensorboard.
log_dir = tempfile.mkdtemp()
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1)

# Model.
model = compile_model()
model.summary()
history = model.fit(
    x=data['x_train'],
    y=data['y_train'],
    batch_size=256,
    epochs=10,
    validation_data=(data['x_val'], data['y_val']),
    callbacks=[tensorboard_callback])

# Launch tensorboard.
sp.run(["tensorboard", "--logdir", log_dir])

# History data.
# df = pd.DataFrame(history.history)
# df.head()

# Evaluation.
test_loss, test_acc = model.evaluate(
    x = data['x_test'],
    y = data['y_test'],
    verbose = 2)

# Prediction.
plot_random_prediction(x=data['x_test'], model=model)
