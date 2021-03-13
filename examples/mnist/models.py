# Purpose: Model components.

import tensorflow as tf

# ----------------------------------------------------------------------------


class DenseBlock(tf.keras.layers.Layer):
  """Denses layer followed by dropout."""

  def __init__(self, units, dropout=0.2, l1=0, l2=0, name=None):
    super(DenseBlock, self).__init__(name=name)
    self.dense = tf.keras.layers.Dense(
        units=units, 
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2),
        name=name)
    self.drop = tf.keras.layers.Dropout(dropout)
  
  def call(self, inputs):
    h = self.dense(inputs)
    h = self.drop(h)
    return h


# ----------------------------------------------------------------------------


def _mnist_compile(model: tf.keras.Model) -> tf.keras.Model:
    """Compiles MNIST model."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    return model


# ----------------------------------------------------------------------------


def mnist_conv_model():
    """MNIST convolutional model."""
    
    # Model layers.
    inputs = tf.keras.layers.Input(shape=(28, 28, 1), name="Input")
    conv1 = tf.keras.layers.Conv2D(
        filters=10, 
        kernel_size=(3, 3),
        name="Conv1")(inputs)
    conv2 = tf.keras.layers.Conv2D(
        filters=10,
        kernel_size=(3, 3),
        activation='relu',
        name="Conv2")(conv1)
    maxp = tf.keras.layers.MaxPooling2D(name="MaxPool")(conv2)
    flat = tf.keras.layers.Flatten(name="Flat")(maxp)
    dens1 = DenseBlock(units=10, dropout=0.2, name="Dens1")(flat)
    outputs = tf.keras.layers.Dense(
        units=10, 
        activation='softmax',
        name="Outputs")(dens1)
    
    # Specify model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model.
    model = _mnist_compile(model)   
    return model


# ----------------------------------------------------------------------------


def mnist_dens_model():
    """MNIST model with single dense layer."""
    
    # Model layers.
    inputs = tf.keras.layers.Input(shape=(28, 28, 1), name="Input")
    flat = tf.keras.layers.Flatten(name="Flat")(inputs)
    dens1 = DenseBlock(units=784, l1=1e-3, l2=1e-3, name="Dens1")(flat)
    outputs = tf.keras.layers.Dense(
        units=10,
        activation='softmax',
        name="Outputs")(dens1)
    
    # Specify model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model.
    model = _mnist_compile(model)
    return model
    
