# Purpose: Model components.

from typing import List

import ml_collections
import sys
import tensorflow as tf

sys.path.append("..")
import utils.model_utils as model_utils

# -----------------------------------------------------------------------------


class DenseBlock(tf.keras.layers.Layer):
    """Denses layer followed by dropout."""

    def __init__(self, units, dropout=0.2, l1=0, l2=0, 
                 trainable=True, name=None,  **kwargs):
        # Including **kwargs allows for saving.
        super(DenseBlock, self).__init__(name=name,  **kwargs)
        
        # Store arguments.
        self.units = units
        self.dropout = dropout
        self.l1 = l1
        self.l2 = l2
        self.trainable = trainable
        
        # Layers.
        self.dense_layer = tf.keras.layers.Dense(
            units=units, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2),
            trainable=trainable,
            name=name)
        self.drop_layer = tf.keras.layers.Dropout(dropout)


    def call(self, inputs):
        h = self.dense_layer(inputs)
        h = self.drop_layer(h)
        return h


    def get_config(self):
        config = super().get_config().copy()
        # These match the arguments passed to __init__.
        config.update({
            'units': self.units,
            'dropout': self.dropout,
            'l1': self.l1,
            'l2': self.l2,
            'trainable': self.trainable,
            'name': self.name,
        })
        return config
        

# -----------------------------------------------------------------------------


def _make_conv_layer(inputs: tf.Tensor, name: str):
    """Make convlution layer."""
    out = tf.keras.layers.Conv2D(
        filters=10,
        kernel_size=(3, 3),
        name=name)(inputs)
    return out


def _make_dens_layer(
    config: ml_collections.ConfigDict, inputs: tf.Tensor, name: str):
    """Construct DenseBlock with given hyperparameters."""
    out = DenseBlock(
        units=config.hparams.dens_units,
        dropout=config.hparams.dens_dropout,
        l1=config.hparams.dens_l1_penalty,
        l2=config.hparams.dens_l2_penalty,
        name=name)(inputs)
    return out


def _make_model(
    config: ml_collections.ConfigDict, 
    inputs: tf.Tensor, outputs: List[tf.Tensor]):
    """Specify and compile model."""
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=model_utils.get_losses(config.outputs),
        metrics=model_utils.get_metrics(config.outputs))
    return model

# -----------------------------------------------------------------------------


def mnist_conv_model(config: ml_collections.ConfigDict):
    """MNIST convolutional model.

    Args:
        config: Model ConfigDict.

    Returns:
        model: Compiled tf.keras.Model
    """
    inputs = model_utils.get_inputs(config.inputs)

    h = _make_conv_layer(inputs['Image'], name="Conv1")
    h = _make_conv_layer(h, name="Conv2")
    h = tf.keras.layers.MaxPooling2D(name="MaxPool")(h)
    h = tf.keras.layers.Flatten(name="Flat")(h)
    h = _make_dens_layer(config, h, name="Dense1")

    outputs = model_utils.get_outputs(h, config.outputs)
    model = _make_model(config, inputs, outputs)
    
    return model


# -----------------------------------------------------------------------------


def mnist_dens_model(config: ml_collections.ConfigDict):
    """MNIST model with single dense layer."""
    inputs = model_utils.get_inputs(config.inputs)

    h = tf.keras.layers.Flatten(name="Flat")(inputs['Image'])
    h = _make_dens_layer(config, h, name="Dense1")

    outputs = model_utils.get_outputs(h, config.outputs)

    model = _make_model(config, inputs, outputs)
    
    return model
    
