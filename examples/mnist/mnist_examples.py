#!/usr/bin/env python3

import pandas as pd
import subprocess as sp
import tempfile
import tensorflow as tf

# Set local directory?
set_dir = False
if set_dir:
    import os
    os.chdir('/Users/zmccaw/Documents/Tensorflow/examples/mnist')

import custom_callbacks
import models
import load_data
import mnist_utils

# ----------------------------------------------------------------------------
# Tensorboard
# ----------------------------------------------------------------------------

# Use tensorboard?
use_tensorboard = False

# Load data.
data = load_data.load_mnist_data()

# Tensorboard.
if use_tensorboard: 
    log_dir = tempfile.mkdtemp()
    callbacks = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1)
else:
    callbacks = []

# Model.
model = models.mnist_conv_model()
model.summary()
history = model.fit(
    x=data['x_train'],
    y=data['y_train'],
    batch_size=256,
    epochs=10,
    validation_data=(data['x_val'], data['y_val']),
    callbacks=callbacks,
    verbose=2)

# History contents.
type(history.history)
history.history.keys()

# Launch tensorboard.
if use_tensorboard:
    sp.run(["tensorboard", "--logdir", log_dir])

# History data.
df = pd.DataFrame(history.history)
df.head()

# Evaluation.
test_loss, test_acc = model.evaluate(
    x=data['x_test'],
    y=data['y_test'],
    verbose=2)

# Prediction.
mnist_utils.plot_random_prediction(x=data['x_test'], model=model)


# ----------------------------------------------------------------------------
# Early stopping.
# ----------------------------------------------------------------------------


# Model.
model = models.mnist_dens_model()
model.summary()

# Early stopping callback.
callbacks = tf.keras.callbacks.EarlyStopping(
    monitor='val_categorical_accuracy',
    min_delta=0.001,
    patience=2)

# Training.
history = model.fit(
    x=data['x_train'],
    y=data['y_train'],
    batch_size=256,
    epochs=10,
    validation_data=(data['x_val'], data['y_val']),
    callbacks=callbacks,
    verbose=2)


# Evaluation.
test_loss, test_acc = model.evaluate(
    x=data['x_test'],
    y=data['y_test'],
    verbose=2)

# Prediction.
mnist_utils.plot_random_prediction(x=data['x_test'], model=model)


# ----------------------------------------------------------------------------
# Custom callbacks.
# ----------------------------------------------------------------------------

# Model.
model = models.mnist_dens_model()
model.summary()

# Reporting callbacks.
callbacks = [
    #custom_callbacks.TrainCallbacks(),
    custom_callbacks.ValLossCallback()
  ]

# Training.
history = model.fit(
    x=data['x_train'],
    y=data['y_train'],
    batch_size=512,
    epochs=10,
    validation_data=(data['x_val'], data['y_val']),
    callbacks=callbacks,
    verbose=0)

# Learning rate schedule.
def rate_schedule(epoch, lr):
    discount = 0.99
    out = float(lr * pow(discount, epoch))
    return out

# Reporting callbacks.
callbacks = [
    custom_callbacks.CustomLearningRateScheduler(rate_schedule)
  ]

# Training.
history = model.fit(
    x=data['x_train'],
    y=data['y_train'],
    batch_size=512,
    epochs=10,
    validation_data=(data['x_val'], data['y_val']),
    callbacks=callbacks,
    verbose=0)
