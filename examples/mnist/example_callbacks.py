#!/usr/bin/env python3

import os
import pandas as pd
import sys
import tensorflow as tf

# Set local directory?
set_dir = False
if set_dir:
    os.chdir('/Users/zmccaw/Documents/Tensorflow/examples/mnist')

import configs
import load_data
import models
import mnist_utils

sys.path.append("..")
import utils.custom_callbacks as custom_callbacks
import utils.train_utils as train_utils

# Load data.
data = load_data.load_mnist_data()
print(data.keys())

# ----------------------------------------------------------------------------

def eval_model(model, data):
    """Evaluate model, report accuracy."""
    out = model.evaluate(
        x=data['x_test'], 
        y=data['y_test'], 
        return_dict=True,
        verbose=0)
    if out['categorical_accuracy']:      
        print('Accuracy: {acc:.3f}'.format(
            acc=out['categorical_accuracy']))
    return out


# ----------------------------------------------------------------------------
# CSV writer.
# ----------------------------------------------------------------------------

# Model.
config = configs.conv_model_config()
model = models.mnist_conv_model(config)
model.summary()

# Callbacks.
callbacks = [
    tf.keras.callbacks.CSVLogger(
        filename="results/conv_model_callbacks.tsv",
        separator="\t")    
]

# Training.
history = model.fit(
    x=data['x_train'],
    y=data['y_train'],
    batch_size=256,
    epochs=10,
    validation_data=(data['x_val'], data['y_val']),
    callbacks=callbacks,
    verbose=2)

# Load saved data.
train_output = pd.read_csv("results/conv_model_callbacks.tsv", sep="\t")
train_output.head()

# ----------------------------------------------------------------------------
# Model saving.
# ----------------------------------------------------------------------------

# Model.
config = configs.conv_model_config()
model = models.mnist_conv_model(config)
model.summary()

# Model checkpointing.
check_dir = 'checkpoints'
prefix = 'conv_model'
train_utils.clear_checkpoint_dir(check_dir, prefix)

# Set save_weights_only=False to save the entire model.
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            check_dir,
            prefix + '.epoch_{epoch}.val_acc_{val_categorical_accuracy:.2f}'),
        save_weights_only=True),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(check_dir, 'conv_model_best'),
        save_weights_only=True,
        save_best_only=True,
        monitor='val_categorical_accuracy')
]

# Training.
history = model.fit(
    x=data['x_train'],
    y=data['y_train'],
    batch_size=256,
    epochs=2,
    validation_data=(data['x_val'], data['y_val']),
    callbacks=callbacks,
    verbose=2)


# Load saved data.
model = models.mnist_conv_model(config)
_ = eval_model(model, data) # Chance accuracy.

model.load_weights('checkpoints/conv_model_best')
_ = eval_model(model, data) # Best training accuracy.


# Saving an entire model (architecture + weights).
# If include_optimizer=False, the loaded model will need to be compiled.
# If saved in .h5 format, load_model is unable to recover the optimizer state.
#   This seems related to use of a custom layer (DenseBlock).
model.save(filepath='checkpoints/conv_save')

restored_model = tf.keras.models.load_model(
    'checkpoints/conv_save',
    custom_objects={"DenseBlock": models.DenseBlock})

_ = eval_model(restored_model, data)

# ----------------------------------------------------------------------------
# Early stopping.
# ----------------------------------------------------------------------------

# Model.
config = configs.dens_model_config()
model = models.mnist_dens_model(config)
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
eval_model(model, data)

# Prediction.
mnist_utils.plot_random_prediction(x=data['x_test'], model=model)

# ----------------------------------------------------------------------------
# Learning rate scheduler.
# ----------------------------------------------------------------------------

# Model.
config = configs.dens_model_config()
model = models.mnist_dens_model(config)
model.summary()

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

# Evaluation.
eval_model(model, data)