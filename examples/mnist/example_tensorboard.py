#!/usr/bin/env python3

import os
import pandas as pd
import subprocess as sp
import tempfile
import tensorflow as tf

# Set local directory?
set_dir = False
if set_dir:
    os.chdir('/Users/zmccaw/Documents/Tensorflow/examples/mnist')

import configs
import load_data
import models
import mnist_utils

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
# Tensorboard
# ----------------------------------------------------------------------------

# Tensorboard.
log_dir = tempfile.mkdtemp()
callbacks = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1)

# Model.
config = configs.conv_model_config()
model = models.mnist_conv_model(config)
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
sp.run(["tensorboard", "--logdir", log_dir])

# History data.
df = pd.DataFrame(history.history)
df.head()

# Evaluation.
_ = eval_model(model, data)

# Prediction.
mnist_utils.plot_random_prediction(x=data['x_test'], model=model)