# Example of a multi-input, multi-output model.

import matplotlib.pyplot as plt
import os
import tensorflow as tf

# Set local directory?
set_dir = False
if set_dir:
    os.chdir('/Users/zmccaw/Documents/Tensorflow/examples/mnist')

import load_data

# ---------------------------------------------------------------------------

# Load data.
data = load_data.load_inflammation_data()

# Inputs.
inputs = {
    'temp_celsius': tf.keras.layers.Input(shape=(1,), name='temp_celsius'),
    'nausea': tf.keras.layers.Input(shape=(1,), name='nausea'),
    'lumbar_pain':  tf.keras.layers.Input(shape=(1,), name='lumbar_pain'),
    'urination':  tf.keras.layers.Input(shape=(1,), name='urination'),
    'micturition_pain':  tf.keras.layers.Input(shape=(1,), 
                                               name='micturition_pain'),
    'burning':  tf.keras.layers.Input(shape=(1,), name='burning')
}

# Concatenat inputs.
x = tf.keras.layers.concatenate(list(inputs.values()))

# Outputs.
outputs = {
    'inflammation': 
        tf.keras.layers.Dense(1, activation='sigmoid', 
                              name='inflammation')(x),
    'nephritis': 
        tf.keras.layers.Dense(1, activation='sigmoid', 
                              name='nephritis')(x)
}
    
# Model.
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# tf.keras.utils.plot_model(model)

# Losses.
losses = {
    'inflammation': tf.keras.losses.BinaryCrossentropy(),
    'nephritis': tf.keras.losses.BinaryCrossentropy()    
}

# Metrics:
metrics = {
    'inflammation': [
            tf.keras.metrics.BinaryAccuracy(name='acc'),
            tf.keras.metrics.AUC(name='auroc')
        ],
    'nephritis': [
            tf.keras.metrics.BinaryAccuracy(name='acc'),
            tf.keras.metrics.AUC(name='auroc')
        ]
}

# Compile.
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=losses,
    metrics=metrics    
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=5
)

# Fit.
history = model.fit(
    x=data['x_train'],
    y=data['y_train'],
    validation_data=(data['x_val'], data['y_val']),
    epochs=1000,    
    verbose=0,
    callbacks=[early_stopping]
)

# Plot losses.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Validation'])
plt.show()

# Evaluate.
_ = model.evaluate(
    x=data['x_test'],
    y=data['y_test']    
)
