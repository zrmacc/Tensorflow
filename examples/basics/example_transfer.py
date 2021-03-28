import tensorflow as tf

# ----------------------------------------------------------------------------
# Data generator
# ----------------------------------------------------------------------------

def gen_data(batch_size):
    while True:
        # Cases and controls.
        n1 = int(batch_size / 2.)
        n0 = batch_size - n1
        y1 = tf.constant(1, shape=(n1,))
        y0 = tf.constant(0, shape=(n0,))
        y = tf.concat([y1, y0], axis=0)
        
        # Inputs.
        x11 = tf.random.normal((n1, 299, 299), mean=1., stddev=1.)
        x12 = tf.random.normal((n1, 299, 299), mean=2., stddev=2.)
        x13 = tf.random.normal((n1, 299, 299), mean=3., stddev=3.)
        x1 = tf.stack([x11, x12, x13], axis=3)
        
        x21 = tf.random.normal((n1, 299, 299), mean=3., stddev=3.)
        x22 = tf.random.normal((n1, 299, 299), mean=1., stddev=1.)
        x23 = tf.random.normal((n1, 299, 299), mean=2., stddev=2.)
        x0 = tf.stack([x21, x22, x23], axis=3)
        x = tf.concat([x1, x0], axis=0)
        yield x, y
        

# ----------------------------------------------------------------------------
# Transfer model.
# ----------------------------------------------------------------------------

# Xception model.
xception = tf.keras.applications.Xception()

# Layers.
n_layer = len(xception.layers)

# Input layer.
print(xception.input)

# Output shape.
print(xception.output)

# Last layer.
last_layer = xception.layers[n_layer - 1]

# Create model by removing the output layer.
feature_extractor = tf.keras.Model(
    inputs=xception.input,
    outputs=last_layer.input,
    name='Feature_extraction')

# Set trainable to false for entire sub-model.
feature_extractor.trainable = False

# New model based on feature extractor.
model = tf.keras.models.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(units=1, type='sigmoid', name='New_output')
])

# Compile and summarize.
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

model.summary()

# Train.
datagen = gen_data(batch_size=32)
model.fit(datagen, steps_per_epoch=32, epochs=10)
    

