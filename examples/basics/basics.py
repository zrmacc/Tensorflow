import tensorflow as tf

# Tensorflow variable.
t1 = tf.Variable([0, 1, 2], dtype=tf.float32, name='t1')

# Tensorflow constant.
t2 = tf.constant([2], shape=(2, 2), name='t2')

# Tensor rank (number of dimensions, not matrix rank).
tf.rank(t2)

# Reshape.
t3 = tf.reshape([[0, 1], [1, 2]], shape=(4, 1, 1))
t3 = tf.squeeze(t3)  # Remove unused dimensions.
t3 = tf.expand_dims(t3, axis=1)  # Add an extra dimension.

# Standard tensors.
t0 = tf.zeros(shape=(3, 3), name='zeros')
t1 = tf.ones(shape=(4, 1), name='ones')
t2 = tf.eye(num_rows=3, name='ident')

# Matrix multiplication.
a = tf.Variable([[2, 1],[1, 2]], dtype=tf.float32)
b = tf.constant([1, 1], dtype=tf.float32, shape=(2, 1))
c = tf.matmul(a, b)

# Random numbers.
t1 = tf.random.normal(shape=(2, 2), mean=0.0, stddev=1.0)
t2 = tf.random.uniform(shape=(2, 2), minval=0, maxval=1.0)

# ---------------------------------------------------------------------------

# Simple example model.
inputs = tf.keras.layers.Input(shape=(32,), name='Input')
h = tf.keras.layers.Dense(10, activation='relu', name='Internal')(inputs)
outputs = tf.keras.layers.Dense(32, activation='linear', name='Output')(h)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# Extract output of hidden layer. 
h = model.get_layer('Internal').outputs
model2 = tf.keras.Model(inputs=model.input, outputs=h)
