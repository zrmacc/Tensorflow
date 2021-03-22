# Utilities specific to MNIST data.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------------

def plot_random_prediction(x, model: tf.keras.Model) -> None:
    """Plot random image with predicted label."""
    n = x.shape[0]
    idx = np.random.choice(n)
    probs = model.predict(np.expand_dims(x[idx,:], axis=0))['Number']
    probs = probs[0, :]
    yhat = np.argmax(probs)
    yprob = probs[yhat]
    
    plt.imshow(x[idx,:,:,0] * 255.0, cmap='gray')
    plt.title(f'Prediction: {yhat}. Probability: {yprob:.2f}')
    plt.show()

    return None