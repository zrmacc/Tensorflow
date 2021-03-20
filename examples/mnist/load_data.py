# Load datasets. 

import numpy as np
import sys
import tensorflow as tf

sys.path.append("..")
import utils.data_utils as utils

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

