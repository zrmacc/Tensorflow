# Load datasets. 

import numpy as np
import pandas as pd
import sys

sys.path.append("..")
import utils.data_utils as utils

# ----------------------------------------------------------------------------

def load_inflammation_data():
    """Loads and splits the inflammation data.
    
    Returns:
        Dictionary containing x_train, y_train, x_val, y_val, x_test, y_test.
    """
    
    # Load data.
    data = pd.read_csv('data/data.tsv', sep='\t')
    x = data[['temp_celsius', 'nausea', 'lumbar_pain', 
              'urination', 'micturition_pain', 'burning']]
    y = data[['inflammation', 'nephritis']]
    
    # Names.
    x_names = x.columns
    y_names = y.columns
    
    # Convert to numpy.
    x = np.array(x)
    y = np.array(y)
    
    # Partition training data into training and evaluation.
    data = utils.three_way_split(x, y)
    
    # Split into dictionaries.
    out = {}
    for key in data.keys():
        if 'x' in key:
            out[key] = utils.split_array(data[key], x_names)
        elif 'y' in key:
            out[key] = utils.split_array(data[key], y_names)
    
    return out

# ----------------------------------------------------------------------------