import numpy as np
import pandas as pd

_SEED = 2013

# ----------------------------------------------------------------------------

def split_df(df: pd.DataFrame):
    """Split DataFrame into dictionary."""
    out = {}
    for col in df.columns:
        out[col] = df[[col]].to_numpy()
    return out


def split_array(x: np.array, names):
    """Split array into dictionary."""
    out = {}
    for i in range(x.shape[1]):
        out[names[i]] = x[:,i]
    return out


# ----------------------------------------------------------------------------


def _permute_data(x, y):
    """Permutes data along the first axis."""
    prng = np.random.RandomState(_SEED)
    n = x.shape[0]
    shuffle = np.arange(n)
    prng.shuffle(shuffle)
    xp = x[shuffle, :]
    yp = y[shuffle, :]
    return (xp, yp)    


# ----------------------------------------------------------------------------


def two_way_split(x, y, permute=False, prop=np.array([0.8, 0.2])):
    """Splits data into training and evaluation sets.
    
    Args:
      x: Input array.
      y: Output array.
      permute: Permute data?
      prop: Vector of length 3 specifying the proportion in each data set.
    
    Returns:
      Dictionary containing: x_train, y_train, x_val, y_val. 
    """
    
    # Determine group sizes.
    prop = prop / prop.sum()
    n = x.shape[0]
    n0 = int(np.round(prop[0] * n))
  
    # Permute data.
    if permute:
        (xp, yp) = _permute_data(x, y)
    else:
        (xp, yp) = (x, y)
    
    # Partition data.
    x_train = xp[:n0, :]
    y_train = yp[:n0, :]

    x_val = xp[n0:, :]
    y_val = yp[n0:, :]
    
    data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
    }

    return data
  

# ----------------------------------------------------------------------------


def three_way_split(x, y, permute=False, prop=np.array([0.8, 0.1, 0.1])):
    """Splits data into training, validation, and test sets.
    
    Args:
      x: Input array.
      y: Output array.
            permute: Permute data?
      prop: Vector of length 3 specifying the proportion in each data set.
    
    Returns:
      Dictionary containing: x_train, y_train, x_val, y_val, x_test, y_test.
    """
    
    # Determine group sizes.
    prop = prop / prop.sum()
    n = x.shape[0]
    n0 = int(np.round(prop[0] * n))
    n1 = n0 + int(np.round(prop[1] * n))
    
    # Permute data.
    if permute:
        (xp, yp) = _permute_data(x, y)
    else:
        (xp, yp) = (x, y)
    
    # Partition data.
    x_train = xp[:n0, :]
    y_train = yp[:n0, :]
    
    x_val = xp[n0:n1, :]
    y_val = yp[n0:n1, :]
    
    x_test = xp[n1:, :]
    y_test = yp[n1:, :]
    
    data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test
    }

    return data

