from typing import List

import ml_collections
import tensorflow as tf

# ----------------------------------------------------------------------------


def get_inputs(inputs: List[ml_collections.ConfigDict]) -> List[tf.Tensor]:
    """Generate model inputs.
    
    Args:
        inputs: List of ConfigDicts specifying the model inputs.
    
    Returns:
        model_inputs: Dictionary of tf.Tensors, keyed by input name.
    
    """
    model_inputs = {}
    for entry in inputs: 
        model_inputs[entry.name] = tf.keras.layers.Input(
            shape=entry.input_shape)
    return model_inputs


# ----------------------------------------------------------------------------


def get_outputs(inputs: tf.Tensor, 
                outcomes: List[ml_collections.ConfigDict]) -> List[tf.Tensor]:
    """Generate outputs.

    Args:
        outcomes: List of ConfigDicts specifying the model outcomes.

    Returns:
        outputs: Dictionary of tf.Tensors, keyed by outcome name.

    """
    outputs = {}
    for out in outcomes: 
        if out.type == 'binary':
            outputs[out.name] = tf.keras.layers.Dense(
                1, activation='sigmoid', name=out.name)(inputs)
        elif out.type == 'continuous':
            outputs[out.name] = tf.keras.layers.Dense(
                1, activation='linear', name=out.name)(inputs)
        elif out.type == 'categorical':
            outputs[out.name] = tf.keras.layers.Dense(
                out.levels, activation='softmax', name=out.name)(inputs)

    return outputs


# ----------------------------------------------------------------------------


def get_losses(outcomes: List[ml_collections.ConfigDict]):
    """Enumerates losses.

    Args:
        outcomes: List of ConfigDicts specifying the model outcomes.

    Returns:
        losses: Dictionary of losses, keyed by outcome name.

    """
    losses = {}
    for out in outcomes:
        if out.type == 'binary':
            losses[out.name] = tf.keras.losses.BinaryCrossentropy()
        elif out.type == 'continuous':
            losses[out.name] = tf.keras.losses.Huber(delta=3)
        elif out.type == 'categorical':
            losses[out.name] = tf.keras.losses.CategoricalCrossentropy()
    return losses


# ----------------------------------------------------------------------------


def get_metrics(outcomes: List[ml_collections.ConfigDict]):
    """Enumerate metrics.

    Args:
        outcomes: List of ConfigDicts specifying the model outcomes.

    Returns:
        metrics: Dictionary of metrics, keyed by outcome name.
    """
    metrics = {}
    for out in outcomes:
        if out.type == 'binary':
            metrics[out.name] = [
                tf.keras.metrics.AUC(name='auprc', curve='PR'),
                tf.keras.metrics.AUC(name='auroc', curve='ROC'),
                tf.keras.metrics.BinaryAccuracy(name='accuracy')
            ]
        elif out.type == 'continuous':
            metrics[out.name] = [
                tf.keras.metrics.MeanSquaredError()
            ]
        elif out.type == 'categorical':
            metrics[out.name] = [
                tf.keras.metrics.CategoricalAccuracy()
            ]
    return metrics


