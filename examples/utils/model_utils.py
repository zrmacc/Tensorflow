from typing import List

import ml_collections
import tensorflow as tf

# ----------------------------------------------------------------------------


def get_outputs(inputs: tf.Tensor, outcomes: List[ml_collections.ConfigDict]) -> List[tf.Tensor]:
    """Generate outputs.

    Args:
        outcomes: List of ConfigDicts specifying the model outcomes.

    Returns:
        outputs: List of tf.Tensors.

    """
    outputs = []
    for out in outcomes: 
        if out.type == 'binary':
            emit = tf.keras.layers.Dense(
                1, activation='sigmoid', name=out.name)(inputs)
        elif out.type == 'continuous':
            emit = tf.keras.layers.Dense(
                1, activation='linear', name=out.name)(inputs)
        elif out.type == 'categorical':
            emit = tf.keras.layers.Dense(
                out.levels, activation='softmax', name=out.name)(inputs)
        outputs.append(emit)

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


