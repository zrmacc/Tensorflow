# Callback examples.
# See: <https://www.tensorflow.org/guide/keras/custom_callback>.

import tensorflow as tf

# ----------------------------------------------------------------------------
# Reporting callbacks.
# ----------------------------------------------------------------------------

class TrainCallbacks(tf.keras.callbacks.Callback):
    
    def on_train_begin(self, logs=None):
        print("Starting training.")
        
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")
    
    def on_train_batch_begin(self, batch, logs=None):
        print(f"Starting batch {batch}")
    
    def on_train_batch_end(self, batch, logs=None):
        print(f"Finished batch {batch}")
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"Finished epoch {epoch}")
    
    def on_train_end(self, logs=None):
        print("Finished training.")


class ValLossCallback(tf.keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs=None):
        print("Epoch {} validation loss: {:.3f}".format(epoch, logs['val_loss']))
        

# ----------------------------------------------------------------------------
# Learning rate scheduler.
# ----------------------------------------------------------------------------

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler.

  Args:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch {}: Learning rate is {:.5f}".format(epoch, scheduled_lr))