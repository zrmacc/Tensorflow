# pip install ml-collections==0.1.0

import ml_collections

# -----------------------------------------------------------------------------
# Convolution model configuration.
# -----------------------------------------------------------------------------

def model_config() -> ml_collections.ConfigDict:

	config = ml_collections.ConfigDict()

    # List of ConfigDicts. 1 for each input.
	config.inputs = [
        ml_collections.ConfigDict({
            'name': 'temp',
            'input_shape': (1,)
        }),
        ml_collections.ConfigDict({
            'name': 'nausea',
            'input_shape': (1,)
        }),
        ml_collections.ConfigDict({
            'name': 'lumbar',
            'input_shape': (1,)
        }),
        ml_collections.ConfigDict({
            'name': 'nausea',
            'input_shape': (1,)
        }),
        ml_collections.ConfigDict({
            'name': 'urine',
            'input_shape': (1,)
        }),
        ml_collections.ConfigDict({
            'name': 'mictur',
            'input_shape': (1,)
        }),
    ]

	# List of ConfigDicts. 1 for each output.
	config.outputs = [
        ml_collections.ConfigDict({
            'name': 'inflammation',
            'input_shape': (1,)
        }),
        ml_collections.ConfigDict({
            'name': 'nephritis',
            'input_shape': (1,)
        }),
    ]

	# Hyperparameters.
	config.hparams = ml_collections.ConfigDict({
		'dens_units': 10,
		'dens_l1_penalty': 1e-6,
		'dens_l2_penalty': 1e-6,
		'dens_dropout': 0.2
	})

	return config 


# -----------------------------------------------------------------------------