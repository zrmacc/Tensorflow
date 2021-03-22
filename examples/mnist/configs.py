# pip install ml-collections==0.1.0

import ml_collections

# -----------------------------------------------------------------------------
# Convolution model configuration.
# -----------------------------------------------------------------------------

def conv_model_config() -> ml_collections.ConfigDict:

	config = ml_collections.ConfigDict()

    # List of ConfigDicts. 1 for each input.
	config.inputs = [
        ml_collections.ConfigDict({
            'name': 'Image',
            'input_shape': (28, 28, 1)
        })
    ]


	# List of ConfigDicts. 1 for each output.
	config.outputs = [
        ml_collections.ConfigDict({
            'name': 'Number',
            'type': 'categorical',
            'levels': 10
        })
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
# Dense model configuration.
# -----------------------------------------------------------------------------


def dens_model_config() -> ml_collections.ConfigDict:

	config = ml_collections.ConfigDict()

    # List of ConfigDicts. 1 for each input.
	config.inputs = [
        ml_collections.ConfigDict({
            'name': 'Image',
            'input_shape': (28, 28, 1)
        })
    ]

	# List of ConfigDicts. 1 for each output.
	config.outputs = [
        ml_collections.ConfigDict({
            'name': 'Number',
            'type': 'categorical',
            'levels': 10
        })
    ]

	# Hyperparameters.
	config.hparams = ml_collections.ConfigDict({
		'dens_units': 784,
		'dens_l1_penalty': 1e-3,
		'dens_l2_penalty': 1e-3,
		'dens_dropout': 0.5
	})

	return config 