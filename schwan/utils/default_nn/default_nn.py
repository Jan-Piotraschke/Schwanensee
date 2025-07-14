import deepxde as dde
import os

def create(layer_sizes, data, project_name):
    """
    Creates a default neural network with the specified configuration.

    Parameters:
    -----------
    layer_sizes : list
        Sizes of the neural network layers.
    data : deepxde.data.PDE or similar
        The DeepXDE data object for training.
    project_name : str
        Name of the project, used for checkpoint directory naming.

    Returns:
    --------
    model : deepxde.Model
        The built DeepXDE model.
    callbacks : list
        List of callbacks for training.
    """
    # Create network with default settings
    activation = "tanh"
    kernel_initializer = "He normal"
    dropout_rate = 0.01
    net = dde.nn.FNN(
        layer_sizes=layer_sizes,
        activation=activation,
        kernel_initializer=kernel_initializer,
        dropout_rate=dropout_rate,
    )

    # Build model
    model = dde.Model(data, net)

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = f"./checkpoints/{project_name}"
    os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)

    # Set up callbacks
    checkpointer = dde.callbacks.ModelCheckpoint(
        checkpoint_dir,
        verbose=1,
        save_better_only=False,
        period=1000,
    )
    early_stopping = dde.callbacks.EarlyStopping(min_delta=0.5, patience=1000)

    return model, [checkpointer, early_stopping]
