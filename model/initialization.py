import numpy as np

def initialize_parameters_deep(layer_dims, layer_inits=None):
    """
    Initialize weights and biases for an L-layer deep neural network, with optional per-layer initialization.

    Arguments:
    layer_dims -- list of layer sizes including input and output layers
    layer_inits -- dict mapping layer index (1-based) to init type: "zeros", "random", "he", "xavier" or "glorot"
                   If None, defaults to "he" initialization for all layers.
                   (e.g., {1: "he", 2: "xavier"})

    Returns:
    parameters -- dictionary of initialized weights and biases
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    total_parameters = 0

    print(f"--- Initializing Deep Neural Network Parameters ---")
    print(f"Network Architecture (Layer Sizes): {layer_dims}")

    for l in range(1, L):
        init_type = "he"                          # Default
        if layer_inits and l in layer_inits:
            init_type = layer_inits[l].lower()

        prev_layer_size = layer_dims[l - 1]
        curr_layer_size = layer_dims[l]

        if init_type == "zeros":
            W = np.zeros((curr_layer_size, prev_layer_size))
        elif init_type == "random":
            W = np.random.randn(curr_layer_size, prev_layer_size) * 10
        elif init_type == "he":
            W = np.random.randn(curr_layer_size, prev_layer_size) * np.sqrt(2. / prev_layer_size)
        elif init_type == "xavier":
            W = np.random.randn(curr_layer_size, prev_layer_size) * np.sqrt(1. / prev_layer_size)
        elif init_type == "glorot":
            W = np.random.randn(curr_layer_size, prev_layer_size) * np.sqrt(2. / (prev_layer_size + curr_layer_size))
        else:
            raise ValueError(f"Invalid initialization type '{init_type}' for layer {l}")

        b = np.zeros((curr_layer_size, 1))

        parameters[f"W{l}"] = W
        parameters[f"b{l}"] = b

        # Calculate parameters for current layer
        num_weights = W.size
        num_biases = b.size
        layer_params = num_weights + num_biases
        total_parameters += layer_params

        print(f"\n--- Layer {l} ---")
        print(f"  Initialization Type: {init_type}")
        print(f"  W{l} shape: {W.shape}")
        print(f"  b{l} shape: {b.shape}")
        print(f"  Parameters in Layer {l}: {layer_params}")

    print(f"\n--- Initialization Summary ---")
    print(f"Total Trainable Parameters in Network: {total_parameters:,}")

    print(f"Parameters Breakdown by Layer:")
    for l in range(1, L):
        print(f"  Layer {l}: W{l} shape: {parameters[f'W{l}'].shape}, b{l} shape: {parameters[f'b{l}'].shape} initialized with {layer_inits.get(l, 'he') if layer_inits else 'he'}")

    return parameters


def initialize_momentum(parameters):
    """
    Initialize momentum parameters for gradient descent.

    Arguments:
    parameters -- dictionary of parameters from initialize_parameters_deep()

    Returns:
    v -- dictionary containing initialized velocity vectors
    """
    v = {}
    L = len(parameters) // 2

    for l in range(1, L + 1):
        v[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
        v[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])

    return v


def initialize_rmsprop(parameters):
    """
    Initialize RMSProp parameters for gradient descent.

    Arguments:
    parameters -- dictionary of parameters from initialize_parameters_deep()

    Returns:
    s -- dictionary containing initialized squared gradients
    """
    s = {}
    L = len(parameters) // 2

    for l in range(1, L + 1):
        s[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
        s[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])

    return s


def initialize_adam(parameters):
    """
    Initialize Adam optimizer parameters.

    Arguments:
    parameters -- dictionary of parameters from initialize_parameters_deep()

    Returns:
    v -- dictionary containing initialized first moment vector
    s -- dictionary containing initialized second moment vector
    t -- time step counter (initialized to 0)
    """
    v = {}
    s = {}
    L = len(parameters) // 2

    for l in range(1, L + 1):
        v[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
        v[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])
        s[f"dW{l}"] = np.zeros_like(parameters[f"W{l}"])
        s[f"db{l}"] = np.zeros_like(parameters[f"b{l}"])

    t = 0

    return v, s, t

