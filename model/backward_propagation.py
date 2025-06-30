import numpy as np

def relu_backward(dA, cache):
    """
    Compute the backward propagation for the ReLU activation function.

    Arguments:
    dA -- Gradient of the cost with respect to the activation output (A) of the current layer,
          shape: (size of current layer, number of examples)
    cache -- Z, stored from forward propagation

    Returns:
    dZ -- Gradient of the cost with respect to the linear output (Z) of the current layer,
          shape: (size of current layer, number of examples)
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # Create a copy to avoid modifying dA directly
    dZ[Z <= 0] = 0  # Set gradients to 0 where Z <= 0
    return dZ


def sigmoid_backward(dA, cache):
    """
    Compute the backward propagation for the sigmoid activation function.

    Arguments:
    dA -- Gradient of the cost with respect to the activation output (A) of the current layer,
          shape: (size of current layer, number of examples)
    cache -- Z, stored from forward propagation

    Returns:
    dZ -- Gradient of the cost with respect to the linear output (Z) of the current layer,
          shape: (size of current layer, number of examples)
    """
    Z = cache
    A = 1 / (1 + np.exp(-Z))  # Recompute A from Z
    dZ = dA * A * (1 - A)  # Derivative of sigmoid: A * (1 - A)
    return dZ


def linear_backward(dZ, cache):
    """
    Compute the linear part of backward propagation for a single layer.

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (Z) of the current layer,
          shape: (size of current layer, number of examples)
    cache -- tuple of (A_prev, W, b) from the forward propagation of the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation from the previous layer (A_prev),
               shape: (size of previous layer, number of examples)
    dW -- Gradient of the cost with respect to the weights (W),
          shape: (size of current layer, size of previous layer)
    db -- Gradient of the cost with respect to the bias (b),
          shape: (size of current layer, 1)
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Perform the backward propagation for a LINEAR -> ACTIVATION layer.

    Arguments:
    dA -- Gradient of the loss with respect to the activation output of the current layer,
          shape: (size of current layer, number of examples)
    cache -- tuple of (linear_cache, activation_cache) from forward propagation
    activation -- activation function used in this layer: "relu" or "sigmoid"
    print_info -- boolean, whether to print detailed information about shapes and values

    Returns:
    dA_prev -- Gradient with respect to the activation from the previous layer
    dW -- Gradient with respect to the weights
    db -- Gradient with respect to the bias
    """
    linear_cache, activation_cache = cache

    # Step 1: Backward through Activation Function
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


    # Step 2: Backward through Linear Function
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db




def L_model_backward(AL, Y, caches, parameters, dropout_masks=None, keep_probs=None, lambd=0.0):
    """
    Implements full backward propagation with optional L2 and Dropout.

    Arguments:
    AL -- predicted probabilities from forward pass, shape (1, m)
    Y -- true labels, shape (1, m)
    caches -- list of caches from forward propagation
    parameters -- dictionary of weights and biases (used for L2 reg)
    dropout_masks -- list of dropout masks (D) generated during forward pass for hidden layers (L-1 masks)
    keep_probs -- dictionary of probability of keeping a neuron active.
                  Used to retrieve per-layer keep_prob for inverted dropout scaling.
    lambd -- L2 regularization parameter (0.0 means no L2)

    Returns:
    grads -- dictionary with gradients: dA{l}, dW{l+1}, db{l+1}
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))

    # Output layer (sigmoid)
    current_cache = caches[L - 1]
    dA_prev, dW, db = linear_activation_backward(dAL, current_cache, activation='sigmoid')

    if lambd != 0.0:
        dW += (lambd / m) * parameters[f"W{L}"]

    grads[f"dA{L - 1}"] = dA_prev
    grads[f"dW{L}"] = dW
    grads[f"db{L}"] = db

    # Hidden layers (ReLU + dropout)
    for l in range(L - 1, 0, -1):
        dA = grads[f"dA{l}"]

        # Drop Out
        D = dropout_masks[l - 1] if dropout_masks else None
        kp = keep_probs.get(l, 1.0) if keep_probs else 1.0
        if D is not None and kp < 1.0:
            # Apply inverted dropout scaling
            dA = (dA * D) / kp

        current_cache = caches[l - 1]
        dA_prev, dW, db = linear_activation_backward(dA, current_cache, activation='relu')

        if lambd != 0.0:
            dW += (lambd / m) * parameters[f"W{l}"]

        grads[f"dA{l - 1}"] = dA_prev
        grads[f"dW{l}"] = dW
        grads[f"db{l}"] = db

    return grads