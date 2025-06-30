import numpy as np

def sigmoid(Z):
    """
    Compute the sigmoid activation function.

    Arguments:
    Z -- linear component (pre-activation), shape: (size of current layer, number of examples)

    Returns:
    A -- output of the sigmoid function (post-activation value)
    cache -- Z, stored for backpropagation
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
    """
    Compute the ReLU activation function.

    Arguments:
    Z -- linear component (pre-activation), shape: (size of current layer, number of examples)

    Returns:
    A -- output of the ReLU function (post-activation value)
    cache -- Z, stored for backpropagation
    """
    A = np.maximum(0, Z)
    cache = Z

    return A, cache


def linear_forward(A, W, b):
    """
    Compute the linear part of forward propagation for a single layer.

    Arguments:
    A -- activations from the previous layer (or input data),
         shape: (size of previous layer, number of examples)
    W -- weights matrix,
         shape: (size of current layer, size of previous layer)
    b -- bias vector,
         shape: (size of current layer, 1)

    Returns:
    Z -- linear component of activation (pre-activation)
    cache -- tuple (A, W, b) stored for backpropagation
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Perform forward propagation for the LINEAR -> ACTIVATION layer.

    Arguments:
    A_prev -- activations from the previous layer (or input data),
              shape: (size of previous layer, number of examples)
    W -- weights matrix,
          shape: (size of current layer, size of previous layer)
    b -- bias vector,
          shape: (size of current layer, 1)
    activation -- string: "sigmoid" or "relu" specifying the activation function

    Returns:
    A -- output of the activation function (post-activation value)
    cache -- tuple (linear_cache, activation_cache) for use in backpropagation
    """

    # Step 1: Linear part
    Z, linear_cache = linear_forward(A_prev, W, b)

    # Step 2: Activation part
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

    cache = (linear_cache, activation_cache)

    return A, cache



def L_model_forward(X, parameters, keep_probs=None, training=True):
    """
    Perform forward propagation with optional Dropout.

    Arguments:
    X -- input data, shape: (input size, number of examples)
    parameters -- dictionary of parameters from initialize_parameters_deep()
    keep_probs -- probability of keeping a neuron active (e.g., 0.8 means drop 20%)
                  - A dict, apply per layer (mapping layer index (1-based) to keep_prob)
    training -- boolean flag, True for training (dropout active), False for inference

    Returns:
    AL -- output of the final layer (predictions), shape: (1, number of examples)
    caches -- list of caches from each layer, used for backpropagation
    dropout_masks -- list of dropout masks applied to hidden layers during training;
                    empty if dropout is disabled or in inference mode
    """
    caches = []
    dropout_masks = []
    A = X
    L = len(parameters) // 2

    # Loop for hidden layers (L-1 layers)
    for l in range(1, L):
        A_prev = A
        current_keep_prob = keep_probs.get(l, 1.0) if keep_probs is not None else 1.0

        A, cache = linear_activation_forward(
            A_prev,
            parameters[f'W{l}'],
            parameters[f'b{l}'],
            activation='relu'       # Using ReLU for hidden layers
        )
        caches.append(cache)

        if training and current_keep_prob < 1.0:
            # Apply dropout mask during training
            D = (np.random.rand(A.shape[0], A.shape[1]) < current_keep_prob).astype(int)
            dropout_masks.append(D)
            A = np.multiply(A, D)
            A /= current_keep_prob  

    # Final layer (output layer)
    AL, cache = linear_activation_forward(
        A,
        parameters[f'W{L}'],
        parameters[f'b{L}'],
        activation='sigmoid'
    )
    caches.append(cache)

    return AL, caches, dropout_masks 
