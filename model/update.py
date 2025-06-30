import numpy as np

def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent.

    Arguments:
    params -- dictionary containing current weights and biases (before update)
    grads -- dictionary containing gradients from backpropagation
    learning_rate -- learning rate for gradient descent

    Returns:
    parameters -- dictionary containing updated weights and biases
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    return parameters


def update_parameters_with_momentum(parameters, grads, v, learning_rate=0.01, beta=0.9):
    """
    Update parameters using momentum-based gradient descent.

    Arguments:
    parameters -- dictionary containing current weights and biases
    grads -- dictionary containing gradients from backpropagation
    v -- velocity vectors (initialized with initialize_momentum)
    learning_rate -- learning rate for gradient descent
    beta -- momentum hyperparameter (default: 0.9)

    Returns:
    parameters -- dictionary containing updated weights and biases
    v -- updated velocity vectors
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        # Update velocity
        v[f"dW{l}"] = beta * v[f"dW{l}"] + (1 - beta) * grads[f"dW{l}"]
        v[f"db{l}"] = beta * v[f"db{l}"] + (1 - beta) * grads[f"db{l}"]

        # Update parameters
        parameters[f"W{l}"] -= learning_rate * v[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * v[f"db{l}"]

    return parameters, v


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam optimization.

    Arguments:
    parameters -- dictionary containing current weights and biases
    grads -- dictionary containing gradients from backpropagation
    v -- first moment vector (initialized with initialize_adam)
    s -- second moment vector (initialized with initialize_adam)
    t -- time step counter (initialized with initialize_adam)
    learning_rate -- learning rate for Adam
    beta1 -- exponential decay rate for the first moment estimates
    beta2 -- exponential decay rate for the second moment estimates
    epsilon -- small constant to avoid division by zero

    Returns:
    parameters -- dictionary containing updated weights and biases
    v -- updated first moment vector
    s -- updated second moment vector
    t -- incremented time step counter
    """
    L = len(parameters) // 2
    t += 1

    for l in range(1, L + 1):
        # Moving average of gradients
        v[f"dW{l}"] = beta1 * v[f"dW{l}"] + (1 - beta1) * grads[f"dW{l}"]
        v[f"db{l}"] = beta1 * v[f"db{l}"] + (1 - beta1) * grads[f"db{l}"]
        # Compute bias-corrected first raw moment estimate
        v_corrected_W = v[f"dW{l}"] / (1 - np.power(beta1, t))
        v_corrected_b = v[f"db{l}"] / (1 - np.power(beta1, t))
        
        # Moving average of squared gradients
        s[f"dW{l}"] = beta2 * s[f"dW{l}"] + (1 - beta2) * np.square(grads[f"dW{l}"])
        s[f"db{l}"] = beta2 * s[f"db{l}"] + (1 - beta2) * np.square(grads[f"db{l}"])
        # Compute bias-corrected second raw moment estimate
        s_corrected_W = s[f"dW{l}"] / (1 - np.power(beta2, t))
        s_corrected_b = s[f"db{l}"] / (1 - np.power(beta2, t))

        # Update parameters using Adam update rule
        parameters[f"W{l}"] -= learning_rate * v_corrected_W / (np.sqrt(s_corrected_W) + epsilon)
        parameters[f"b{l}"] -= learning_rate * v_corrected_b / (np.sqrt(s_corrected_b) + epsilon)

    return parameters, v, s, t



