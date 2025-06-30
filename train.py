import numpy as np
from model.initialization import initialize_parameters_deep, initialize_momentum, initialize_adam, initialize_rmsprop
from model.forward_propagation import L_model_forward
from model.utils import random_mini_batches, compute_cost
from model.backward_propagation import L_model_backward
from model.update import update_parameters_with_adam, update_parameters_with_momentum, update_parameters_with_gd
from model.predict import predict
from model.utils import schedule_lr_decay


def L_layer_model(X, Y, layers_dims, layer_inits=None, learning_rate=0.0075, optimizer="gd", 
                mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, 
                num_epochs=1000, keep_probs=None, lambd=0.0, decay=None, decay_rate=0, time_interval=100, print_cost=True):
    """    
    Implements a deep neural network model with optional dropout, L2 regularization, and various optimization algorithms (GD, Momentum, Adam).
    Arguments:
    X -- input data, shape: (input size, number of examples)
    Y -- true labels, shape: (1, number of examples)
    layers_dims -- list of layer sizes including input and output layers
    layer_inits -- dict mapping layer index (1-based) to init type: "zeros", "random", "he", "xavier" or "glorot"
                   If None, defaults to "he" initialization for all layers.
                   (e.g., {1: "he", 2: "xavier"})
    learning_rate -- learning rate for gradient descent (default: 0.0075)
    optimizer -- optimization algorithm: "gd" (Gradient Descent), "momentum", or "adam"
    mini_batch_size -- size of each mini-batch (default: 64)
    beta -- momentum hyperparameter (default: 0.9, used only if optimizer is "momentum")
    beta1 -- exponential decay rate for the first moment estimates (default: 0.9, used only if optimizer is "adam")
    beta2 -- exponential decay rate for the second moment estimates (default: 0.999, used only if optimizer is "adam")
    epsilon -- small constant to avoid division by zero (default: 1e-8, used only if optimizer is "adam")
    num_epochs -- number of epochs for training (default: 1000)
    keep_probs -- dictionary of probability of keeping a neuron active.
                  Used to retrieve per-layer keep_prob for inverted dropout scaling.
                  If None, no dropout is applied.
    lambd -- L2 regularization hyperparameter (default: 0.0, i.e., no regularization)
    decay -- boolean, whether to apply learning rate decay (default: None, i.e., no decay)
    decay_rate -- decay rate for learning rate (default: 0, i.e., no decay)
    time_interval -- time interval for decay (default: 100, used only if decay is True)
    print_cost -- boolean, whether to print the cost every 100 iterations (default: True)
    
    Returns:
    parameters -- Learned parameters
    costs -- every 100 epochs

    """

    np.random.seed(1)
    costs = []
    seed = 10
    t = 0  # For Adam

    # Initialize parameters
    parameters = initialize_parameters_deep(layers_dims, layer_inits)

    # Initialize optimizer-specific parameters
    if optimizer == "momentum":
        v = initialize_momentum(parameters)
    elif optimizer == "gd":
        pass
    elif optimizer == "adam":
        v, s, t = initialize_adam(parameters)

    # Optimiztion loop
    for i in range(num_epochs):
        # Shuffle and create mini-batches
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        seed += 1
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch

            # Forward propagation
            AL, caches, dropout_masks = L_model_forward(
                mini_batch_X, parameters, keep_probs=keep_probs, training=True
            )

            # Compute cost
            cost_total += compute_cost(AL, mini_batch_Y, parameters, lambd)

            # Backward propagation
            grads = L_model_backward(AL, mini_batch_Y, caches, parameters,
                                     dropout_masks=dropout_masks,
                                     keep_probs=keep_probs,
                                     lambd=lambd)
                                     
            # Update parameters based on the optimizer
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(
                    parameters, grads, v, learning_rate=learning_rate, beta=beta
                )
            elif optimizer == "adam":
                t = t + 1  
                parameters, v, s, t = update_parameters_with_adam(
                    parameters, grads, v, s, t,
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon
                )
        
        cost_avg = cost_total / len(mini_batches)
        if print_cost and (i % 100 == 0 or i == num_epochs - 1):
            print(f"Cost after epoch {i}: {cost_avg:.4f}")
            costs.append(cost_avg)
            
        # Learning rate decay
        if decay and i % 100 == 0:
            learning_rate = schedule_lr_decay(learning_rate, i, decay_rate=decay_rate, time_interval=time_interval)
            print(f"Updated learning rate after epoch {i}: {learning_rate:.6f}")
        
    return parameters, costs