import numpy as np
from model.forward_propagation import L_model_forward

def predict(X, y, parameters, verbose=True):
    """
    Predicts binary labels for a dataset using a trained L-layer neural network.

    Arguments:
    X -- input data of shape (n_x, m)
    y -- true labels of shape (1, m)
    parameters -- learned parameters of the model
    verbose -- if True, prints the accuracy

    Returns:
    p -- predicted labels of shape (1, m)
    """
    # Forward propagation (no dropout during testing)
    AL, _, _ = L_model_forward(X, parameters, training=False)

    # Convert probabilities to binary predictions
    p = (AL > 0.5).astype(int)
    if verbose:
        accuracy = np.mean(p == y)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    return p