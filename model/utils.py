import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.datasets import make_circles


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Create random mini-batches from the dataset.

    Arguments:
    X -- input data, shape: (input size, number of examples)
    Y -- true labels, shape: (1, number of examples)
    mini_batch_size -- size of each mini-batch
    seed -- random seed for reproducibility

    Returns:
    mini_batches -- list of tuples (X_mini_batch, Y_mini_batch) for each mini-batch
    """
    np.random.seed(seed)
    m = X.shape[1]  # number of examples
    mini_batches = []

    # Shuffle the dataset
    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    inc = mini_batch_size
    num_complete_mini_batches = m // mini_batch_size

    for i in range(0, num_complete_mini_batches):
        mini_batch_X = shuffled_X[:, i * inc:(i + 1) * inc]
        mini_batch_Y = shuffled_Y[:, i * inc:(i + 1) * inc]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    # Handle the last mini-batch if it is smaller than mini_batch_size
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_mini_batches * inc:]
        mini_batch_Y = shuffled_Y[:, num_complete_mini_batches * inc:]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches


def schedule_lr_decay(learning_rate0, epoch_num, decay_rate=0.1, time_interval=100):
    """
    Update learning rate using exponential decay.

    Arguments:
    learning_rate0 -- initial learning rate
    epoch_num -- current epoch number
    decay_rate -- decay rate for learning rate (default: 0.1)
    time_interval -- time interval for decay (default: 100)

    Returns:
    new_learning_rate -- updated learning rate
    """
    new_learning_rate = learning_rate0 * (1 / (1 + decay_rate * np.floor(epoch_num / time_interval)))
    return new_learning_rate


def compute_cost(AL, Y, parameters=None, lambd=0.0):
    """
    Compute the binary cross-entropy cost, with optional L2 regularization.

    Arguments:
    AL -- predicted probabilities, shape: (1, number of examples)
    Y -- true labels, shape: (1, number of examples)
    parameters -- dictionary containing weights "W1", ..., "WL" (used for L2 regularization)
    lambd -- L2 regularization hyperparameter (default: 0, i.e., no regularization)

    Returns:
    cost -- scalar value of the (regularized) cross-entropy loss
    """
    m = Y.shape[1]

    # Cross-entropy cost
    epsilon = 1e-15  # small constant to avoid log(0)
    AL = np.clip(AL, epsilon, 1 - epsilon)
    cross_entropy_cost = (-1 / m) * np.sum(
        np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))
    )


    # cross_entropy_cost = (-1 / m) * np.sum(
    #     np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))
    # )

    # L2 regularization cost
    l2_cost = 0
    if lambd != 0 and parameters is not None:
        L = len(parameters) // 2  # number of layers
        for l in range(1, L + 1):
            W = parameters[f"W{l}"]
            l2_cost += np.sum(np.square(W))
        l2_cost = (lambd / (2 * m)) * l2_cost

    cost = cross_entropy_cost + l2_cost
    cost = np.squeeze(cost)  # ensure scalar

    return cost


def load_and_prepare_cat_data(show_info=True):
    """
    Loads the cat vs non-cat dataset, reshapes labels, flattens and standardizes inputs.
    
    Args:
        show_info -- if True, prints dataset statistics

    Returns:
        train_x -- training set inputs, shape (num_px*num_px*3, m_train)
        train_y -- training set labels, shape (1, m_train)
        test_x -- test set inputs, shape (num_px*num_px*3, m_test)
        test_y -- test set labels, shape (1, m_test)
        classes -- class labels array
    """
    # Load from .h5 files
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_x_orig = np.array(train_dataset["train_set_x"][:])
    train_y = np.array(train_dataset["train_set_y"][:]).reshape(1, -1)

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_x_orig = np.array(test_dataset["test_set_x"][:])
    test_y = np.array(test_dataset["test_set_y"][:]).reshape(1, -1)

    classes = np.array(test_dataset["list_classes"][:])

    # Dataset info
    m_train = train_x_orig.shape[0]
    m_test = test_x_orig.shape[0]
    num_px = train_x_orig.shape[1]

    if show_info:
        print(f"Number of training examples: {m_train}")
        print(f"Number of testing examples: {m_test}")
        print(f"Each image is of size: ({num_px}, {num_px}, 3)")
        print(f"train_x_orig shape: {train_x_orig.shape}")
        print(f"train_y shape: {train_y.shape}")
        print(f"test_x_orig shape: {test_x_orig.shape}")
        print(f"test_y shape: {test_y.shape}")

    # Flatten images (num_px, num_px, 3) → (num_px*num_px*3, m)
    train_x = train_x_orig.reshape(m_train, -1).T / 255.
    test_x = test_x_orig.reshape(m_test, -1).T / 255.

    if show_info:
        print(f"train_x's shape: {train_x.shape}")
        print(f"test_x's shape: {test_x.shape}")

    return train_x, train_y, test_x, test_y, classes


def show_image(index, images_flatten, labels, classes):
    plt.imshow(images_flatten[:, index].reshape((64, 64, 3)))
    print ("y = " + str(labels[0,index]) + ". It's a " + classes[labels[0,index]].decode("utf-8") +  " picture.")
    plt.show()


def load_circle_dataset(n=300, noise=0.05):
    """
    Generates a synthetic binary classification dataset using sklearn's make_circles.
    
    Returns:
    - train_X: training features of shape (2, m_train)
    - train_Y: training labels of shape (1, m_train)
    - test_X: test features of shape (2, m_test)
    - test_Y: test labels of shape (1, m_test)
    """
    # Generate training and test data
    np.random.seed(1)
    train_X, train_Y = make_circles(n_samples=n, noise=noise)
    np.random.seed(2)
    test_X, test_Y = make_circles(n_samples=n//3, noise=noise)

    # Plot dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    plt.title("Training Data - Circular Dataset")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

    # Reshape data
    train_X = train_X.T  # shape (2, m_train)
    train_Y = train_Y.reshape(1, -1)  # shape (1, m_train)
    test_X = test_X.T    # shape (2, m_test)
    test_Y = test_Y.reshape(1, -1)    # shape (1, m_test)

    # Print shapes
    print(f"train_X shape: {train_X.shape}")
    print(f"train_Y shape: {train_Y.shape}")
    print(f"test_X shape:  {test_X.shape}")
    print(f"test_Y shape:  {test_Y.shape}")

    return train_X, train_Y, test_X, test_Y






def plot_costs(costs, learning_rate=0.0075):
    """
    Plot the cost function over epochs.

    Arguments:
    costs -- list of costs recorded during training
    learning_rate -- learning rate used in training (for labeling the plot)
    """
    plt.plot(costs)
    plt.title(f"Cost vs Epochs (Learning Rate: {learning_rate})")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.grid()
    plt.show()





def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary of a model over a 2D feature space.

    Parameters:
    - model: a function that takes a 2D array of shape (m, 2) and returns predictions
    - X: input data of shape (2, m)
    - y: labels of shape (m,)
    """
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    # Create a grid of points to classify
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Prepare input for prediction
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    print(f"Grid shape: {xx.shape}, Flat shape: {grid_points.shape}")

    # Predict over the grid
    Z = model(grid_points)
    print(f"Prediction shape before reshape: {Z.shape}")
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and data points
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.show()