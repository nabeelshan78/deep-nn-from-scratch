import numpy as np
import matplotlib.pyplot as plt
import h5py


def show_image(index, images_flatten, labels, classes):
    plt.imshow(images_flatten[:, index].reshape((64, 64, 3)))
    print ("y = " + str(labels[0,index]) + ". It's a " + classes[labels[0,index]].decode("utf-8") +  " picture.")
    plt.show()

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ



def load_and_prepare_data(show_info=True):
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
    


def plot_costs(costs, learning_rate=0.0075):
    """
    Plots the cost values over training iterations.

    Arguments:
    costs -- list of cost values recorded every 100 iterations
    learning_rate -- learning rate used during training (for annotation)
    """
    plt.figure(figsize=(8, 5))
    plt.plot(costs, color='blue', linewidth=2)
    plt.ylabel("Cost")
    plt.xlabel("Iterations (per hundreds)")
    plt.title(f"Cost reduction over iterations (learning rate = {learning_rate})")
    plt.grid(True)
    plt.show()