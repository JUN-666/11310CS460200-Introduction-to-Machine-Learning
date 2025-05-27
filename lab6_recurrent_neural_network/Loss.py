import numpy as np

def compute_CCE_loss(Y, AL):
    """
    Compute the Categorical Cross-Entropy loss.
    Arguments:
    Y -- true "label" vector (one-hot encoded), shape (number of classes, number of examples)
    AL -- probability vector corresponding to your label predictions, shape (number of classes, number of examples)
    Returns:
    cost -- categorical cross-entropy cost
    """
    m = Y.shape[1]
    cost = - (1./m) * np.sum(Y * np.log(AL + 1e-8)) # Add epsilon for numerical stability
    cost = np.squeeze(cost)      # E.g., turns [[17]] into 17
    # assert(isinstance(cost, float)) # Cost can be a numpy.float64
    return cost

def compute_MSE_loss(Y, AL):
    """
    Compute the Mean Squared Error loss.
    Arguments:
    Y -- true "label" vector, shape (output_size, number of examples)
    AL -- predicted value vector, shape (output_size, number of examples)
    Returns:
    cost -- mean squared error cost
    """
    m = Y.shape[1]
    cost = (1./m) * np.sum(np.square(AL - Y))
    cost = np.squeeze(cost)
    return cost
