import numpy as np

def compute_BCE_loss(Y, AL):
    """
    Compute the Binary Cross-Entropy loss.
    Arguments:
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Returns:
    cost -- binary cross-entropy cost
    """
    m = Y.shape[1]
    # Compute loss from AL and Y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)      # E.g., turns [[17]] into 17
    assert(isinstance(cost, float))
    return cost
