import numpy as np

def predict(model, X, y=None):
    """
    This function is used to predict the results of a  n-layer neural network.
    Arguments:
    model -- a Model object, the n-layer neural network
    X -- data set of examples you would like to label
    y -- true "label" vector (optional)
    Returns:
    p -- predictions for the given dataset X
    """
    m = X.shape[0]
    p = np.zeros((1,m), dtype = np.int_) # Ensures p is integer array

    # Forward propagation
    AL = model.forward(X)

    # Convert probas to 0/1 predictions
    if AL.shape[0] == 1: # Binary classification
        for i in range(AL.shape[1]):
            if AL[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
    else: # Multi-class classification
        p = np.argmax(AL, axis=0).reshape(1, -1)


    # Print accuracy if y is provided
    if y is not None:
        y = y.reshape(1, -1) # Ensure y is 2D row vector
        print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))

    return p
