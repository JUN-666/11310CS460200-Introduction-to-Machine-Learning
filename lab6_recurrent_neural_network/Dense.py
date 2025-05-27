import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim, seed=None): # Modified signature
        if seed is not None:
            np.random.seed(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize W, b 
        self.W = np.random.randn(output_dim, input_dim) * 0.01 # (output, input)
        self.b = np.zeros((output_dim, 1)) # (output, 1)
        
        self.A_prev = None # Input to this layer
        self.Z = None      # Output of this layer (pre-activation)
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        # A_prev shape: (batch_size, input_dim)
        self.A_prev = A_prev
        # W.T shape: (input_dim, output_dim)
        # Z = A_prev @ W.T + b.T
        # Z shape: (batch_size, output_dim)
        self.Z = np.dot(A_prev, self.W.T) + self.b.T 
        return self.Z

    def backward(self, dZ): # dZ is dL/dZ for this layer, shape (batch_size, output_dim)
        m = self.A_prev.shape[0] # batch_size
        
        # dL/dW = dL/dZ * dZ/dW = dL/dZ * A_prev
        # dZ (batch_size, output_dim)
        # A_prev (batch_size, input_dim)
        # dW must be (output_dim, input_dim)
        self.dW = (1/m) * np.dot(dZ.T, self.A_prev)
        
        # dL/db = dL/dZ * dZ/db = dL/dZ * 1
        # db must be (output_dim, 1)
        self.db = (1/m) * np.sum(dZ.T, axis=1, keepdims=True)
        
        # dL/dA_prev = dL/dZ * dZ/dA_prev = dL/dZ * W
        # dA_prev must be (batch_size, input_dim)
        dA_prev = np.dot(dZ, self.W)
        return dA_prev

    def update(self, learning_rate):
        if self.dW is not None and self.db is not None:
            self.W -= learning_rate * self.dW
            self.b -= learning_rate * self.db
