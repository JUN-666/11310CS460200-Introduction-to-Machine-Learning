class Dense:
    def __init__(self, input_size=None, output_size=None, seed=None): # Added common params
        self.W = None
        self.b = None
        self.cache = None
        self.dW = None
        self.db = None
        if input_size and output_size: # Allow initialization if sizes are given
            pass # Actual parameter initialization would go here

    def forward(self, A_prev):
        pass

    def backward(self, dA):
        pass

    def update(self, learning_rate):
        pass
