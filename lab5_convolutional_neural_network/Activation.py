class Activation:
    def __init__(self, activation_function="relu"): # Added common param
        self.activation_function = activation_function
        self.cache = None

    def forward(self, Z):
        pass

    def backward(self, dA):
        pass

    # Activation layers typically don't have parameters to update
    # def update(self, learning_rate):
    #     pass
