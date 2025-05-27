import numpy as np

class Flatten:
    def __init__(self):
        self.cache = None

    def forward(self, A_prev):
        self.cache = A_prev.shape
        return A_prev.reshape(A_prev.shape[0], -1)

    def backward(self, dA):
        return dA.reshape(self.cache)
