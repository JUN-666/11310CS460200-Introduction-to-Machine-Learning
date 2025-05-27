import numpy as np

class Activation:
    def __init__(self, activation_function="relu", loss_function_name=None): # Added loss_function_name for consistency with some prompts
        self.activation_function = activation_function
        self.Z = None # Store input Z for backward pass (pre-activation output)
        self.A = None # Store output A for backward pass (activation output, e.g. for softmax)

    def forward(self, Z_input):
        self.Z = Z_input
        if self.activation_function == "relu":
            self.A = np.maximum(0, Z_input)
        elif self.activation_function == "linear":
            self.A = Z_input
        elif self.activation_function == "sigmoid":
            self.A = 1 / (1 + np.exp(-Z_input))
        elif self.activation_function == "softmax":
            # Subtract max for numerical stability
            exp_Z = np.exp(Z_input - np.max(Z_input, axis=-1, keepdims=True))
            self.A = exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)
        else: # Default to linear if unknown
            self.A = Z_input
        return self.A

    def backward(self, dA_output, Y=None): # dA_output is dL/dA (gradient of Loss w.r.t. Activation output)
        dZ = np.copy(dA_output) # Initialize dZ as dL/dA
        if self.activation_function == "relu":
            dZ[self.Z <= 0] = 0 # dL/dZ = dL/dA * (1 if Z > 0 else 0)
        elif self.activation_function == "linear":
            pass # dL/dZ = dL/dA * 1
        elif self.activation_function == "sigmoid":
            # s = 1 / (1 + np.exp(-self.Z)) # This is self.A
            dZ = dA_output * self.A * (1 - self.A) # dL/dZ = dL/dA * A * (1-A)
        elif self.activation_function == "softmax":
            # If CCE loss is used, the combined derivative (dL/dZ) is often AL - Y.
            # The Model.backward handles this specific case by passing Y.
            if Y is not None:
                # This assumes dA_output was AL from forward, and Y are true labels.
                # This makes dZ = AL - Y, which is dL/dZ for Softmax + CCE.
                # The Model.backward method passes dA=AL and Y=Y to this.
                dZ = self.A - Y # A is the cached output of softmax (AL)
            else:
                # This case is more complex if not combined with CCE.
                # For a generic dL/dAL, dL/dZ = dL/dAL * d(softmax)/dZ.
                # This is often not implemented this way due to the CCE combination.
                # For simplicity in dummy, we'll assume the CCE case is handled by Y.
                # If Y is not provided, we'll just pass dL/dAL through (incorrect for general case but placeholder).
                pass # Placeholder for generic softmax derivative if Y is not given
        return dZ
