import numpy as np


class MorphOperation:
    def __init__(self, operation, kernelSize, iterations=1):
        self.operation = operation
        self.kernel = np.ones((kernelSize, kernelSize), np.uint8)
        self.iterations = iterations
