import numpy as np


class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    def step(self):
        for p, v in zip(self.parameters, self.velocities):
            v *= self.momentum
            v += p.grad
            p.data -= self.lr * v
