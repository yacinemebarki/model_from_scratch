import numpy as np

class flatt:
    def __init__(self,input,n_neurons):
        self.n_neurons=n_neurons
        self.input=input
        self.weight=np.random.rand(input.size(),n_neurons)*0.01
        self.bias=np.zeros(n_neurons)
        self.original_shape=input.shape
    def forward(self):
        self.input=self.input.reshape(-1)

        z=self.input @ self.weight +self.bias
        return z
    def backdrop(self,da):
        return da.reshape(self.original_shape)
    

