import numpy as np
from pprint import pprint
import math

class Neural:
    """
    -anodeLayers is a list of numbers. The length of it is the number of layers and it's elemnts are the number of nodes in it's layer.
    This nodes are the hidden layers and the last are the output.
    -ainputSize is the number of the input nodes
    """
    def __init__(self, ainputSize: int, anodeLayers: list[int]) -> None:
        self.inputSize : int = ainputSize
        self.nodeLayers: list[np.ndarray] = [np.random.rand(i, 1) for i in anodeLayers]  #create vectors of each leayer with random values as biass from [0,1)
        self.weightLayers: list[np.ndarray] = []
        self.weightLayers.append(np.random.rand(self.inputSize, anodeLayers[0]))
        for i in range(len(anodeLayers)-1) :
            self.weightLayers.append(np.random.rand(anodeLayers[i], anodeLayers[i+1]))

    def run(self, inputs: np.ndarray) -> np.ndarray:
        activationVect: np.ndarray = inputs
        for i in range(len(self.nodeLayers)):
            #print(i)
            #print(activationVect)
            #print(self.weightLayers[i])
            #print(np.matmul(activationVect, self.weightLayers[i]))
            #print(self.nodeLayers[i].shape)
            activationVect = self.activation_function(np.dot(activationVect, self.weightLayers[i]) + np.transpose(self.nodeLayers[i]))
            pprint(activationVect)
        return activationVect
    
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.nodeLayers, self.weightLayers):
            a = self.activation_function(np.dot(w, a)+b)
        return a


    def activation_function(self, z: np.ndarray) -> np.ndarray:
        return self.sigmoid(z)

    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def print_nodeLayers(self) -> None:
        for v in self.nodeLayers: pprint(v)
    
    def print_weightLayers(self) -> None:
        for v in self.weightLayers: pprint(v)

neural = Neural(50, [10,10])
pprint(neural.run(np.random.rand(1,50)))
