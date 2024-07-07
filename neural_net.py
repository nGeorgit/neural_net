import numpy as np
from pprint import pprint

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
            self.weightLayers.append(np.random.rand(self.inputSize, anodeLayers[i+1]))

    def print_nodeLayers(self) -> None:
        for v in self.nodeLayers: pprint(v)
    
    def print_weightLayers(self) -> None:
        for v in self.weightLayers: pprint(v)

neural = Neural(2, [2,3])
neural.print_nodeLayers()
neural.print_weightLayers()
