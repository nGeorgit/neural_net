import numpy as np
from pprint import pprint

class Neural:
    """
    anodeLayers is a list of numbers. The length of it is the number of layers and it's elemnts are the number of nodes in it's layer.
    This nodes are the hidden layers and the last are the output.
    """
    def __init__(self, anodeLayers: list[int]) -> None:
        self.nodeLayers: list[np.ndarray] = [np.random.rand(i, 1) for i in anodeLayers]  #create vectors of each leayer with random values as biass from [0,1)

    def print_nodeLayers(self) -> None:
        for v in self.nodeLayers: pprint(v)

neural = Neural([2,3])
neural.print_nodeLayers()
