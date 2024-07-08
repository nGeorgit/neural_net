import numpy as np
from pprint import pprint
import math

class Neural:
    """
    -anodeLayers is a list of numbers. The length of it is the number of layers and it's elemnts are the number of nodes in it's layer.
    This nodes are the hidden layers and the last are the output.
    -ainputSize is the number of the input nodes
    """
    def __init__(self,anodeLayers: list[int]) -> None:
        self.inputSize : int = anodeLayers[0]
        self.size: int = len(anodeLayers)-1
        self.nodeLayers: list[np.ndarray] = [np.random.rand(i, 1) for i in anodeLayers[1:]]  #create vectors of each leayer with random values as biass from [0,1)
        self.weightLayers: list[np.ndarray] = []
        #self.weightLayers.append(np.random.rand(self.inputSize, anodeLayers[0]))
        for i in range(len(anodeLayers)-1) :
            self.weightLayers.append(np.random.rand(anodeLayers[i], anodeLayers[i+1]))

    def run(self, inputs: np.ndarray) -> np.ndarray:
        activationVect: np.ndarray = inputs
        for i in range(len(self.nodeLayers)):
            #print(i)
            #print(activationVect)
            #print(self.weightLayers[i])
            print(np.dot(activationVect, self.weightLayers[i]))
            #print(self.nodeLayers[i].shape)
            activationVect = self.activation_function(np.dot(activationVect, self.weightLayers[i]) + np.transpose(self.nodeLayers[i]))
            pprint(activationVect)
        return activationVect
    
    def runGetActivationMatrixAndZMatrix(self, inputs: np.ndarray) :#-> tuple(list[np.ndarray], list[np.ndarray]):
        activationMatrix: list[np.ndarray] = []
        ZMatrix: list[np.ndarray] = []
        activationVect: np.ndarray = inputs
        for i in range(len(self.nodeLayers)):
            #print(i)
            #print(activationVect)
            print(self.weightLayers[i])
            #print(np.matmul(activationVect, self.weightLayers[i]))
            #print(self.nodeLayers[i].shape)
            z: np.ndarray = np.dot(activationVect, self.weightLayers[i]) + np.transpose(self.nodeLayers[i])
            activationVect = self.activation_function(z)
            activationMatrix.append(activationVect)
            ZMatrix.append(z)
            pprint(activationVect)
        return activationMatrix, 

    def feedforward(self, a: np.ndarray) -> list[np.ndarray]:
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.nodeLayers, self.weightLayers):
            a = self.activation_function(np.dot(w, a)+b)
        return a


    def activation_function(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(z)

    def train(self, trainInputs: list[np.ndarray], trainExpOut: list[np.ndarray], step: float = 1.0) -> np.ndarray:
        weightDeriv: list[np.ndarray] = []
        for i in self.weightLayers:
            weightDeriv.append(np.zeros(i.shape))
        biasDeriv: list[np.ndarray] = []
        for i in self.nodeLayers:
            biasDeriv.append(np.zeros(i.shape))

        for i in range(len(trainInputs)):
            activationMatrix, zMatrix = self.runGetActivationMatrixAndZMatrix(trainInputs[i])
            costVect: np.ndarray = (activationMatrix[-1] - trainExpOut[i])^2
            activationDerivMat: list[np.ndarray] = activationMatrix.copy()
            weightDerivOfTrain: list[np.ndarray] = []

            for i in self.weightLayers:
                weightDerivOfTrain.append(np.zeros(i.shape))

            biasDerivOfTrain: list[np.ndarray] = []
            for i in self.nodeLayers:
                biasDerivOfTrain.append(np.zeros(i.shape))
            
            activationDerivMat[-1] = 2*(activationMatrix[-1] - trainExpOut[i])

            for j in range(len(weightDeriv[-1][0])):
                biasDerivOfTrain[-1][j] = sigmoidDeriv(zMatrix[-1][j])*activationDerivMat[-1][j]
                for k in range(len(weightDeriv[-1])):
                    weightDerivOfTrain[-1][k][j] = activationMatrix[-1][j]*biasDeriv[-1][j]

            for l in reversed(range(len(activationMatrix)-1)):
                for j in range(len(weightDeriv[l][0])):
                    activationDerivMat[l][j] = 0
                    for j2 in range(len(biasDeriv[l+1])):
                        activationDerivMat[l][j] += self.weightLayers[l+1][j][j2]*biasDerivOfTrain[l+1][j2]
                    biasDerivOfTrain[l][j] = sigmoidDeriv(zMatrix[l][j])*activationDerivMat[l][j]
                    for k in range(len(weightDeriv[l])):
                        weightDerivOfTrain[l][k][j] = activationMatrix[l][j]*biasDerivOfTrain[l][j]
            
            for l in range(self.size):
                weightDeriv[l] += weightDerivOfTrain[l]
                biasDeriv[l] += biasDerivOfTrain[l]

        for i in range(self.size):
            weightDeriv[l] = weightDeriv[l]/len(trainInputs)
            biasDeriv[l] = biasDeriv[l]/len(trainInputs)

        for i in range(self.size):
            self.weightLayers[l] -= weightDeriv[l]
            self.nodeLayers = biasDeriv[l]




            

    def print_nodeLayers(self) -> None:
        for v in self.nodeLayers: pprint(v)
    
    def print_weightLayers(self) -> None:
        for v in self.weightLayers: pprint(v)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoidDeriv(z):
    return -np.exp(z)/(np.exp(z)+1)^2

neural = Neural([2,2, 1])
neural.weightLayers[0]=[[0,1],[0,1]]
pprint(neural.weightLayers)
pprint(neural.run(np.random.rand(1,2)))
