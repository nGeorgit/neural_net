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
        self.nodeLayers: list[np.ndarray] = [np.random.rand(i) for i in anodeLayers[1:]]  #create vectors of each leayer with random values as biass from [0,1)
        self.weightLayers: list[np.ndarray] = []
        #self.weightLayers.append(np.random.rand(self.inputSize, anodeLayers[0]))
        for i in range(len(anodeLayers)-1) :
            self.weightLayers.append(np.random.uniform(-1, 1,size=(anodeLayers[i], anodeLayers[i+1])))

    def run(self, inputs: np.ndarray) -> np.ndarray:
        activationVect: np.ndarray = inputs
        for i in range(len(self.nodeLayers)):
            #print(i)
            #print(activationVect)
            #print(self.weightLayers[i])
            #print(np.dot(activationVect, self.weightLayers[i]))
            #print(self.nodeLayers[i].shape)
            activationVect = self.activation_function(np.matmul(activationVect, self.weightLayers[i]) + self.nodeLayers[i])
            #pprint(activationVect)
        return activationVect
    
    def runGetActivationMatrixAndZMatrix(self, inputs: np.ndarray) :#-> tuple(list[np.ndarray], list[np.ndarray]):
        activationMatrix: list[np.ndarray] = []
        ZMatrix: list[np.ndarray] = []
        activationVect: np.ndarray = inputs
        for i in range(len(self.nodeLayers)):
            #print(i)
            #print(activationVect)
            #print(self.weightLayers[i])
            #print(np.matmul(activationVect, self.weightLayers[i]))
            #print(self.nodeLayers[i].shape)
            z: np.ndarray = np.matmul(activationVect, self.weightLayers[i]) + self.nodeLayers[i]
            activationVect = self.activation_function(z)
            activationMatrix.append(activationVect)
            ZMatrix.append(z)
            #pprint(activationVect)
        return activationMatrix,  ZMatrix

    def feedforward(self, a: np.ndarray) -> list[np.ndarray]:
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.nodeLayers, self.weightLayers):
            a = self.activation_function(np.dot(w, a)+b)
        return a


    def activation_function(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(z)

    def train(self, trainInputs: list[np.ndarray], trainExpOut: np.ndarray, step: float = 1.0) -> np.ndarray:
        weightDeriv: list[np.ndarray] = []
        for i in self.weightLayers:
            weightDeriv.append(np.zeros(i.shape))
        biasDeriv: list[np.ndarray] = []
        for i in self.nodeLayers:
            biasDeriv.append(np.zeros(i.shape))

        for i in range(len(trainInputs)):
            #trainInputs[i] = [1,1] #DELETE
            activationMatrix, zMatrix = self.runGetActivationMatrixAndZMatrix(trainInputs[i])

            activationDerivMat: list[np.ndarray] = activationMatrix.copy()
            weightDerivOfTrain: list[np.ndarray] = []

            for j in self.weightLayers:
                weightDerivOfTrain.append(np.zeros(j.shape))

            biasDerivOfTrain: list[np.ndarray] = []
            for j in self.nodeLayers:
                biasDerivOfTrain.append(np.zeros(len(j)))
            
            #trainExpOut[i] = [1,0] #DELETE
            activationDerivMat[-1] = 2*(activationMatrix[-1] - trainExpOut[i])

            biasDerivOfTrain[-1] = sigmoidDeriv(zMatrix[-1])*activationDerivMat[-1]
            for j in range(len(weightDeriv[-1][0])):
                for k in range(len(weightDeriv[-1][0])):
                    weightDerivOfTrain[-1][j][k] = activationMatrix[-2][j]*biasDerivOfTrain[-1][k]

            for l in reversed(range(len(activationMatrix)-1)):
                activationDerivMat[l] = np.zeros(activationDerivMat[l].shape)
                for j in range(len(weightDeriv[l][0])):
                    
                    for j2 in range(len(biasDeriv[l+1])):
                        activationDerivMat[l][j] += self.weightLayers[l+1][j][j2]*biasDerivOfTrain[l+1][j2]
                    biasDerivOfTrain[l][j] = sigmoidDeriv(zMatrix[l][j])*activationDerivMat[l][j]
                    for k in range(len(weightDeriv[l])):
                        if (l-1<0):
                            weightDerivOfTrain[l][k][j] = trainInputs[i][k]*biasDerivOfTrain[l][j]
                        else:
                            weightDerivOfTrain[l][k][j] = activationMatrix[l-1][k]*biasDerivOfTrain[l][j]
            
            
            for l in range(self.size):
                weightDeriv[l] += weightDerivOfTrain[l]
                biasDeriv[l] += biasDerivOfTrain[l]

        for i in range(self.size):
            weightDeriv[i] = weightDeriv[i]/len(trainInputs)
            biasDeriv[i] = biasDeriv[i]/len(trainInputs)


        for i in range(self.size):
            self.weightLayers[i] -= weightDeriv[i]*step
            self.nodeLayers[i] -= biasDeriv[i]*step




            

    def print_nodeLayers(self) -> None:
        for v in self.nodeLayers: pprint(v)
    
    def print_weightLayers(self) -> None:
        for v in self.weightLayers: pprint(v)

def sigmoid(z):
    """The sigmoid function."""
    return 1/(1+np.exp(-z))

def sigmoidDeriv(z):
    return sigmoid(z)*(1-sigmoid(z))

neural = Neural([2,100,2])

trainingData: np.array = np.random.randint(2, size=(500, 10 , 2))




excepted: np.array = []#[[if bool(j[0]) or bool(j[1]) : np.array([1.0,0.0]) else np.array([0.0,1.0]) for j in i] for i in trainingData]
for i in trainingData:
    v = []

    for j in i:

        if bool(j[0]) ^ bool(j[1]):
            v.append([1.0,0.0])
        else:
            v.append([0.0,1.0])
    excepted.append(v)

def printOutputBoolean(output) :
    if (output[0]>output[1]):
        print(True)
    else:
        print(False)


print('weights:')
print(neural.weightLayers)
print('bias:')
print(neural.nodeLayers)
# print('1,0')
# pprint(neural.run([1,0]))
# print('0,1')
# pprint(neural.run([0,1]))
# print('1,1')
# pprint(neural.run([1,1]))
# print('0,0')
# pprint(neural.run([0,0]))
for i in range(len(trainingData)):
    
    neural.train(trainingData[i], excepted[i], step=5)
    
print('weights:')
print(neural.weightLayers)
print('bias:')
print(neural.nodeLayers)



print('1,0')
printOutputBoolean(neural.run([1,0]))
print('0,1')
printOutputBoolean(neural.run([0,1]))
print('1,1')
printOutputBoolean(neural.run([1,1]))
print('0,0')
printOutputBoolean(neural.run([0,0]))

