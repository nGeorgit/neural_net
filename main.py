from neural_net import Neural
import numpy as np
neural = Neural([2,4,2])

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

