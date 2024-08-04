from neural_net import Neural
import numpy as np
from mnist_loader import load_data
neural = Neural([784,112, 112,10])



# trainingData: np.array = np.random.randint(2, size=(500, 10 , 2))


tr, val, test = load_data()

trainingData = tr[0]

excepted = []
for i in range(len(tr[1])):
    v = np.zeros(10)
    v[tr[1][i]] = 1
    excepted.append(v)


#make training data into mini batches
trainingData = [trainingData[i:i+10] for i in range(0, len(trainingData), 10)]

# excepted: np.array = []#[[if bool(j[0]) or bool(j[1]) : np.array([1.0,0.0]) else np.array([0.0,1.0]) for j in i] for i in trainingData]
# for i in trainingData:
#     v = []

#     for j in i:

#         if bool(j[0]) ^ bool(j[1]):
#             v.append([1.0,0.0])
#         else:
#             v.append([0.0,1.0])
#     excepted.append(v)

# def printOutputBoolean(output) :
#     if (output[0]>output[1]):
#         print(True)
#     else:
#         print(False)


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
    print(str(i*100/len(trainingData)) + '%')
    print("cost: "+str(neural.train(trainingData[i], excepted[i*10:i*10+10], step=3)))
    
print('weights:')
print(neural.weightLayers)
print('bias:')
print(neural.nodeLayers)

#validation
def get_output_int(output) -> int:
    return np.argmax(output)
print('validation')
s = 0
for i in range(len(val[0])):
    print(str((i*100)/len(val[0]))+"%")
    if get_output_int(neural.run(val[0][i])) == val[1][i]:
        s+=1
        print('correct')
print((s*100)/len(val[1]))

#save the weights and bias on jspn files
import json
with open('./neural_network_save/weights.json.', 'w') as f:
    json.dump([i.tolist() for i in neural.weightLayers], f)
with open('./neural_network_save/bias.json', 'w') as f:
    json.dump([i.tolist() for i in neural.nodeLayers], f)
