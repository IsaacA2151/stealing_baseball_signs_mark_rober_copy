import numpy as np
import random
import decimal

def seqToBin(actions,seq,show=False):
    seqBin = []
    for i in range(len(seq)):
        seqBin.append([0 for i in range(len(actions))])
        for k in range(len(seq[i])):
            if seq[i][k] in actions:
                seqBin[i][actions.index(seq[i][k])] = 1
                
    seqBin = np.asarray(seqBin)
    return seqBin                

def findSign(weights,actions):
    largest = weights[0]
    ind = 0
    for i in range(len(weights)):
        if weights[i] > largest:
            largest = weights[i]
            ind = i
            
    return actions[ind]
            

actions = ['eye','mouth','nose','ear']
seq = [['eye','mouth','ear'],
       ['eye','mouth','nose'],
       ['eye','nose','ear'],
       ['mouth','nose'],
       ['nose'],
       ['ear','mouth'],
       ['ear','nose'],
       ['mouth','nose','ear'],
       ['eye','nose','mouth']]
       


############################################ Mouth means steal ############################################

seqBin = seqToBin(actions,seq)

steal_or_not = np.array([1,1,0,1,0,1,0,1,1])
weights = np.array([float(random.uniform(-1,1)) for i in range(len(actions))])
alpha = 0.1

input = seqBin[0]
goal = steal_or_not[0]

for it in range(100):
    for row in range(len(steal_or_not)):
        
        input = seqBin[row]
        goal = steal_or_not[row]
        pred = input.dot(weights)
        
        error = (goal - pred) ** 2
        delta = pred - goal
        
        weights = weights - (alpha * (input * delta))
        
    print('Prediction: ' + str(pred))
    

print(weights)
print(findSign(weights,actions))
    
