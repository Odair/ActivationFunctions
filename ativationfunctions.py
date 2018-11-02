import numpy as np

def sumFuction(soma):
    if(soma >= 1):
        return 1
    return 0    

def sigmoidFunction(soma):
    return 1/(1 + np.exp(-soma))    

def tangFunction(soma):
    return (np.exp(soma) - np.exp(-soma))/(np.exp(soma) + np.exp(-soma))    

def reluFunction(soma):
    if(soma > 0):
        return soma
    return 0        

def linearFunction(soma):
    return soma    

def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()    

valor = 2.1

value = [5.0,1.0,3.1]

testSoma = sumFuction(-1)
testSigmoide = sigmoidFunction(valor)
testTang = tangFunction(valor)
testRelu = reluFunction(valor)
testLinear = linearFunction(valor)
testSoftMax = softmaxFunction(value)


print(testSoma)
print(testSigmoide)
print(testTang)    
print(testRelu)
print(testLinear)