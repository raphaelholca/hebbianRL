import numpy as np
import math

def sigmoid(x):
	return 1/(1+math.exp(-x))

np.random.seed(12)

nStates = 1
nDimActions, nDimStates = 1, 5
nInpNeurons = nDimStates + nDimActions + 1 # +1 for bias node
nHidNeurons = 0
nEpi = 1000
nInput = 1
npos, nneg = 1.2, 0.5
dmin, dmax, dinit = 0.000001, 50.0, 1.0
d = np.ones(nInpNeurons)*dinit
dW = np.zeros(nInpNeurons)
W = np.random.rand(1, nInpNeurons)-0.5 #weight matrix
Q = 0. #output neuron; Q-value
E = np.zeros(nInpNeurons) #error gradient
prevE = np.zeros(nInpNeurons) #gradient at previous step
lr = 1. #learning rate

allStates = np.ones((nInput, nDimStates+1)) #add 1 to input vector for bias node
allStates[:,:-1] = np.random.randint(0 , 2, size=(nInput, nDimStates))
concInput = np.zeros(nInpNeurons) #concatenated input vector with state and action input and bias node
allActions = np.array([0, 1], dtype=int)
R = np.array([[+1, 0],[0, +1]]) #reward matrix (states x actions)

for e in range(nEpi):
	idxState = np.random.randint(nStates) #pick a current state at random
	cState = allStates[idxState, :]
	cAction = np.random.choice(allActions) #choose a random action
	concInput[:nDimStates+1] = cState
	concInput[nDimStates+1:] = cAction
	cReward = R[idxState, cAction] #get reward associated with state and action

	Q = sigmoid(np.dot(W,concInput[:,np.newaxis])) #compute Q-value/neuron activation
	err = (1./2.)*(Q-cReward)**2
	dOut = (Q-cReward)*Q*(1-Q)*concInput #partial derivatives of the error with respect to the input weights

	E = dOut
	s = np.sign(prevE * E)

	d[s==+1] *= npos
	d[s==-1] *= nneg
	np.clip(d, dmin, dmax)
	dW[s==+1] = d[s==+1] * np.sign(E[s==+1])
	# dW[s==-1] = -d[s==-1] ##? back-tracking, to include or not?
	dW[s==0] = d[s==0] * np.sign(E[s==0])
	E[s==-1] = 0.

	W -= dW
	prevE = E

	# W -= dOut*lr #gradient descent
	if e%100==0:print np.round(err, 6)