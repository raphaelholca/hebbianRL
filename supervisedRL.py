import numpy as np
import math
import os
os.chdir('../DPM/')
import classes.input.mnist as mnist
os.chdir('../RL/')

# np.random.seed(12)

def sigmoid(x):
	try:
		return 1/(1+math.exp(-x))
	except:
		if x <= -100: return 1.
		else: raise OverflowError('math range error; x='+str(np.round(x[0][0],1)))

def evenLabels(images, labels):
	nDigits, bins = np.histogram(labels, bins=nClasses, range=(0,nClasses-1))
	m = np.min(nDigits)
	images_even = np.zeros((m*nClasses, np.size(images,1)))
	labels_even = np.zeros(m*nClasses, dtype=int)
	for c in range(nClasses):
		images_even[c*m:(c+1)*m,:] = images[labels==c,:][0:m,:]
		labels_even[c*m:(c+1)*m] = labels[labels==c][0:m]
	images, labels = np.copy(images_even), np.copy(labels_even)
	return images, labels

classes =  [0,1] # [4,9] #
nClasses = len(classes)
dataset = 'test' # 'train' #
path = '../DPM/data-sets/MNIST'
images, labels = mnist.read_images_from_mnist(classes = classes, dataset = dataset, path = path)
##? Normalize each image to the sum of its pixel values (due to feedforward inhibition in model)
# A=900
# images = (A-images.shape[1])*images/np.sum(images,1)[:,np.newaxis] + 1.
images/=255. #normalize images to range [0,1]
#even out label class distribution
images, labels = evenLabels(images, labels)


nImages = np.size(images,0)

nDimActions, nDimStates = 1, np.size(images,1)
nInpNeurons = nDimStates + nDimActions + 1 # +1 for bias node
nHidNeurons = 0
nEpi = 10000
nPrint = nEpi/10
nImages = np.size(images,0)
npos, nneg = 1.2, 0.5
dmin, dmax, dinit = 0.000001, 50.0, 1.0
d = np.ones(nInpNeurons)*dinit
dW = np.zeros(nInpNeurons)
W = np.random.rand(1, nInpNeurons)-0.5 #weight matrix
Q = 0. #output neuron; Q-value
E = np.zeros(nInpNeurons) #error gradient
prevE = np.zeros(nInpNeurons) #gradient at previous step

allStates = np.ones((nImages, nDimStates+1)) #add 1 to input vector for bias node
allStates[:,:-1] = images #np.random.randint(0 , 2, size=(nInput, nDimStates))
concInput = np.zeros(nInpNeurons) #concatenated input vector with state and action input and bias node
allActions = np.array([0, 1], dtype=int)
allRewards = {'00':1, '01':0, '10':0, '11':1, '40':1, '41':0, '90':0, '91':1} #state+action

#training
tmpErr = 0.
for e in range(nEpi):
	idxState = np.random.randint(nImages) #pick a current state at random
	cState = allStates[idxState, :]
	cAction = np.random.choice(allActions) #choose a random action
	concInput[:nDimStates+1] = cState #concatenate current state and action to be fed to the network
	concInput[nDimStates+1:] = cAction
	cReward = allRewards[str(labels[idxState])+str(cAction)] #get reward associated with state and action

	Q = sigmoid(np.dot(W,concInput[:,np.newaxis])) #compute Q-value/neuron activation
	err = np.abs(Q-cReward)
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
	tmpErr += err
	if e%nPrint==0 and e!=0:
		# print concInput
		# print Q
		# print cReward
		# print
		print np.round(tmpErr/nPrint, 2)
		tmpErr = 0.


#testing
print
totErr = 0.
nEpiTest = 1000
labErr = [0,0]
for e in range(nEpiTest):
	idxState = np.random.randint(nImages) #pick a current state at random
	cState = allStates[idxState, :]
	cAction = np.random.choice(allActions) #choose a random action
	concInput[:nDimStates+1] = cState #concatenate current state and action to be fed to the network
	concInput[nDimStates+1:] = cAction

	Q = sigmoid(np.dot(W,concInput[:,np.newaxis])) #compute Q-value/neuron activation
	cReward = allRewards[str(labels[idxState])+str(cAction)] #get reward associated with state and action

	err = np.abs(Q-cReward)
	totErr += err
	if err > 0.01:
		labErr[labels[idxState]]+=1
print np.round(totErr/nEpiTest, 3)
print labErr







