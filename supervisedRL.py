import numpy as np
import math
import matplotlib.pyplot as plt
import os
os.chdir('../DPM/')
import classes.input.mnist as mnist
os.chdir('../RL/')

np.random.seed(12)

def sigmoid(x):
	# print 'x : ' + str(np.round(x[0][0],0))
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
dataset = 'train' # 'train' #
path = '../DPM/data-sets/MNIST'
images, labels = mnist.read_images_from_mnist(classes = classes, dataset = dataset, path = path)
##? Normalize each image to the sum of its pixel values (due to feedforward inhibition in model)
# A=900
# images = (A-images.shape[1])*images/np.sum(images,1)[:,np.newaxis] + 1.
images/=255. #normalize images to range [0,1]
#even out label class distribution
images, labels = evenLabels(images, labels)

nImages = np.size(images,0)
nDimActions, nDimStates = 0, np.size(images,1)
nInpNeurons = nDimStates + nDimActions + 1 # +1 for bias node
nHidNeurons = 0
nEpi = 20
nImages = np.size(images,0)
npos, nneg = 1.2, 0.5
dmin, dmax, dinit = 0.000001, 50.0, 0.1
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
for e in range(nEpi):
	batchErr = 0.
	E = np.zeros(nInpNeurons)
	dW = np.zeros(nInpNeurons)
	for idxState in range(nImages):
		cState = allStates[idxState, :]
		# cAction = np.random.choice(allActions) #choose a random action
		concInput[:nDimStates+1] = cState #concatenate current state and action to be fed to the network
		##concInput[nDimStates+1:] = cAction
		##cReward = allRewards[str(labels[idxState])+str(cAction)] #get reward associated with state and action
		cReward = labels[idxState] ##

		# print 'labels : ' + str(cReward)
		Q = sigmoid(np.dot(W,concInput[:,np.newaxis])) #compute Q-value/neuron activation
		# print 'Q : ' + str(np.round(Q,3))

		batchErr += np.abs(Q-cReward)
		E += (Q-cReward)*Q*(1-Q)*concInput #partial derivatives of the error with respect to the input weights

	print 'batch err : ' + str(np.round(batchErr/nImages,5))
	s = np.sign(prevE * E)

	#compute weight update and adaptation
	d[s==+1] *= npos
	d[s==-1] *= nneg
	np.clip(d, dmin, dmax)
	dW[s==+1] = d[s==+1] * np.sign(E[s==+1])
	# dW[s==-1] = -d[s==-1] ##? back-tracking, to include or not?
	dW[s==0] = d[s==0] * np.sign(E[s==0])
	E[s==-1] = 0.

	W -= dW
	prevE = E
	print


#testing
# print 'testing...'
# dataset = 'test'
# images, labels = mnist.read_images_from_mnist(classes = classes, dataset = dataset, path = path)
# images/=255.
# images, labels = evenLabels(images, labels)
# nImages = np.size(images,0)

# allStates = np.ones((nImages, nDimStates+1)) #add 1 to input vector for bias node
# allStates[:,:-1] = images

# totErr = 0.
# countErr = 0.
# for i in range(nImages):
# 	Q = sigmoid(np.dot(W,allStates[i,:][:,np.newaxis]))
# 	err = np.abs(Q-labels[i])
# 	totErr += err
# 	if err >= 0.1: countErr += 1.
# print 'test error: ' + str(np.round(totErr/nImages,3))
# print countErr
# print 'test error count: ' + str(np.round(countErr/nImages,3))






