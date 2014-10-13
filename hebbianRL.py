import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.chdir('../DPM/')
import classes.input.mnist as mnist
from utils import accel
os.chdir('../RL/')

# np.random.seed(12)

def softmax(activ):
	""" input should be (training examples x neurons) """
	#vectorial
	scale = np.clip(np.max(activ,1)-700, 0, np.inf)
	tmpRND=np.random.rand(np.shape(activ)[0],np.shape(activ)[1])/100000
	activ+=tmpRND #add a random offset to insure that there is only a single min
	activ[activ==np.min(activ,1)[:,np.newaxis]] = np.clip(np.min(activ,1), -740+scale, np.inf)
	activ-=tmpRND
	return np.exp(activ-scale[:,np.newaxis]) / np.sum(np.exp(activ-scale[:,np.newaxis]), 1)[:,np.newaxis]

	# iterative
	# activ_SM = np.zeros_like(activ)
	# for i in range(np.size(activ,0)):
	# 	scale = 0
	# 	I = np.copy(activ[i,:])
	# 	if (I[np.argmax(I)] > 700):
	# 	    scale  = I[np.argmax(I)] - 700
	# 	if (I[np.argmin(I)] < -740 + scale):
	# 	    I[np.argmin(I)] = -740 + scale
	# 	activ_SM[i,:] = np.exp(I-scale) / np.sum(np.exp(I-scale))
	# return activ_SM

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

def learningStep(bInput, W_in):
	hidNeurons = np.dot(bInput, accel.log(W_in))
	hidNeurons = softmax(hidNeurons)
	dW_in = lr*(np.dot(bInput.T, hidNeurons) - np.sum(hidNeurons, 0)*W_in)
	return W_in + dW_in

def plotRF(W, e=''):
	plt.figure()
	for i in range(np.size(W,1)):
		plt.subplot(4,5,i)
		plt.imshow(np.reshape(W[:,i], (28,28)), interpolation='nearest')
		plt.xticks([])
		plt.yticks([])
	plt.suptitle('episode ' + e)

classes =  [0,1] # [4,9]
nClasses = len(classes)
dataset = 'test'
path = '../DPM/data-sets/MNIST'
images, labels = mnist.read_images_from_mnist(classes = classes, dataset = dataset, path = path)
#Normalize each image to the sum of its pixel values (due to feedforward inhibition in model)
A=900
images = (A-images.shape[1])*images/np.sum(images,1)[:,np.newaxis] + 1.
images, labels = evenLabels(images, labels)

nEpi = 8
nImages = np.size(images,0)
nDimStates = np.size(images,1)
nDimActions = 0
nDimReward = 0
nInpNeurons = nDimStates + nDimActions + nDimReward
nHidNeurons = 20
lr = 0.5*(nHidNeurons/np.float(nImages)) #learning rate
sBatch = 20 #mini-batch size
hidNeurons = np.zeros(nHidNeurons)[np.newaxis,:]
W_in = np.random.random_sample(size=(nInpNeurons, nHidNeurons)) + 1. ##something more than this?
concInput = np.zeros((nImages, nInpNeurons)) #concatenated input vector with state and action input and bias node

for e in range(nEpi):
	print e

	#concatenate state-action-reward
	#stateIdx = np.random.randint(0, nImages)
	#cState
	#cAction
	#cReward
	concInput[:] = images ##add cAction and cReward

	#shuffle input
	rndIdx = np.arange(nImages)
	np.random.shuffle(rndIdx)
	concInput = concInput[rndIdx,:]
	rndLabel = labels[rndIdx]

	for b in range(int(nImages/sBatch)): #may leave a few training examples out
		bInput = concInput[b*sBatch:(b+1)*sBatch,:]
		W_in = learningStep(bInput, W_in)

plotRF(np.copy(W_in), e=str(e))

plt.show()

































