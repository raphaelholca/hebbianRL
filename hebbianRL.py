import numpy as np
import matplotlib.pyplot as plt
import random as rdm
import sys
import h5py
import pdb
import os
os.chdir('../DPM/')
import classes.input.mnist as mnist
from utils import accel
os.chdir('../RL/')

np.random.seed(10)

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

def evenLabels(images, labels, classes):
	nDigits, bins = np.histogram(labels, bins=10, range=(0,9))
	m = np.min(nDigits[nDigits!=0])
	images_even = np.zeros((m*nClasses, np.size(images,1)))
	labels_even = np.zeros(m*nClasses, dtype=int)
	for i, c in enumerate(classes):
		images_even[i*m:(i+1)*m,:] = images[labels==c,:][0:m,:]
		labels_even[i*m:(i+1)*m] = labels[labels==c][0:m]
	images, labels = np.copy(images_even), np.copy(labels_even)
	return images, labels

def learningStep(bInput, W_in, bReward=np.ones(1)):
	hidNeurons = np.dot(bInput, accel.log(W_in))
	hidNeurons = softmax(hidNeurons)*bReward[:, np.newaxis]
	dW_in = lr*(np.dot(bInput.T, hidNeurons) - np.sum(hidNeurons, 0)*W_in)
	return W_in + dW_in, hidNeurons

def plotRF(W, e=''):
	v = int(np.sqrt(nHidNeurons))
	h = int(np.ceil(float(nHidNeurons)/v))
	plt.figure()
	for i in range(np.size(W,1)):
		plt.subplot(v,h,i+1)
		plt.imshow(np.reshape(W[:nDimStates,i], (28,28)), interpolation='nearest')
		plt.xticks([])
		plt.yticks([])
	plt.suptitle('episode ' + e)

classes  = [4, 7, 9] # np.arange(10) #
rActions = [0, 1, 2] #rewarded actions for each class
nClasses = len(classes)
dataset = 'test'
path = '../DPM/data-sets/MNIST'
images, labels = mnist.read_images_from_mnist(classes = classes, dataset = dataset, path = path)

#Normalize each image to the sum of its pixel value (feedforward inhibition)
A=900
images = (A-images.shape[1])*images/np.sum(images,1)[:,np.newaxis] + 1.
images, labels = evenLabels(images, labels, classes)

nEpi = 20
singleActiv = 20.
nImages = np.size(images,0)
nDimStates = np.size(images,1)
nDimActions = 3 #nClasses
nInpNeurons = nDimStates + nDimActions
nHidNeurons = 3
rHigh = 1.0
rLow = 0.0
lr = 0.5*(nHidNeurons/np.float(nImages)) #learning rate
nBatch = 60 #mini-batch size
hidNeurons = np.zeros(nHidNeurons)[np.newaxis,:]
W_in = np.random.random_sample(size=(nInpNeurons, nHidNeurons)) + 1. ##something more than this?
concInput = np.zeros((nImages, nInpNeurons)) #concatenated input vector with state and action input and bias node
randActions = True

for e in range(nEpi):
	print e
	cAction = np.ones((nImages, nDimActions))
	cReward = np.ones(nImages)

	#(randomly) pick actions and assign reward accordingly
	if randActions:
		cActionIdx = np.random.randint(0, nDimActions, size=nImages)
	else:
		cActionIdx = np.ones(nImages, dtype=int)
		for i,c in enumerate(classes):
			cActionIdx[labels==c] = rActions[i]

	cAction[np.arange(nImages),cActionIdx] = singleActiv
	cReward = np.ones(nImages)*rLow
	for i in range(nClasses):
		cReward[np.logical_and(labels==classes[i], cActionIdx==rActions[i])] = rHigh

	#concatenate state-action
	concInput[:,0:nDimStates] = images #States
	concInput[:,nDimStates:nDimStates+nDimActions] = cAction #Actions

	#shuffle input
	rndIdx = np.arange(nImages)
	np.random.shuffle(rndIdx)
	concInput = concInput[rndIdx,:]
	rndLabel = np.copy(labels[rndIdx])
	cReward = cReward[rndIdx]

	#train network
	for b in range(int(nImages/nBatch)): #may leave a few training examples out
		bInput = concInput[b*nBatch:(b+1)*nBatch,:]
		bReward = cReward[b*nBatch:(b+1)*nBatch]
		W_in, hidNeurons = learningStep(bInput, W_in, bReward)

plotRF(np.copy(W_in), e=str(e))


# h5file = h5py.File('data/W_untuned', 'w')
# h5file['W_untuned'] = W_in
# h5file.close()

plt.show(block=False)

































