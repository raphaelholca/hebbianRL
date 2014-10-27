""" 
This function trains a hebbian neural network to learn a representation from the MNIST dataset. It makes use of a reward/relevance signal that increases the learning rate when the network makes a correct state-action pair selection.

Output is saved under RL/data/[runName]
"""

import numpy as np
import matplotlib.pyplot as pyplot
import random as rdm
import support.external as ex
import support.plots as pl
from configobj import ConfigObj
import support.mnist as mnist
pl = reload(pl)
ex = reload(ex)

""" 
experimental variables

classes (int) 	: class of the MNIST dataset to use to train the network
rActions (str)	: for each class of MNIST, the action that is rewarded. '0' indicates a class that is never rewarded; '1' indicates a class that is always rewarded; chararcters (e.g., 'a', 'b', etc.) indicate the specific action that is rewarded.
"""
classes 	= np.array([ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
rActions 	= np.array(['0','0','0','0','0','0','0','0','0','1'], dtype='|S1')

# classes 	= np.array([ 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
# rActions 	= np.array(['0','0','0','0','0','1'], dtype='|S1')

# classes 	= np.array([ 4 , 7 , 9 ], dtype=int)
# rActions 	= np.array(['a','a','a'], dtype='|S1')


""" parameters """
nRun 		= 2					# number of runs
nEpiCrit	= 20				# number of 'critical period' episodes in each run (episodes when reward is not required for learning)
nEpiAdlt	= 5					# number of 'adult' episodes in each run (episodes when reward is not required for learning)
seed 		= None 				# seed of the random number generator
A 			= 900 				# image normalization constant
runName 	= 'test'			# name of the folder where to save results
dataset 	= 'test'			# MNIST dataset to use; legal values: 'test', 'train'
singleActiv = 20. 				# activation value of the action neurons
nHidNeurons = 20				# number of hidden neurons
rCrit		= 1.0 				# learning rate multiplier during 'critica period'
rHigh 		= 3.0				# learning rate multiplier with relevance signal (ACh) during critical period
rLow 		= 1.0				# lr multiplier without relevant signal (no ACh), i.e., most of the time outside of critical period
nBatch 		= 60 				# mini-batch size
lr 			= 0.05 				# learning rate
randActions = False				# whether to take random actions (True) or to take best possible action

""" load and pre-process images """
ex.checkdir(runName)
print "importing data..."
imPath = '../data-sets/MNIST'
images, labels = mnist.read_images_from_mnist(classes = classes, dataset = dataset, path = imPath)
images = ex.normalize(images, A)
images, labels = ex.evenLabels(images, labels, classes)

""" variable initialization """
W_in_save = {}
W_class_save = {}
nEpiTot = nEpiCrit + nEpiAdlt
lActions = np.unique(rActions[np.logical_and(rActions!='0', rActions!='1')]) #legal actions
np.random.seed(seed)
nClasses = len(classes)
nImages = np.size(images,0)
nDimStates = np.size(images,1)
nDimActions = len(lActions)
nInpNeurons = nDimStates + nDimActions
concInput = np.zeros((nImages, nInpNeurons)) #concatenated input vector with state, action
lr *= nHidNeurons/np.float(nBatch) #learning rate adjusted to the number of neurons and mini-batch size

""" training of the network """
for r in range(nRun):
	print 'run: ' + str(r+1)
	#initialize network variables
	W_in = np.random.random_sample(size=(nInpNeurons, nHidNeurons)) + 1.
	W_class = np.random.random_sample(size=(nHidNeurons, nClasses)) + 1.

	for e in range(nEpiTot):
		#reset reward-action variables
		cAction = np.ones((nImages, nDimActions))
		cReward = np.ones(nImages)

		#(randomly) pick actions
		if nDimActions != 0:
			if randActions:
				cActionVal = np.random.choice(lActions, size=nImages)
				cActionIdx = ex.val2idx(cActionVal, lActions)
			else:
				cActionVal = ex.labels2actionVal(labels, classes, rActions)	
				cActionIdx = ex.val2idx(cActionVal, lActions)
		else: cActionIdx, cActionVal = [], []

		#assign reward according to state-action pair
		if nDimActions != 0: cAction[np.arange(nImages),cActionIdx] = singleActiv
		if e >= nEpiCrit:
			cReward = np.ones(nImages)*rLow
			for i in range(nClasses):
				cReward[np.logical_and(labels==classes[i], cActionVal==rActions[i])] = rHigh #reward correct state-action pairs
				cReward[np.logical_and(labels==classes[i], '1'==rActions[i])] = rHigh #reward states that are always rewarded
				cReward[np.logical_and(labels==classes[i], '0'==rActions[i])] = rLow #do not reward states that are never rewarded
		else:
			cReward = np.ones(nImages)*rCrit

		#concatenate state-action
		concInput[:,0:nDimStates] = images #States
		concInput[:,nDimStates:nDimStates+nDimActions] = cAction #Actions

		#shuffle input
		rndInput, rndLabel, cReward, rndIdx = ex.shuffle(concInput, labels, cReward)

		#compute activation of hid and class neurons
		hidNeurons = ex.propL1(rndInput, W_in, cReward)
		classNeurons = ex.propL2_learn(classes, rndLabel, cReward)

		#train network with mini-batches
		for b in range(int(nImages/nBatch)): #may leave a few training examples out
			bInput = rndInput[b*nBatch:(b+1)*nBatch,:]
			bHidNeurons = hidNeurons[b*nBatch:(b+1)*nBatch,:]
			bClassNeurons = classNeurons[b*nBatch:(b+1)*nBatch,:]
			
			#update weights
			W_in += ex.learningStep(bInput, bHidNeurons, W_in, lr)
			W_class += ex.learningStep(bHidNeurons, bClassNeurons, W_class, lr)

	fig = pl.plotRF(np.copy(W_in), e=str(e))
	pyplot.savefig('output/' + runName + '/RFs/RF_' + str(r).zfill(3))
	# pyplot.show(block=False)
	pyplot.close(fig)
	W_in_save[str(r).zfill(3)] = np.copy(W_in)
	W_class_save[str(r).zfill(3)] = np.copy(W_class)


ex.savedata(runName, W_in_save, W_class_save, seed, classes, rActions, dataset, A, nEpiCrit, nEpiAdlt, singleActiv, nImages, nDimStates, nDimActions, nHidNeurons, rHigh, rLow, np.round(lr, 5), nBatch, randActions)


































