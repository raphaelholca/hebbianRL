""" 
This function trains a hebbian neural network to learn a representation from the MNIST dataset. It makes use of a reward/relevance signal that increases the learning rate when the network makes a correct state-action pair selection.

Output is saved under RL/data/[runName]
"""

import numpy as np
import matplotlib.pyplot as pyplot
import support.external as ex
import support.plots as pl
import support.classifier as cl
import support.assessRF as rf
import support.mnist as mnist
import pdb
ex = reload(ex)
pl = reload(pl)
cl = reload(cl)
rf = reload(rf)

""" 
experimental variables

classes (int) 	: class of the MNIST dataset to use to train the network
rActions (str)	: for each class of MNIST, the action that is rewarded. '0' indicates a class that is never rewarded; '1' indicates a class that is always rewarded; chararcters (e.g., 'a', 'b', etc.) indicate the specific action that is rewarded.
"""
# classes 	= np.array([ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
# rActions 	= np.array(['0','0','0','0','1','0','0','0','0','0'], dtype='|S1')

# classes 	= np.array([ 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
# rActions 	= np.array(['0','0','0','0','0','1'], dtype='|S1')

# classes 	= np.array([ 4 , 7 , 9 ], dtype=int)
# rActions 	= np.array(['a','b','b'], dtype='|S1')

classes 	= np.array([ 4 , 9 ], dtype=int)
rActions 	= np.array(['a','b'], dtype='|S1')

""" parameters """
nRun 		= 20 				# number of runs
nEpiCrit	= 20				# number of 'critical period' episodes in each run (episodes when reward is not required for learning)
nEpiAdlt	= 20				# number of 'adult' episodes in each run (episodes when reward is not required for learning)
seed 		= None 				# seed of the random number generator
A 			= 900 				# image normalization constant
runName 	= 't-2'				# name of the folder where to save results
dataset 	= 'test'			# MNIST dataset to use; legal values: 'test', 'train'
singleActiv = 50.	 			# activation value of the action neurons
nHidNeurons = 4					# number of hidden neurons
lrCrit		= 0.01 				# learning rate during 'critica period' (pre-training, nEpiCrit)
lrAdlt		= 0.0 				# learning rate after the end of the 'critica period' (adult/training, nEpiADlt)
aHigh 		= 0.01#0.0005			# learning rate increase for relevance signal (high ACh) outside of critical period
aLow		= 0 				# learning rate increase without relevant signal (no ACh)
dHigh 		= 0.01				# learning rate increase for unexpected reward (high dopamine) outside of critical period
dNeut 		= 0.0				# learning rate increase for correct reward prediction (neutral dopamine)
dLow 		= -0.001			# learning rate increase for incorrect reward prediction (low dopamine)
nBatch 		= 60 				# mini-batch size
randActions = False				# whether to take random actions (True) or to take best possible action
classifier	= 'neural'			# which classifier to use for performance assessment. Possible values are: 'neural', 'SVM', 'neuronClass'
showPlots	= False				# whether to display plots
target		= None 				# target digit (to be used to color plots). Use 'None' if not desired
balReward	= False				# whether to insure that reward sums to the same value for stimuli that are always rewarded and those that are rewarded for specific actions

print randActions

""" load and pre-process images """
ex.checkClassifier(classifier)
runName = ex.checkdir(runName)
print "training network..."
imPath = '../data-sets/MNIST'
images, labels = mnist.read_images_from_mnist(classes = classes, dataset = dataset, path = imPath)
images = ex.normalize(images, A)
images, labels = ex.evenLabels(images, labels, classes)

""" variable initialization """
W_in_save = {}
W_class_save = {}
nClasses = len(classes)
lActions = np.unique(rActions[np.logical_and(rActions!='0', rActions!='1')]) #legal actions
# lActions = ['a', 'b']
if randActions: nEpiAdlt *= len(lActions)
nEpiTot = nEpiCrit + nEpiAdlt
np.random.seed(seed)
nImages = np.size(images,0)
nDimStates = np.size(images,1)
nDimActions = len(lActions)
nInpNeurons = nDimStates + nDimActions
concInput = np.zeros((nImages, nInpNeurons)) #concatenated input vector with state, action
trainNeuro = np.where(classifier == 'neural', True, False)
rewardDiv = np.where(balReward, len(lActions), 1.)

""" training of the network """
for r in range(nRun):
	print 'run: ' + str(r+1)
	#initialize network variables
	lr = lrCrit
	W_in = np.random.random_sample(size=(nInpNeurons, nHidNeurons)) + 1.
	if trainNeuro: W_class = np.random.random_sample(size=(nHidNeurons, nClasses)) + 1.

	for e in range(nEpiTot):
		#reset reward-action variables
		cAction = np.ones((nImages, nDimActions))
		cReward = np.ones(nImages)

		#(randomly) pick actions
		if nDimActions != 0:
			if randActions:
				cActionVal = np.random.choice(lActions, size=nImages)
			else:
				cActionVal = ex.labels2actionVal(labels, classes, rActions)	
			cActionIdx = ex.val2idx(cActionVal, lActions)
		else: cActionIdx, cActionVal = [], []

		#assign reward according to state-action pair, after the end of the critical period
		if e >= nEpiCrit:
			if nDimActions != 0: cAction[np.arange(nImages),cActionIdx] = singleActiv
			lr = lrAdlt
			cReward = np.ones(nImages)*aLow
			for i in range(nClasses):
				cReward[np.logical_and(labels==classes[i], cActionVal==rActions[i])] = aHigh #reward correct state-action pairs
				cReward[np.logical_and(labels==classes[i], '1'==rActions[i])] = aHigh/rewardDiv #reward states that are always rewarded
				cReward[np.logical_and(labels==classes[i], '0'==rActions[i])] = aLow #do not reward states that are never rewarded
		else:
			cReward = np.zeros(nImages)

		#concatenate state-action
		concInput[:,0:nDimStates] = images #States
		concInput[:,nDimStates:nDimStates+nDimActions] = cAction #Actions

		#shuffle input
		rndInput, rndLabels, rndReward, rndIdx = ex.shuffle(concInput, labels, cReward)

		#compute activation of hid and class neurons
		hidNeurons = ex.propL1(rndInput, W_in)
		if trainNeuro: classNeurons = ex.propL2_learn(classes, rndLabels)

		#train network with mini-batches
		for b in range(int(nImages/nBatch)): #may leave a few training examples out
			bInput = rndInput[b*nBatch:(b+1)*nBatch,:]
			bReward = rndReward[b*nBatch:(b+1)*nBatch]
			bHidNeurons = hidNeurons[b*nBatch:(b+1)*nBatch,:]
			if trainNeuro: bClassNeurons = classNeurons[b*nBatch:(b+1)*nBatch,:]
			
			#update weights
			W_in += ex.learningStep(bInput, bHidNeurons, W_in, lr, ach=bReward)
			W_in = np.clip(W_in,1.0,np.inf) #necessary if using negative lr
			if trainNeuro: W_class += ex.learningStep(bHidNeurons, bClassNeurons, W_class, lr)

	#save weights
	W_in_save[str(r).zfill(3)] = np.copy(W_in)
	if trainNeuro: W_class_save[str(r).zfill(3)] = np.copy(W_class)

#compute histogram of RF classes
RFproba, _ = rf.hist(runName, W_in_save, classes, nDimStates, proba=False, show=showPlots)

#plot the weights
rf.plot(runName, W_in_save, RFproba, target)

#assess classification performance with neural classifier or SVM 
if classifier=='neural': cl.neural(runName, W_in_save, W_class_save, classes, rActions, nHidNeurons, nDimStates, A, show=showPlots)
if classifier=='SVM': cl.SVM(runName, W_in_save, images, labels, classes, nDimStates, A, dataset, show=showPlots)
if classifier=='neuronClass': cl.neuronClass(runName, W_in_save, classes, RFproba, nDimStates, A, show=showPlots)

#save data
ex.savedata(runName, W_in_save, W_class_save, seed, classes, rActions, dataset, A, nEpiCrit, nEpiAdlt, singleActiv, nImages, nDimStates, nDimActions, nHidNeurons, aHigh, aLow, np.round(lr, 5), nBatch, randActions, classifier)


































