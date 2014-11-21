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

classes 	= np.array([ 4 , 7 , 9 ], dtype=int)
rActions 	= np.array(['a','b','c'], dtype='|S1')

# classes 	= np.array([ 4 , 9 ], dtype=int)
# rActions 	= np.array(['a','b'], dtype='|S1')

""" parameters """
nRun 		= 1 				# number of runs
nEpiCrit	= 20				# number of 'critical period' episodes in each run (episodes when reward is not required for learning)
nEpiAdlt	= 20					# number of 'adult' episodes in each run (episodes when reward is not required for learning)
seed 		= 1 				# seed of the random number generator
A 			= 900 				# image normalization constant
runName 	= 'dopa'			# name of the folder where to save results
dataset 	= 'test'			# MNIST dataset to use; legal values: 'test', 'train'
singleActiv = 5.	 			# activation value of the action neurons
t 			= 1. 				# temperature parameter of the softmax function
nHidNeurons = 10				# number of hidden neurons
lrCrit		= 0.01 				# learning rate during 'critica period' (pre-training, nEpiCrit)
lrAdlt		= 0.0 				# learning rate after the end of the 'critica period' (adult/training, nEpiAdlt)
aHigh 		= 0.01#0.0005		# learning rate increase for relevance signal (high ACh) outside of critical period
aLow		= 0 				# learning rate increase without relevant signal (no ACh)
dHigh 		= 0.01				# learning rate increase for unexpected reward (high dopamine) outside of critical period
dNeut 		= 0.01				# learning rate increase for correct reward prediction (neutral dopamine)
dLow 		= 0.#-0.01/3			# learning rate increase for incorrect reward prediction (low dopamine)
nBatch 		= 60 				# mini-batch size
randActions = True				# whether to take random actions (True) or to take best possible action
classifier	= 'neural'			# which classifier to use for performance assessment. Possible values are: 'neural', 'SVM', 'neuronClass'
showPlots	= False				# whether to display plots
target		= None 				# target digit (to be used to color plots). Use 'None' if not desired
balReward	= False				# whether to insure that reward sums to the same value for stimuli that are always rewarded and those that are rewarded for specific actions

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
# if randActions: nEpiAdlt *= int(np.clip(len(lActions),1,np.inf)) ##
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
	# if trainNeuro: W_class = np.random.random_sample(size=(nHidNeurons, nClasses)) + 1.
	W_class = np.random.random_sample(size=(nHidNeurons, nClasses)) + 1.

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
				# cActionVal=ex.shuffle([cActionVal])[0]
			cActionIdx = ex.val2idx(cActionVal, lActions)
		else: cActionIdx, cActionVal = np.empty(nImages, dtype='|S1'), np.empty(nImages, dtype='|S1')

		#assign reward according to state-action pair, after the end of the critical period. In cReward, -1=never, 0=incorrect, 1=correct, 2=always
		cReward = np.zeros(nImages, dtype=int)
		if e >= nEpiCrit or True: ##remove, only for computational efficiency
			if e >= nEpiCrit: 
				lr = lrAdlt
				if nDimActions != 0: cAction[np.arange(nImages),cActionIdx] = singleActiv
			for i in range(nClasses):
				cReward[np.logical_and(labels==classes[i], cActionVal==rActions[i])] = 1 #aHigh #reward correct state-action pairs
				cReward[np.logical_and(labels==classes[i], '1'==rActions[i])] = 2 #aHigh/rewardDiv #reward states that are always rewarded
				cReward[np.logical_and(labels==classes[i], '0'==rActions[i])] = -1 #aLow #do not reward states that are never rewarded

		#concatenate state-action
		concInput[:,0:nDimStates] = images #States
		concInput[:,nDimStates:nDimStates+nDimActions] = cAction #Actions

		#shuffle input
		rndInput, rndLabels, rndReward, rndActionVal = ex.shuffle([concInput, labels, cReward, cActionVal])

		#train network with mini-batches
		for b in range(int(nImages/nBatch)): #may leave a few training examples out
			bInput = rndInput[b*nBatch:(b+1)*nBatch,:]
			bReward = rndReward[b*nBatch:(b+1)*nBatch]
			bLabels = rndLabels[b*nBatch:(b+1)*nBatch]
			bActionsVal = rndActionVal[b*nBatch:(b+1)*nBatch]
			
			#compute activation of hid and class neurons
			bHidNeurons = ex.propL1(bInput, W_in, t=t)
			bHidNeurons_noClass = ex.propL1(bInput[:,0:nDimStates], W_in[0:nDimStates,:], t=t)
			# bPredict = classes[np.argmax(ex.propL2_class(bHidNeurons_noClass, W_class),1)] ##now excludes class neurons for prediction
			bPredict = classes[np.argmax(ex.propL2_class(bHidNeurons, W_class),1)]
			bClassNeurons = ex.propL2_learn(classes, bLabels)

			ach = np.zeros(nBatch)
			dopa = np.zeros(nBatch)
			if e >= nEpiCrit:
				#determine acetylcholine strength based on reward
				ach[bReward==-1]=aLow				#never 
				ach[bReward== 0]=aLow				#incorred, or during critical period
				ach[bReward== 1]=aHigh				#correct
				ach[bReward== 2]=aHigh/rewardDiv	#always
			
				#determine dopamine signal strength based on reward
				bPredictActions = ex.labels2actionVal(bPredict, classes, rActions)
				dopa[np.logical_and(bPredictActions==bActionsVal, bReward==1)] 	= dHigh		#correct reward prediction
				dopa[np.logical_and(bPredictActions==bActionsVal, bReward==0)] 	= dLow		#incorrect reward prediction
				dopa[np.logical_and(bPredictActions!=bActionsVal, bReward==0)] 	= dLow		#correct no reward prediction
				dopa[np.logical_and(bPredictActions!=bActionsVal, bReward==1)] 	= dLow		#incorrect no reward prediction
				dopa[bReward==-1]												= dNeut		#never rewarded
				dopa[bReward== 2]												= dNeut		#always rewarded

			#update weights
			W_in += ex.learningStep(bInput, bHidNeurons, W_in, lr, dopa=dopa)
			W_in = np.clip(W_in,1.0,np.inf) #necessary if using negative lr
			W_class += ex.learningStep(bHidNeurons, bClassNeurons, W_class, lrCrit) ##may cause some differences between randActions True and False; the learning rate should be set to 0 when incorrect actions are taken (maybe, not sure...).

		print np.round(W_in[784:,:].T,1)
	#save weights
	W_in_save[str(r).zfill(3)] = np.copy(W_in)
	if trainNeuro: W_class_save[str(r).zfill(3)] = np.copy(W_class)

#compute histogram of RF classes
RFproba, _ = rf.hist(runName, W_in_save, classes, nDimStates, proba=False, show=showPlots)

#plot the weights
rf.plot(runName, W_in_save, RFproba, target)

#assess classification performance with neural classifier or SVM 
if classifier=='neural': cl.neural(runName, W_in_save, W_class_save, classes, rActions, nHidNeurons, nDimStates, A, dataset, show=showPlots)
if classifier=='SVM': cl.SVM(runName, W_in_save, images, labels, classes, nDimStates, A, dataset, show=showPlots)
if classifier=='neuronClass': cl.neuronClass(runName, W_in_save, classes, RFproba, nDimStates, A, show=showPlots)

#save data
ex.savedata(runName, W_in_save, W_class_save, seed, classes, rActions, lActions, dataset, A, nEpiCrit, nEpiAdlt, singleActiv, nImages, nDimStates, nDimActions, nHidNeurons, aHigh, aLow, np.round(lr, 5), nBatch, randActions, classifier)

print runName

































