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
import sys
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
# rActions 	= np.array(['0','0','0','0','a','0','0','0','0','0'], dtype='|S1')

classes 	= np.array([ 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
rActions 	= np.array(['a','0','0','0','0','0'], dtype='|S1')

# classes 	= np.array([ 4 , 7 , 9 ], dtype=int)
# rActions 	= np.array(['c','0','b'], dtype='|S1')

# classes 	= np.array([ 4 , 9 ], dtype=int)
# rActions 	= np.array(['b','a'], dtype='|S1')

""" parameters """
nRun 		= 1				# number of runs
nEpiCrit	= 10			# number of 'critical period' episodes in each run (episodes when reward is not required for learning)
nEpiProc	= 3				# number of 'procedural learning' episodes (to initialize the action weights after critical period)
nEpiAdlt	= 0				# number of 'adult' episodes in each run (episodes when reward is not required for learning)
A 			= 900			# image normalization constant
runName 	= 'proc'		# name of the folder where to save results
dataset 	= 'test'		# MNIST dataset to use; legal values: 'test', 'train' ##use train for actual results
nHidNeurons = 49			# number of hidden neurons
lrCrit		= 0.01 			# learning rate during 'critica period' (pre-training, nEpiCrit)
lrAdlt		= 0.01			# learning rate after the end of the 'critica period' (adult/training, nEpiAdlt)
ach_bool	= True			# whether to use ACh signalling
aHigh 		= 0.05			# learning rate increase for relevance signal (high ACh) outside of critical period
aLow		= 0 			# learning rate increase without relevant signal (no ACh)
dopa_bool	= True			# whether to use dopa signalling
dHigh 		= 0.01			# learning rate increase for unexpected reward (high dopamine) outside of critical period
dMid 		= dHigh/3.		# learning rate increase for correct reward prediction
dNeut 		= 0.0			# learning rate increase for no reward, when none predicted
dLow 		= -dHigh*4		# learning rate increase for incorrect reward prediction (low dopamine)
nBatch 		= 20 			# mini-batch size
classifier	= 'neuronClass'	# which classifier to use for performance assessment. Possible values are: 'neural', 'SVM', 'neuronClass'
bestAction 	= False			# whether to take predicted best action (True) or take random actions (False)
feedback	= False			# whether to feedback activation of classification neurons to hidden neurons
balReward	= False			# whether reward should sum to the same value for stim. that are always rewarded and stim. that are rewarded for specific actions
showPlots	= False			# whether to display plots
show_W_act	= True			# whether to display W_act weights on the weight plots
sort 		= False			# whether to sort weights by their class when displaying
target		= None 			# target digit (to be used to color plots). Use 'None' if not desired
seed 		= 771#np.random.randint(1000) 				# seed of the random number generator

##cannot take random actions (bestAction=False) when using feedback with ACh because taking random actions leads to incorrect/meaningless feedback that is still paired with learning (ACh signal). Should this be solved? How?

""" load and pre-process images """
ex.checkClassifier(classifier)
runName = ex.checkdir(runName, OW_bool=True)
print 'seed: ' + str(seed) + '\n'
print 'training network...'
imPath = '/Users/raphaelholca/Documents/data-sets/MNIST'
images, labels = mnist.read_images_from_mnist(classes = classes, dataset = dataset, path = imPath)
images = ex.normalize(images, A)
images, labels = ex.evenLabels(images, labels, classes)

""" variable initialization """
W_in_save = {}
W_act_save = {}
W_class_save = {}
nClasses = len(classes)
rActions_z = np.copy(rActions)
rActions_z[np.logical_or(rActions=='0', rActions=='1')] = 'z'
_, idx = np.unique(rActions_z, return_index=True)
lActions = rActions_z[np.sort(idx)] #legal actions, with order maintained, with 'z' for all classes with '0' and '1'
nEpiTot = nEpiCrit + nEpiProc + nEpiAdlt
np.random.seed(seed)
nImages = np.size(images,0)
nDimStates = np.size(images,1)
nDimActions = len(lActions)
trainNeuro = np.where(classifier == 'neural', True, False)

""" training of the network """
for r in range(nRun):
	print 'run: ' + str(r+1)
	#initialize network variables
	lr = lrCrit
	ach = np.zeros(nBatch)
	dopa = np.zeros(nBatch)
	W_in = np.random.random_sample(size=(nDimStates, nHidNeurons)) + 1.0
	W_act = np.random.random_sample(size=(nHidNeurons, nDimActions))/20.
	if trainNeuro: W_class = np.random.random_sample(size=(nHidNeurons, nClasses)) + 1.

	for e in range(nEpiTot):
		#shuffle input
		rndImages, rndLabels = ex.shuffle([images, labels])

		#train network with mini-batches
		for b in range(int(nImages/nBatch)): #may leave a few training examples out (< nBatch)
			bImages = rndImages[b*nBatch:(b+1)*nBatch,:]
			bLabels = rndLabels[b*nBatch:(b+1)*nBatch]
			
			#compute activation of hidden, action, and classification neurons
			bHidNeurons = ex.propL1(bImages, W_in, SM=False)
			bActNeurons = ex.propL1(ex.softmax(bHidNeurons), W_act)
			if trainNeuro: bClassNeurons = ex.propL2_learn(classes, bLabels)

			#take action - either random or predicted best
			bPredictActions = lActions[np.argmax(bActNeurons,1)] #predicted best action
			if bestAction: bActions = np.copy(bPredictActions) #predicted best action taken
			else: bActions = np.random.choice(lActions, size=nBatch) #random action taken
			bActNeurons = np.ones_like(bActNeurons)*1e-4 #reset neuron activation
			bActNeurons[np.arange(nBatch), ex.val2idx(bActions, lActions)]=1. #activate the action neuron corresponding to the action taken

			if e >= nEpiCrit and e < nEpiCrit + nEpiProc: #procedural learning
				lr = 0.0
				ach = np.ones(nBatch)*aLow
				dopa = np.ones(nBatch)*dNeut

				#compute reward and dopa signal
				bReward = ex.compute_reward(bLabels, classes, bActions, rActions_z)
				dopa[bReward==1] = dHigh

			elif e >= nEpiCrit + nEpiProc: #perceptual learning (adult)
				lr = lrAdlt
				ach = np.ones(nBatch)*aLow
				dopa = np.ones(nBatch)*dNeut

				#assign reward according to state-action pair, after the end of the critical period. In bReward, -1=never, 0=incorrect, 1=correct, 2=always
				bReward = ex.compute_reward(bLabels, classes, bActions, rActions)
				
				#determine acetylcholine strength based on task involvement
				if ach_bool and e >= (nEpiCrit + nEpiProc): #no ach until after procedural training
					ach[ex.labels2actionVal(bLabels, classes, rActions)!='0'] = aHigh			#stimulus involved in task
			
				#determine dopamine signal strength based on reward
				if dopa_bool:
					dopa[np.logical_and(bPredictActions==bActions, bReward==1)] = dMid			#correct reward prediction
					dopa[np.logical_and(bPredictActions==bActions, bReward==0)] = dLow			#incorrect reward prediction
					dopa[np.logical_and(bPredictActions!=bActions, bReward==0)] = dNeut			#correct no reward prediction
					dopa[np.logical_and(bPredictActions!=bActions, bReward==1)] = dHigh			#incorrect no reward prediction
					dopa[bReward==-1]											= dNeut			#never rewarded
					dopa[bReward== 2]											= dNeut			#always rewarded

				#feedback from classification layer
				if feedback: bHidNeurons += ex.propL1(bActNeurons, ex.softmax(W_act, t=0.01).T)*100 ##feeding through softmax makes feedback even for all hidden neurons that project to the top neurons (i.e., feedback weights are either 1 or 0) ##compare with Pieter Roelfsema's work

			# bHidNeurons = ex.softmax(bHidNeurons)

			#update weights
			if e < (nEpiCrit + nEpiProc): dW_in = ex.learningStep(bImages, ex.softmax(bHidNeurons), W_in, lr) #no neurmodulators before adult in L1
			else: dW_in = ex.learningStep(bImages, ex.softmax(bHidNeurons), W_in, lr, ach=ach, dopa=dopa)
			W_in += dW_in
			W_in = np.clip(W_in,1.0,np.inf) #necessary if using negative lr
			if e >= nEpiCrit or True: 
				dW_act = ex.learningStep(ex.softmax(bHidNeurons, t=0.001), bActNeurons, W_act, lr, ach=ach, dopa=dopa)
				W_act += dW_act #only learn action weights after the end of the critical period
				W_act = np.clip(W_act,1e-10,np.inf) #necessary if using negative lr
			if trainNeuro: W_class += ex.learningStep(ex.softmax(bHidNeurons), bClassNeurons, W_class, lrCrit)
		
	#save weights
	W_in_save[str(r).zfill(3)] = np.copy(W_in)
	W_act_save[str(r).zfill(3)] = np.copy(W_act)
	if trainNeuro: W_class_save[str(r).zfill(3)] = np.copy(W_class)

""" compute network statistics and performance """

#compute histogram of RF classes
RFproba, _ = rf.hist(runName, W_in_save, classes, nDimStates, proba=False, show=showPlots)

#compute correct weight assignment in the action layer
correct_W_act = 0.
for k in W_act_save.keys():
	# correct_W_act += np.sum(np.argmax(RFproba[int(k)],1)==classes[np.argmax(W_act_save[k],1)])
	correct_W_act += np.sum(ex.labels2actionVal(np.argmax(RFproba[int(k)],1), classes, rActions_z) == lActions[np.argmax(W_act_save[k],1)])
correct_W_act/=len(RFproba)

#plot the weights
if show_W_act: W_act_pass=W_act_save
else: W_act_pass=None
rf.plot(runName, W_in_save, RFproba, target, W_act=W_act_pass, sort=sort)

#assess classification performance with neural classifier or SVM 
if classifier=='neural': cl.neural(runName, W_in_save, W_class_save, classes, rActions, nHidNeurons, nDimStates, A, dataset, show=showPlots)
if classifier=='SVM': cl.SVM(runName, W_in_save, images, labels, classes, nDimStates, A, dataset, show=showPlots)
if classifier=='neuronClass': cl.neuronClass(runName, W_in_save, classes, RFproba, nDimStates, A, show=showPlots)

print '\ncorrect action weight assignment: \n' + str(correct_W_act) + ' out of ' + str(nHidNeurons)+'.0'

#save data
ex.save_data(runName, W_in_save, W_act_save, W_class_save, seed, nRun, classes, rActions, dataset, A, nEpiCrit, nEpiProc, nEpiAdlt, nHidNeurons, lrCrit, lrAdlt, ach_bool, aHigh, aLow, dopa_bool, dHigh, dMid, dNeut, dLow, nBatch, bestAction, feedback, classifier)

print '\nrun: '+runName






























