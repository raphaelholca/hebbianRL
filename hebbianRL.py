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
import support.svmutils as su
import sys

ex = reload(ex)
pl = reload(pl)
cl = reload(cl)
rf = reload(rf)
su = reload(su)

""" 
experimental variables

classes (int) 	: class of the MNIST dataset to use to train the network
rActions (str)	: for each class of MNIST, the action that is rewarded. '0' indicates a class that is never rewarded; '1' indicates a class that is always rewarded; chararcters (e.g., 'a', 'b', etc.) indicate the specific action that is rewarded.
"""

classes 	= np.array([ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
# rActions 	= np.array(['a','b','c','d','e','f','g','h','i','j'], dtype='|S1')
rActions 	= np.array(['0','0','0','0','e','0','0','0','0','0'], dtype='|S1')

# classes 	= np.array([ 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
# rActions 	= np.array(['a','0','0','0','0','0'], dtype='|S1')

# classes 	= np.array([ 4 , 7 , 9 ], dtype=int)
# rActions 	= np.array(['a','0','0'], dtype='|S1')

# classes 	= np.array([ 4 , 9 ], dtype=int)
# rActions 	= np.array(['a','b'], dtype='|S1')

# dHigh_list = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75]
# for dd in dHigh_list:
# 	print '\n\n==============================================================\n\n'
# runName 	= 'selec-0_' + str(int(dd*100))

""" parameters """
nRun 		= 1			# number of runs
nEpiCrit	= 2				# number of 'critical period' episodes in each run (episodes when reward is not required for learning)
nEpiAch		= 0				# number of ACh episodes in each run (episodes when ACh only is active)
nEpiProc	= 1				# number of 'procedural learning' episodes (to initialize the action weights after critical period)
nEpiDopa	= 0				# number of 'adult' episodes in each run (episodes when reward is not required for learning)
A 			= 1.2			# input normalization constant. Will be used as: (input size)*A; for images: 784*1.2=940.8
runName 	= 'proc+dopa'			# name of the folder where to save results
dataset 	= 'train'		# MNIST dataset to use; legal values: 'test', 'train' ##use train for actual results
nHidNeurons = 49			# number of hidden neurons
lrCrit		= 0.005 		# learning rate during 'critica period' (pre-training, nEpiCrit)
lrAdlt		= 0.005		# learning rate after the end of the 'critica period' (adult/training, nEpiAch and nEpiDopa)
aHigh 		= 6.			# learning rate increase for relevance signal (high ACh) outside of critical period
aLow		= 1. 			# learning rate increase without relevant signal (no ACh)
dHigh 		= 0.5			# learning rate increase for unexpected reward (high dopamine) outside of critical period
dNeut 		= 0.0			# learning rate increase for no reward, when none predicted
dLow 		= -dHigh/1.5		# learning rate increase for incorrect reward prediction (low dopamine)
nBatch 		= 20 			# mini-batch size
classifier	= 'neuronClass'			# which classifier to use for performance assessment. Possible values are: 'neural', 'SVM', 'neuronClass'
SVM			= False			# whether to use an SVM or the number of stimuli that activate a neuron to determine the class of the neuron
bestAction 	= True			# whether to take predicted best action (True) or take random actions (False)
feedback	= True			# whether to feedback activation of classification neurons to hidden neurons
balReward	= False			# whether reward should sum to the same value for stim. that are always rewarded and stim. that are rewarded for specific actions
showPlots	= False			# whether to display plots
show_W_act	= True			# whether to display W_act weights on the weight plots
sort 		= False			# whether to sort weights by their class when displaying
target		= 4 			# target digit (to be used to color plots). Use None if not desired
seed 		= 992#np.random.randint(1000) 				# seed of the random number generator


""" load and pre-process images """
ex.checkClassifier(classifier)
runName = ex.checkdir(runName, OW_bool=True)
print 'seed: ' + str(seed) + '\n'
print 'loading data...'
imPath = '/Users/raphaelholca/Documents/data-sets/MNIST'

images, labels = mnist.read_images_from_mnist(classes = classes, dataset = dataset, path = imPath)
images = ex.normalize(images, A*np.size(images,1))
images, labels = ex.evenLabels(images, labels, classes)

# if True: #duplicates some of the training examples, to be used instead to ach
# 	labels_ori = np.copy(labels)
# 	images_ori = np.copy(images)
# 	for i in range(3):	#number of times to duplicate the training examples
# 		for d in [4]: #digit classes to duplicate
# 			images = np.append(images, images_ori[labels_ori==d])
# 			images = np.reshape(images, (-1, 784))
# 			labels = np.append(labels, labels_ori[labels_ori==d])

""" variable initialization """
W_in_save = {}
W_act_save = {}
W_class_save = {}
nClasses = len(classes)
rActions_z = np.copy(rActions)
rActions_z[np.logical_or(rActions=='0', rActions=='1')] = 'z'
_, idx = np.unique(rActions_z, return_index=True)
lActions = rActions_z[np.sort(idx)] #legal actions, with order maintained, with 'z' for all classes with '0' and '1'
nEpiTot = nEpiCrit + nEpiAch + nEpiProc + nEpiDopa
np.random.seed(seed)
nImages = np.size(images,0)
nInpNeurons = np.size(images,1)
nActNeurons = nClasses
trainNeuro = np.where(classifier == 'neural', True, False)

""" training of the network """
print 'training network...'
for r in range(nRun):
	print 'run: ' + str(r+1)
	#initialize network variables
	ach = np.zeros(nBatch)
	dopa = np.zeros(nBatch)
	W_in = np.random.random_sample(size=(nInpNeurons, nHidNeurons))+1.0
	W_act = (np.random.random_sample(size=(nHidNeurons, nActNeurons))+1.0)/nHidNeurons
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
			bActNeurons = ex.propL1(ex.softmax(bHidNeurons, t=0.001), W_act, t=0.001)
			if trainNeuro: bClassNeurons = ex.propL2_learn(classes, bLabels)

			#take action - either random or predicted best
			bPredictActions = rActions_z[np.argmax(bActNeurons,1)] #predicted best action
			if bestAction: bActions = np.copy(bPredictActions) #predicted best action taken
			else: #random action taken
				bActions = np.random.choice(lActions, size=nBatch) 
				bActNeurons = np.ones_like(bActNeurons)*1e-4 #reset neuron activation
				bActNeurons[np.arange(nBatch), ex.val2idx(bActions, lActions)]=1. #activate the action neuron corresponding to the action taken

			ach = np.ones(nBatch)*aLow
			dopa = np.ones(nBatch)*dNeut
			dW_in = 0.
			dW_act = 0.

			#compute reward, ach, and dopa based on learning period
			if e < nEpiCrit: #critical period
				lr_current = lrCrit 
				disinhib_L1 = np.ones(nBatch) #learning in L1 during crit. is w/o neuromodulation
				disinhib_L2 = np.zeros(nBatch) #no learning in L1 during crit.

			elif e >= nEpiCrit and e < nEpiCrit + nEpiAch: #ACh - perceptual learning
				#determine acetylcholine strength based on task involvement
				ach[ex.labels2actionVal(bLabels, classes, rActions)!='0'] = aHigh			#stimulus involved in task

				lr_current = lrAdlt
				disinhib_L1 = ach
				disinhib_L2 = np.zeros(nBatch) #no learning in L2 during perc.

			elif e >= nEpiCrit + nEpiAch and e < nEpiCrit + nEpiAch + nEpiProc : #procedural learning
				#assign reward according to state-action pair, after the end of the critical period. In bReward, -1=never, 0=incorrect, 1=correct, 2=always
				bReward = ex.compute_reward(bLabels, classes, bActions, rActions_z)
				
				#compute reward, and ach and dopa signals for procedural learning
				if nEpiAch > 0:
					ach[ex.labels2actionVal(bLabels, classes, rActions)!='0'] = aHigh

				#determine dopamine signal strength based on reward
				dopa[bReward==1] = dHigh
				dopa[bReward==0] = dLow

				lr_current = lrAdlt
				disinhib_L1 = np.zeros(nBatch) #no learning in L1 during proc.
				disinhib_L2 = ach*dopa

			elif e >= nEpiCrit + nEpiAch + nEpiProc: #Dopa - perceptual learning
				#assign reward according to state-action pair, after the end of the critical period. In bReward, -1=never, 0=incorrect, 1=correct, 2=always
				bReward = ex.compute_reward(bLabels, classes, bActions, rActions_z)
			
				#determine acetylcholine strength based on task involvement
				if nEpiAch > 0:
					ach[ex.labels2actionVal(bLabels, classes, rActions)!='0'] = aHigh

				#determine dopamine signal strength based on reward
				dopa[bReward==1] 	= dHigh 	#correct action
				dopa[bReward==0] 	= dLow 	#incorrect action
				dopa[bReward==-1]	= dNeut		#never rewarded
				dopa[bReward== 2]	= dNeut		#always rewarded

				#feedback from classification layer
				if feedback: 
					bFeedback = np.dot(bActNeurons, W_act.T)*100
					bHidNeurons += np.log(bFeedback)

				lr_current = lrAdlt
				disinhib_L1 = ach*dopa
				disinhib_L2 = np.zeros(nBatch) #no learning in L2 during perc.

			bHidNeurons = ex.softmax(bHidNeurons, t=0.001) ##t? ##*A*nHidNeurons? #activation must be done after feedback is added to activity

			#compute weight updates
			dW_in = ex.learningStep(bImages, bHidNeurons, W_in, lr=lr_current, disinhib=disinhib_L1)
			dW_act = ex.learningStep(bHidNeurons, bActNeurons, W_act, lr=lr_current, disinhib=disinhib_L2)
			W_in += dW_in
			W_act += dW_act

			# W_in = np.clip(W_in, 1e-10, np.inf)
			W_act = np.clip(W_act, 1e-10, np.inf)

			if trainNeuro: W_class += ex.learningStep(bHidNeurons, bClassNeurons, W_class, lrCrit)

			# if not (W_act<np.inf).all():
			# if e==2 and b==177:
			# 	import pdb; pdb.set_trace()

	#save weights
	W_in_save[str(r).zfill(3)] = np.copy(W_in)
	W_act_save[str(r).zfill(3)] = np.copy(W_act)
	if trainNeuro: W_class_save[str(r).zfill(3)] = np.copy(W_class)

""" compute network statistics and performance """

#compute histogram of RF classes
if nEpiAch>0: lr_ratio=aHigh/lrAdlt
else: lr_ratio=1.0
RFproba, _, _ = rf.hist(runName, W_in_save, classes, nInpNeurons, images, labels, SVM=SVM, proba=False, show=showPlots, lr_ratio=lr_ratio, rel_classes=classes[rActions!='0'])
#compute the selectivity of RFs
_, _, RFselec = rf.hist(runName, W_in_save, classes, nInpNeurons, images, labels, SVM=False, proba=False, show=showPlots, lr_ratio=1.0)

#compute correct weight assignment in the action layer
correct_W_act = 0.
for k in W_act_save.keys():
	correct_W_act += np.sum(ex.labels2actionVal(np.argmax(RFproba[int(k)],1), classes, rActions_z) == rActions_z[np.argmax(W_act_save[k],1)])
correct_W_act/=len(RFproba)

#plot the weights
if show_W_act: W_act_pass=W_act_save
else: W_act_pass=None
rf.plot(runName, W_in_save, RFproba, target, W_act=W_act_pass, sort=sort)

#assess classification performance with neural classifier or SVM 
if classifier=='neural': cl.neural(runName, W_in_save, W_class_save, classes, rActions, nHidNeurons, nInpNeurons, A, dataset, show=showPlots)
if classifier=='SVM': cl.SVM(runName, W_in_save, images, labels, classes, nInpNeurons, A, 'train', show=showPlots)
if classifier=='neuronClass': cl.neuronClass(runName, W_in_save, classes, RFproba, nInpNeurons, A, show=showPlots)

print '\nmean RF selectivity: \n' + str(np.round(RFselec[RFselec<np.inf],2))

print '\ncorrect action weight assignment:\n ' + str(correct_W_act) + ' out of ' + str(nHidNeurons)+'.0'

#save data
ex.save_data(runName, W_in_save, W_act_save, W_class_save, seed, nRun, classes, rActions, dataset, A, nEpiCrit, nEpiProc, nEpiAch, nEpiDopa, nHidNeurons, lrCrit, lrAdlt, aHigh, aLow, dHigh, dNeut, dLow, nBatch, bestAction, feedback, SVM, classifier)

print '\nrun: '+runName






























