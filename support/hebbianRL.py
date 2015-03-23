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
import support.svmutils as su
import sys

ex = reload(ex)
pl = reload(pl)
cl = reload(cl)
rf = reload(rf)
su = reload(su)

def RLnetwork(classes, rActions, nRun, nEpiCrit, nEpiAch, nEpiProc, nEpiDopa, A, runName, dataset, nHidNeurons, lrCrit, lrAdlt, aHigh, aLow, dMid, dHigh, dNeut, dLow, nBatch, classifier, SVM, bestAction, feedback, balReward, showPlots, show_W_act, sort, target, seed, images, labels, kwargs):

	""" variable initialization """
	runName = ex.checkdir(runName, OW_bool=True) #create saving directory
	images = ex.normalize(images, A*np.size(images,1)) #normalize input images
	W_in_save = {}
	W_act_save = {}
	target_save = {}
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

	""" training of the network """
	print 'training network...'
	for r in range(nRun):
		print 'run: ' + str(r+1)

		#randommly assigns a class with ACh release (used to run multiple runs of ACh)
		# if True: target, rActions, rActions_z, lActions = ex.rand_ACh(nClasses) ##

		#initialize network variables
		ach = np.zeros(nBatch)
		dopa = np.zeros(nBatch)
		W_in = np.random.random_sample(size=(nInpNeurons, nHidNeurons)) + 1.0
		W_act = (np.random.random_sample(size=(nHidNeurons, nActNeurons))/1000+1.0)/nHidNeurons
		W_act_init = np.copy(W_act)

		for e in range(nEpiTot):
			#shuffle input
			rndImages, rndLabels = ex.shuffle([images, labels])

			#train network with mini-batches
			for b in range(int(nImages/nBatch)): #may leave a few training examples out (< nBatch)
				bImages = rndImages[b*nBatch:(b+1)*nBatch,:]
				bLabels = rndLabels[b*nBatch:(b+1)*nBatch]
				
				#compute activation of hidden, action, and classification neurons
				bHidNeurons = ex.propL1(bImages, W_in, SM=False)
				bActNeurons = ex.propL1(ex.softmax(bHidNeurons, t=0.001), W_act, SM=False)

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
				disinhib_Hid = np.zeros(nBatch)
				disinhib_Act = np.zeros(nBatch)

				#compute reward, ach, and dopa based on learning period
				if e < nEpiCrit: #critical period
					lr_current = lrCrit 
					disinhib_Hid = np.ones(nBatch) #learning in L1 during crit. is w/o neuromodulation
					disinhib_Act = np.zeros(nBatch) #no learning in L1 during crit.

				elif e >= nEpiCrit and e < nEpiCrit + nEpiAch: #ACh - perceptual learning
					#determine acetylcholine strength based on task involvement
					ach[np.array([d.isupper() for d in ex.labels2actionVal(bLabels, classes, rActions)])] = aHigh			#stimulus involved in task

					lr_current = lrAdlt
					disinhib_Hid = ach
					disinhib_Act = np.zeros(nBatch) #no learning in L2 during perc.

				elif e >= nEpiCrit + nEpiAch and e < nEpiCrit + nEpiAch + nEpiProc: #procedural learning
					#assign reward according to state-action pair, after the end of the critical period. In bReward, -1=never, 0=incorrect, 1=correct, 2=always
					bReward = ex.compute_reward(bLabels, classes, bActions, rActions_z)
					
					#compute reward, and ach and dopa signals for procedural learning
					ach[np.array([d.isupper() for d in ex.labels2actionVal(bLabels, classes, rActions)])] = aHigh

					#determine dopamine signal strength based on reward
					dopa[np.logical_and(bPredictActions==bActions, bReward==1)] = dMid			#correct reward prediction
					dopa[np.logical_and(bPredictActions==bActions, bReward==0)] = dLow			#incorrect reward prediction
					dopa[np.logical_and(bPredictActions!=bActions, bReward==0)] = dNeut			#correct no reward prediction
					dopa[np.logical_and(bPredictActions!=bActions, bReward==1)] = dHigh			#incorrect no reward prediction
					dopa[bReward==-1]											= dNeut			#never rewarded
					dopa[bReward== 2]											= dNeut			#always rewarded

					lr_current = lrAdlt
					disinhib_Hid = np.zeros(nBatch) #no learning in L1 during proc.
					disinhib_Act = dopa

				elif e >= nEpiCrit + nEpiAch + nEpiProc: #Dopa - perceptual learning
					#assign reward according to state-action pair, after the end of the critical period. In bReward, -1=never, 0=incorrect, 1=correct, 2=always
					bReward = ex.compute_reward(bLabels, classes, bActions, rActions_z)
				
					#determine acetylcholine strength based on task involvement
					ach[np.array([d.isupper() for d in ex.labels2actionVal(bLabels, classes, rActions)])] = aHigh

					#determine dopamine signal strength based on reward
					dopa[np.logical_and(bPredictActions==bActions, bReward==1)] = dMid			#correct reward prediction
					dopa[np.logical_and(bPredictActions==bActions, bReward==0)] = dLow			#incorrect reward prediction
					dopa[np.logical_and(bPredictActions!=bActions, bReward==0)] = dNeut			#correct no reward prediction
					dopa[np.logical_and(bPredictActions!=bActions, bReward==1)] = dHigh			#incorrect no reward prediction
					dopa[bReward==-1]											= dNeut			#never rewarded
					dopa[bReward== 2]											= dNeut			#always rewarded


					#feedback from classification layer
					if feedback: 
						bFeedback = np.log(np.dot(bActNeurons, ex.softmax(W_act, t=0.001).T)*100+1)*10
						# bFeedback = np.dot(bActNeurons, ex.softmax(W_act, t=0.001).T)*100
						# bFeedback = np.log(np.dot(bActNeurons, W_act.T)*100+1)*10
						# bFeedback = np.log(np.dot(bActNeurons, W_act.T)*1e1)*10
						bHidNeurons += bFeedback

					lr_current = lrAdlt
					disinhib_Hid = ach*dopa
					disinhib_Act = dopa

				# lateral inhibition
				bHidNeurons = ex.softmax(bHidNeurons, t=0.001) #activation must be done after feedback is added to activity
				bActNeurons = ex.softmax(bActNeurons, t=0.001)
				
				#compute weight updates
				dW_in 	= ex.learningStep(bImages, 		bHidNeurons, W_in, 		lr=lr_current, disinhib=disinhib_Hid)
				# dW_act 	= ex.learningStep(bHidNeurons, 	bActNeurons, W_act, 	lr=lr_current, disinhib=disinhib_Act)
			
				###
				postNeurons_lr = bActNeurons * (lr_current * disinhib_Act[:,np.newaxis])
				dW_act = (np.dot((bHidNeurons * ach[:,np.newaxis]).T, postNeurons_lr) - np.sum(postNeurons_lr, 0) * W_act)
				###

				# W_in += dW_in
				W_in += dW_in
				W_act += dW_act

				# W_in = np.clip(W_in, 1e-10, np.inf)
				W_act = np.clip(W_act, 1e-10, np.inf)

		#save weights
		W_in_save[str(r).zfill(3)] = np.copy(W_in)
		W_act_save[str(r).zfill(3)] = np.copy(W_act)
		target_save[str(r).zfill(3)] = np.copy(target)

	""" compute network statistics and performance """

	#compute histogram of RF classes
	if nEpiAch>0: lr_ratio=aHigh/aLow
	else: lr_ratio=1.0
	RFproba, _, _ = rf.hist(runName, W_in_save, classes, nInpNeurons, images, labels, SVM=SVM, proba=False, show=showPlots, lr_ratio=lr_ratio, rel_classes=classes[rActions!='0'])
	#compute the selectivity of RFs
	_, _, RFselec = rf.hist(runName, W_in_save, classes, nInpNeurons, images, labels, SVM=False, proba=False, show=showPlots, lr_ratio=1.0)

	#compute correct weight assignment in the action layer
	correct_W_act = 0.
	notsame = {}
	for k in W_act_save.keys():
		same = ex.labels2actionVal(np.argmax(RFproba[int(k)],1), classes, rActions_z) == rActions_z[np.argmax(W_act_save[k],1)]
		notsame[k] = np.argwhere(~same)
		correct_W_act += np.sum(same)
	correct_W_act/=len(RFproba)

	#plot the weights
	if show_W_act: W_act_pass=W_act_save
	else: W_act_pass=None
	rf.plot(runName, W_in_save, RFproba, target=target_save, W_act=W_act_pass, sort=sort, notsame=notsame)

	#assess classification performance with neural classifier or SVM 
	if classifier=='actionNeurons':	allCMs, allPerf = cl.actionNeurons(runName, W_in_save, W_act_save, classes, rActions_z, nHidNeurons, nInpNeurons, A, dataset, show=showPlots)
	if classifier=='SVM': 			allCMs, allPerf = cl.SVM(runName, W_in_save, images, labels, classes, nInpNeurons, A, 'train', show=showPlots)
	if classifier=='neuronClass':	allCMs, allPerf = cl.neuronClass(runName, W_in_save, classes, RFproba, nInpNeurons, A, dataset, show=showPlots)

	# print '\nmean RF selectivity: \n' + str(np.round(RFselec[RFselec<np.inf],2))

	print '\ncorrect action weight assignment:\n ' + str(correct_W_act) + ' out of ' + str(nHidNeurons)+'.0'

	#save data
	ex.save_data(W_in_save, W_act_save, kwargs)

	print '\nrun: '+runName

	return allCMs, allPerf, correct_W_act/nHidNeurons






























