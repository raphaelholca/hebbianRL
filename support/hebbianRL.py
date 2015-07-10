""" 
This function trains a hebbian neural network to learn a representation from the MNIST dataset. It makes use of a reward/relevance signal that increases the learning rate when the network makes a correct state-action pair selection.

Output is saved under RL/data/[runName]
"""

# from progressbar import ProgressBar
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

def RLnetwork(classes, rActions, nRun, nEpiCrit, nEpiDopa, t_hid, t_act, A, runName, dataset, nHidNeurons, lr, aHigh, aPairing, dHigh, dMid, dNeut, dLow, nBatch, classifier, SVM, bestAction, createOutput, showPlots, show_W_act, sort, target, seed, images, labels, kwargs):

	""" variable initialization """
	if createOutput: runName = ex.checkdir(runName, OW_bool=True) #create saving directory
	else: print " !!! ----- not saving data ----- !!! "
	images = ex.normalize(images, A*np.size(images,1)) #normalize input images
	W_in_save = {}
	W_act_save = {}
	nClasses = len(classes)
	_, idx = np.unique(rActions, return_index=True)
	lActions = rActions[np.sort(idx)]
	nEpiTot = nEpiCrit + nEpiDopa
	np.random.seed(seed)
	nImages = np.size(images,0)
	nInpNeurons = np.size(images,1)
	nActNeurons = nClasses
	ach_bal = 0.25 ##optimize

	""" training of the network """
	print 'training network...'
	for r in range(nRun):
		print 'run: ' + str(r+1)

		#randommly assigns a class with ACh release (used to run multiple runs of ACh)
		# if True: target, rActions, rActions, lActions = ex.rand_ACh(nClasses) ##

		#initialize network variables
		ach = np.zeros(nBatch)
		dopa = np.zeros(nBatch)
		W_in = np.random.random_sample(size=(nInpNeurons, nHidNeurons)) + 1.0
		if True: #initialize output neurons with already fixed connections
			W_act = np.zeros((nHidNeurons, nActNeurons))
			nHid_perClass = nHidNeurons/nClasses
			nHidNeurons-nHid_perClass*nClasses
			for c in range(nClasses): 
				W_act[nHid_perClass*c : nHid_perClass*(c+1), c]=1.
			for d,c in enumerate(np.arange(nHid_perClass*(c+1), nHidNeurons)):
				W_act[c, d]=1.
		else:
			W_act = (np.random.random_sample(size=(nHidNeurons, nActNeurons))/1000+1.0)/nHidNeurons
		# W_act_init = np.copy(W_act)
		perf_track = np.zeros((nActNeurons, 2))
		Q = np.zeros((nActNeurons, nActNeurons))

		choice_count = np.zeros((nClasses, nClasses))

		# pbar_epi = ProgressBar()
		# for e in pbar_epi(range(nEpiTot)):
		for e in range(nEpiTot):
			#shuffle input
			rndImages, rndLabels = ex.shuffle([images, labels])

			#train network with mini-batches
			for b in range(int(nImages/nBatch)):
				
				#select batch training images (may leave a few training examples out (< nBatch))
				bImages = rndImages[b*nBatch:(b+1)*nBatch,:]
				bLabels = rndLabels[b*nBatch:(b+1)*nBatch]

				#initialize batch variables
				ach = np.ones(nBatch)
				dopa = np.ones(nBatch)
				dW_in = 0.
				dW_act = 0.
				disinhib_Hid = np.zeros(nBatch)
				disinhib_Act = np.zeros(nBatch)
				
				#compute activation of hidden and classification neurons
				bHidNeurons = ex.propL1(bImages, W_in, SM=False)
				bActNeurons = ex.propL1(ex.softmax(bHidNeurons, t=t_hid), W_act, SM=False)

				#predicted best action
				bPredictActions = rActions[np.argmax(bActNeurons,1)]

				#add noise to activation of hidden neurons and compute lateral inhibition
				if not bestAction and (e >= nEpiCrit):
					bHidNeurons += np.random.uniform(0, 50, np.shape(bHidNeurons)) ##param explore, optimize
					bHidNeurons = ex.softmax(bHidNeurons, t=t_hid)
					bActNeurons = ex.propL1(bHidNeurons, W_act, SM=False)
				else:
					bHidNeurons = ex.softmax(bHidNeurons, t=t_hid)
				bActNeurons = ex.softmax(bActNeurons, t=t_act)
					
				#take action - either deterministically (predicted best) or stochastically (additive noise)			
				bActions = rActions[np.argmax(bActNeurons,1)]	
				bActions_idx = ex.val2idx(bActions, lActions)

				#compute reward and ach signal
				bReward = ex.compute_reward(ex.label2idx(classes, bLabels), np.argmax(bActNeurons,1))
				# pred_bLabels_idx = ex.val2idx(bPredictActions, lActions) ##same as bActions_idx for bestAction = True ??
				# ach, ach_labels = ex.compute_ach(perf_track, pred_bLabels_idx, aHigh=aHigh, rActions=None, aPairing=1.0) # make rActions=None or aPairing=1.0 to remove pairing

				#compute dopa signal and disinhibition based on training period
				if e < nEpiCrit:
					""" critical period """
					dopa = ex.compute_dopa(bPredictActions, bActions, bReward, dHigh=0.0, dMid=0.75, dNeut=0.0, dLow=-0.5)

					disinhib_Hid = ach
					disinhib_Act = dopa

				elif e >= nEpiCrit: 
					""" Dopa - perceptual learning """
					dopa = ex.compute_dopa(bPredictActions, bActions, bReward, dHigh=dHigh, dMid=dMid, dNeut=dNeut, dLow=dLow)
					# rPredicted = Q[ex.val2idx(bPredictActions, lActions), ex.val2idx(bActions, lActions)]
					# dopa = ex.compute_dopa_2(rPredicted, bReward, dHigh=dHigh, dMid=dMid, dLow=dLow)

					disinhib_Hid = ach*dopa
					# disinhib_Act = ex.compute_dopa(bPredictActions, bActions, bReward, dHigh=0.0, dMid=0.75, dNeut=0.0, dLow=-0.5) #continuous learning in L2
					disinhib_Act = np.zeros(nBatch) #no learning in L2 during perc_dopa.
					
					# choice_count[ex.val2idx(bPredictActions, lActions), ex.val2idx(bActions, lActions)] += 1 #used to check the approximation of probability matching in decision making
				
				#compute weight updates
				dW_in 	= ex.learningStep(bImages, 		bHidNeurons, W_in, 		lr=lr, disinhib=disinhib_Hid)
				dW_act 	= ex.learningStep(bHidNeurons, 	bActNeurons, W_act, 	lr=lr, disinhib=disinhib_Act)
			
				### for ach?
				# postNeurons_lr = bActNeurons * (lr * disinhib_Act[:,np.newaxis])
				# # dW_act = (np.dot((bHidNeurons * ach[:,np.newaxis]).T, postNeurons_lr) - np.sum(postNeurons_lr, 0) * W_act)
				# dW_act = (np.dot((bHidNeurons * ((1-ach_bal)+ach[:,np.newaxis]*ach_bal)).T, postNeurons_lr) - np.sum(postNeurons_lr, 0) * W_act)
				###

				#update weights
				W_in += dW_in
				# W_act += dW_act

				W_in = np.clip(W_in, 1e-10, np.inf)
				W_act = np.clip(W_act, 1e-10, np.inf)

				#update Q table
				Q = ex.Q_learn(Q, ex.val2idx(bPredictActions, lActions), ex.val2idx(bActions, lActions), bReward, Q_LR=0.01)

				# if np.isnan(W_in).any(): import pdb; pdb.set_trace()

			##to check Wact assignment after each episode:
			RFproba, _, _ = rf.hist(runName, {'000':W_in}, classes, nInpNeurons, images, labels, SVM=SVM, proba=False, output=False, show=False)
			correct_W_act = 0.	
			same = ex.labels2actionVal(np.argmax(RFproba[0],1), classes, rActions) == rActions[np.argmax(W_act,1)]
			correct_W_act += np.sum(same)
			correct_W_act/=len(RFproba)
			print 'correct action weights: ' + str(int(correct_W_act)) + '/' + str(int(nHidNeurons))

		#save weights
		W_in_save[str(r).zfill(3)] = np.copy(W_in)
		W_act_save[str(r).zfill(3)] = np.copy(W_act)

	""" compute network statistics and performance """

	#compute histogram of RF classes
	RFproba, RFclass, _ = rf.hist(runName, W_in_save, classes, nInpNeurons, images, labels, SVM=SVM, proba=False, output=createOutput, show=showPlots, lr_ratio=1.0, rel_classes=classes[rActions!='0'])
	#compute the selectivity of RFs
	# _, _, RFselec = rf.hist(runName, W_in_save, classes, nInpNeurons, images, labels, SVM=False, proba=False, output=False, show=False, lr_ratio=1.0)

	#compute correct weight assignment in the action layer
	correct_W_act = 0.
	notsame = {}
	for k in W_act_save.keys():
		same = ex.labels2actionVal(np.argmax(RFproba[int(k)],1), classes, rActions) == rActions[np.argmax(W_act_save[k],1)]
		notsame[k] = np.argwhere(~same)
		correct_W_act += np.sum(same)
	correct_W_act/=len(RFproba)

	#plot the weights
	if createOutput:
		if show_W_act: W_act_pass=W_act_save
		else: W_act_pass=None
		rf.plot(runName, W_in_save, RFproba, target=target, W_act=W_act_pass, sort=sort, notsame=notsame)

	#assess classification performance with neural classifier or SVM 
	if classifier=='actionNeurons':	allCMs, allPerf = cl.actionNeurons(runName, W_in_save, W_act_save, classes, rActions, nHidNeurons, nInpNeurons, A, dataset, output=createOutput, show=showPlots)
	if classifier=='SVM': 			allCMs, allPerf = cl.SVM(runName, W_in_save, images, labels, classes, nInpNeurons, A, dataset, output=createOutput, show=showPlots)
	if classifier=='neuronClass':	allCMs, allPerf = cl.neuronClass(runName, W_in_save, classes, RFproba, nInpNeurons, A, dataset, output=createOutput, show=showPlots)

	print '\ncorrect action weight assignment:\n ' + str(correct_W_act) + ' out of ' + str(nHidNeurons)+'.0'

	#save data
	if createOutput:
		ex.save_data(W_in_save, W_act_save, kwargs)

	print '\nrun: '+runName

	# import pickle
	# f = open('choice_count', 'w')
	# pickle.dump(choice_count, f)
	# f.close()
	# f = open('Q', 'w')
	# pickle.dump(Q, f)
	# f.close()

	import pdb; pdb.set_trace()


	return allCMs, allPerf, correct_W_act/nHidNeurons






























