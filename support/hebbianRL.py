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
import support.grating as gr
import support.bayesian_decoder as bc
import sys
import time
import pickle

ex = reload(ex)
pl = reload(pl)
cl = reload(cl)
rf = reload(rf)
su = reload(su)
gr = reload(gr)
bc = reload(bc)


def RLnetwork(	images, labels, orientations, 
				images_test, labels_test, orientations_test, 
				images_task, labels_task, orientations_task,
				nn_regressor, kwargs, 
				classes, rActions, nRun, nEpiCrit, nEpiDopa, t_hid, t_act, A, runName, dataset, nHidNeurons, lim_weights, lr, e_greedy, epsilon, noise_std, proba_predict, exploration, pdf_method, aHigh, aPairing, dHigh, dMid, dNeut, dLow, nBatch, protocol, target_ori, excentricity, noise_crit, noise_train, noise_test, im_size, classifier, param_xplr, pre_train, test_each_epi, SVM, save_data, verbose, show_W_act, sort, target, seed, comment):

	""" variable initialization """
	if save_data: runName = ex.checkdir(runName, OW_bool=True) #create saving directory
	W_in_save = {}
	W_act_save = {}
	perf_save = {}
	nClasses = len(classes)
	nEpiTot = nEpiCrit + nEpiDopa
	nImages = np.size(images,0)
	nInpNeurons = np.size(images,1)
	nActNeurons = nClasses
	train_class_layer = False if classifier=='bayesian' else True
	show_W_act = False if classifier=='bayesian' else show_W_act
	proba_predict = False if classifier!='bayesian' else proba_predict
	nn_input = np.empty((0,2), dtype=float) #input to the regressor neural net; nn_input[:,0] is prediction_error; nn_input[:,1] is tried DA value 

	""" training of the network """
	if verbose: 
		print 'run:  ' + runName
		print '\ntraining network...'
	for r in range(nRun):
		np.random.seed(seed+r)
		if verbose: print '\nrun: ' + str(r+1)

		#initialize network variables
		ach = np.zeros(nBatch)
		dopa = np.zeros(nBatch)
		""" load pre-trained weights """
		
		if pre_train!='' and pre_train!='None':
			f_W_in = open('output/' + pre_train + '/W_in', 'r')
			W_in = pickle.load(f_W_in)['000']
			f_W_in.close()

			f_W_act = open('output/' + pre_train + '/W_act', 'r')
			W_act = pickle.load(f_W_act)['000']
			f_W_act.close()
		else:
			W_in = np.random.random_sample(size=(nInpNeurons, nHidNeurons)) + 1.0
			W_act = (np.random.random_sample(size=(nHidNeurons, nActNeurons))/1000+1.0)/nHidNeurons
		
		W_in_init = np.copy(W_in)
		W_act_init = np.copy(W_act)
		W_in_since_update = np.copy(W_in)
		perf_track = np.zeros((nActNeurons, 2))

		choice_count = np.zeros((nClasses, nClasses))
		dopa_save = []
		perf_epi = []
		dW_save=np.array([])

		# pbar_epi = ProgressBar()
		# for e in pbar_epi(range(nEpiTot)):
		for e in range(nEpiTot):
			if e==nEpiCrit and verbose: print '----------end crit-----------'

			#shuffle input
			if protocol=='digit' or (protocol=='gabor' and e < nEpiCrit):
				rndImages, rndLabels = ex.shuffle([images, labels])
			elif protocol=='gabor' and e >= nEpiCrit:
				rndImages, rndLabels = ex.shuffle([images_task, labels_task])

			#train network with mini-batches
			for b in range(int(nImages/nBatch)):

				#re-compute the pdf for bayesian inference if any weights have changed more than a threshold
				if classifier=='bayesian' and (e >= nEpiCrit or test_each_epi):
					W_mschange = np.sum((W_in_since_update - W_in)**2, 0)
					if (W_mschange/940 > 0.005).any() or (e==0 and b==0): 
						W_in_since_update = np.copy(W_in)
						pdf_marginals, pdf_evidence, pdf_labels = bc.pdf_estimate(rndImages, rndLabels, W_in, kwargs, pdf_method)
				
				#select batch training images (may leave a few training examples out (< nBatch))
				bImages = rndImages[b*nBatch:(b+1)*nBatch,:]
				bLabels = rndLabels[b*nBatch:(b+1)*nBatch]

				#initialize batch variables
				ach = np.ones(nBatch)
				dopa = np.ones(nBatch)
				dW_in = 0.
				dW_act = 0.
				disinhib_Hid = np.ones(nBatch)##np.zeros(nBatch)
				disinhib_Act = np.zeros(nBatch)
				
				#compute activation of hidden and classification neurons
				bHidNeurons = ex.propL1(bImages, W_in, SM=False)
				if train_class_layer: 
					bActNeurons = ex.propL1(ex.softmax(bHidNeurons, t=t_hid), W_act, SM=False)
					bPredictActions = rActions[np.argmax(bActNeurons,1)]
				elif e >= nEpiCrit:
					posterior = bc.bayesian_decoder(ex.softmax(bHidNeurons, t=t_hid), pdf_marginals, pdf_evidence, pdf_labels, pdf_method)
					bPredictActions = rActions[np.argmax(posterior,1)]

				#add noise to activation of hidden neurons and compute lateral inhibition
				if exploration and (e >= nEpiCrit):
					exploratory = epsilon>np.random.uniform(0, 1, nBatch) if e_greedy else np.ones(nBatch, dtype=bool)
					bHidNeurons[exploratory] += np.random.normal(0, np.std(bHidNeurons)*noise_std, np.shape(bHidNeurons))[exploratory]
					bHidNeurons = ex.softmax(bHidNeurons, t=t_hid)
					bActNeurons = ex.propL1(bHidNeurons, W_act, SM=False)
				else:
					bHidNeurons = ex.softmax(bHidNeurons, t=t_hid)

				if train_class_layer:
					#adds noise in W_act neurons
					if e < nEpiCrit:
						exploratory = epsilon>np.random.uniform(0, 1, nBatch) if e_greedy else np.ones(nBatch, dtype=bool)
						# bActNeurons[exploratory] += np.random.normal(0, noise_std, np.shape(bActNeurons))[exploratory]
						bActNeurons[exploratory] += np.random.normal(0, 4.0, np.shape(bActNeurons))[exploratory]
					bActNeurons = ex.softmax(bActNeurons, t=t_act)
					#take action			
					bActions = rActions[np.argmax(bActNeurons,1)]	
					#compute reward
					bReward = ex.reward_delivery(ex.labels2actionVal(bLabels), bActions)
				elif e >= nEpiCrit:
					posterior_noise = bc.bayesian_decoder(bHidNeurons, pdf_marginals, pdf_evidence, pdf_labels, pdf_method)
					bActions = rActions[np.argmax(posterior_noise,1)]
					#compute reward
					bReward = ex.reward_delivery(ex.labels2actionVal(bLabels), bActions)

				# pred_bLabels_idx = ex.val2idx(bPredictActions) ##same as bActions_idx for exploration = True ??
				# ach, ach_labels = ex.compute_ach(perf_track, pred_bLabels_idx, aHigh=aHigh, rActions=None, aPairing=1.0) # make rActions=None or aPairing=1.0 to remove pairing

				#determine predicted reward
				if e_greedy:
					# predicted_reward = ~exploratory #doesn't predict a reward on all exploratory trials (slightly worse performance than next line; ~2%)
					predicted_reward = np.logical_or(bPredictActions==bActions, ~exploratory) #doesn't predict a reward only on exploratory trials that lead to a different output
				elif classifier!='bayesian':
					predicted_reward = ex.reward_prediction(bPredictActions, bActions, proba_predict)
				elif classifier=='bayesian' and e >= nEpiCrit:
					predicted_reward = ex.reward_prediction(bPredictActions, bActions, proba_predict, posterior)

				#compute dopa signal and disinhibition based on training period
				if e < nEpiCrit and train_class_layer:
					""" critical period """
					# dopa = ex.compute_dopa(bPredictActions, bActions, bReward, dHigh=0.0, dMid=0.75, dNeut=0.0, dLow=-0.5) #original param give close to optimal results
					# dopa = ex.compute_dopa(bPredictActions, bActions, bReward, dHigh=dHigh, dMid=dMid, dNeut=dNeut, dLow=dLow)
					dopa = ex.compute_dopa(predicted_reward, bReward, dHigh=0.0, dMid=0.2, dNeut=-0.3, dLow=-0.5)

					disinhib_Hid = ach
					disinhib_Act = dopa

				elif e >= nEpiCrit: 
					""" Dopa - perceptual learning """
					if proba_predict:
						dopa, prediction_error = ex.compute_dopa_proba(predicted_reward, bReward, nn_regressor, dopa_function=np.sign, param_xplr=param_xplr)
						tmp_input = np.zeros((nBatch, 2))
						tmp_input[:,0] = prediction_error
						tmp_input[:,1] = dopa
						nn_input = np.append(nn_input, tmp_input, axis=0)
					else:
						dopa = ex.compute_dopa(predicted_reward, bReward, dHigh=dHigh, dMid=dMid, dNeut=dNeut, dLow=dLow)

					disinhib_Hid = ach*dopa
					# disinhib_Act = ex.compute_dopa(bPredictActions, bActions, bReward, dHigh=0.0, dMid=0.75, dNeut=0.0, dLow=-0.5) #continuous learning in L2
					disinhib_Act = np.zeros(nBatch) #no learning in L2 during perc_dopa.
					dopa_save = np.append(dopa_save, dopa)
					
				#compute weight updates
				dW_in = ex.learningStep(bImages, bHidNeurons, W_in, lr=lr, disinhib=disinhib_Hid)
				if train_class_layer: dW_act = ex.learningStep(bHidNeurons, bActNeurons, W_act, lr=lr*1e-4, disinhib=disinhib_Act)

				#update weights
				if e<nEpiCrit or not lim_weights:
					W_in += dW_in
				elif e>=nEpiCrit: #artificially prevents weight explosion; used to dissociate influences in parameter exploration
					mask = np.logical_and(np.sum(W_in+dW_in,0)<=940.801, np.min(W_in+dW_in,0)>0.2)
					W_in[:,mask] += dW_in[:,mask]
				if train_class_layer: W_act += dW_act

				W_in = np.clip(W_in, 1e-10, np.inf)
				if train_class_layer: W_act = np.clip(W_act, 1e-10, np.inf)

				# if (W_act>1.5).any(): import pdb; pdb.set_trace()
				# if np.isnan(W_in).any(): import pdb; pdb.set_trace()

			#check Wact assignment after each episode:
			if protocol=='digit':
				RFproba, _, _ = rf.hist(runName, {'000':W_in}, classes, images, labels, protocol, SVM=SVM, save_data=False, verbose=False)
			elif protocol=='gabor':
				pref_ori = gr.preferred_orientations({'000':W_in}, params=kwargs)
				RFproba = np.zeros((1, nHidNeurons, nClasses), dtype=int)
				RFproba[0,:,:][pref_ori['000']<=target_ori] = [1,0]
				RFproba[0,:,:][pref_ori['000']>target_ori] = [0,1]
			same = ex.labels2actionVal(np.argmax(RFproba[0],1)) == rActions[np.argmax(W_act,1)]
			if train_class_layer:
				correct_W_act = 0.
				correct_W_act += np.sum(same)
				correct_W_act/=len(RFproba)

			#check performance after each episode
			if verbose:
				if train_class_layer:
					print ('correct action weights: ' + str(int(correct_W_act)) + '/' + str(int(nHidNeurons)) + '; '),
			if save_data:
				if r==0 and e==nEpiCrit-1:
					if protocol=='digit':
						pl.plot_noise_proba(W_in, images, kwargs)
					else:
						pl.plot_noise_proba(W_in, images_task, kwargs)
			if test_each_epi and (verbose or save_data):
				if classifier=='bayesian':
					rdn_idx = np.random.choice(len(labels_test), 1000, replace=False)
					_, perf_tmp = cl.bayesian({'000':W_in}, images_test[rdn_idx], labels_test[rdn_idx], pdf_marginals, pdf_evidence, pdf_labels, kwargs, pdf_method, save_data=False, verbose=False)
				if classifier=='actionNeurons':
					_, perf_tmp = cl.actionNeurons({'000':W_in}, {'000':W_act}, images_test, labels_test, kwargs, save_data=False, verbose=False)
				perf_epi.append(perf_tmp[0])
				if verbose: print 'performance: ' + str(np.round(perf_tmp[0]*100,1)) + '%'
			elif verbose and train_class_layer: print 

		#save weights
		W_in_save[str(r).zfill(3)] = np.copy(W_in)
		W_act_save[str(r).zfill(3)] = np.copy(W_act)
		perf_save[str(r).zfill(3)] = np.copy(perf_epi)

	""" compute histogram of RF classes """
	if protocol=='digit':
		RFproba, RFclass, _ = rf.hist(runName, W_in_save, classes, images, labels, protocol, SVM=SVM, save_data=save_data, verbose=verbose, lr_ratio=1.0, rel_classes=classes[rActions!='0'])

	elif protocol=='gabor':
		n_bins = 10
		bin_size = 180./n_bins
		orientations_bin = np.zeros(len(orientations), dtype=int)
		for i in range(n_bins): 
			mask_bin = np.logical_and(orientations >= i*bin_size, orientations < (i+1)*bin_size)
			orientations_bin[mask_bin] = i

		pref_ori = gr.preferred_orientations(W_in_save, params=kwargs)
		RFproba = np.zeros((nRun, nHidNeurons, nClasses), dtype=int)
		for r in pref_ori.keys():
			RFproba[int(r),:,:][pref_ori[r]<=target_ori] = [1,0]
			RFproba[int(r),:,:][pref_ori[r]>target_ori] = [0,1]
		_, _, _ = rf.hist(runName, W_in_save, range(n_bins), images, orientations_bin, protocol, n_bins=n_bins, SVM=SVM, save_data=save_data, verbose=verbose)

	""" compute correct weight assignment in the ouput layer """
	if train_class_layer:
		correct_W_act = 0.
		notsame = {}
		for k in W_act_save.keys():
			same = ex.labels2actionVal(np.argmax(RFproba[int(k)],1)) == rActions[np.argmax(W_act_save[k],1)]
			notsame[k] = np.argwhere(~same)
			correct_W_act += np.sum(same)
		correct_W_act/=len(RFproba)
	else:
		notsame = None
		correct_W_act = 0.

	""" plot weights """
	if save_data:
		if show_W_act: W_act_pass=W_act_save
		else: W_act_pass=None
		if protocol=='digit':
			rf.plot(runName, W_in_save, RFproba, target=target, W_act=W_act_pass, sort=sort, notsame=notsame, verbose=verbose)
			slopes = {}
		elif protocol=='gabor':
			rf.plot(runName, W_in_save, RFproba, W_act=W_act_pass, notsame=notsame, verbose=verbose)
			curves = gr.tuning_curves(W_in_save, params=kwargs, method='no_softmax', plot=True) #basic, no_softmax, with_noise
			slopes = gr.slopes(W_in_save, curves, pref_ori, kwargs)
		if test_each_epi:
			pl.perf_progress(perf_save, kwargs)

	""" compute network performance """
	if classifier=='actionNeurons':	allCMs, allPerf = cl.actionNeurons(W_in_save, W_act_save, images_test, labels_test, kwargs, save_data, verbose)
	if classifier=='SVM': 			allCMs, allPerf = cl.SVM(runName, W_in_save, images, labels, classes, nInpNeurons, A, dataset, save_data, verbose)
	if classifier=='neuronClass':	allCMs, allPerf = cl.neuronClass(runName, W_in_save, classes, RFproba, nInpNeurons, A, images_test, labels_test, save_data, verbose)
	if classifier=='bayesian':		allCMs, allPerf = cl.bayesian(W_in_save, images_test, labels_test, pdf_marginals, pdf_evidence, pdf_labels, kwargs, pdf_method, save_data, verbose)

	if verbose and train_class_layer: print 'correct action weight assignment:\n' + str(correct_W_act) + ' out of ' + str(nHidNeurons)

	"""" save data """
	if save_data: ex.save_data(W_in_save, W_act_save, perf_save, slopes, kwargs)

	if verbose: print '\nrun: '+runName + '\n'

	if param_xplr=='None': import pdb; pdb.set_trace()

	return allCMs, allPerf, correct_W_act/nHidNeurons, W_in, W_act, RFproba, nn_input





	


















