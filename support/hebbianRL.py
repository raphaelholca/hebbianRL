""" 
This function trains a hebbian neural network to learn a representation from the MNIST dataset. It makes use of a reward/relevance signal that increases the learning rate when the network makes a correct state-action pair selection.

Output is saved under RL/data/[self.name]
"""

# from progressbar import ProgressBar
from inspect import isfunction
import numpy as np
import matplotlib.pyplot as pyplot
import support.external as ex
import support.plots as pl
import support.classifier as cl
import support.assessRF as rf
import support.grating as gr
import support.bayesian_decoder as bc
import sys
import time
import pickle
import warnings

ex = reload(ex)
pl = reload(pl)
cl = reload(cl)
rf = reload(rf)
gr = reload(gr)
bc = reload(bc)

class Network:

	def __init__(self, name, n_run=1, n_epi_crit=10, n_epi_dopa=10, t=0.1, A=1.2, n_hid_neurons=49, lim_weights=False, lr=0.01, noise_std=0.2, exploration=True, pdf_method='fit', n_batch=20, protocol='digit', classifier='actionNeurons', pre_train=None, test_each_epi=True, SVM=False, save_data=True, verbose=True, show_W_act=True, sort=None, target=None):

		"""
		Sets network parameters 

			Args:
				name (str, optional): name of the folder where to save results. Default='net'
				n_run (int, optional): number of runs. Default=1
				n_epi_crit (int, optional): number of 'critical period' episodes in each run (episodes when reward is not required for learning). Default=10
				n_epi_dopa (int, optional): number of 'adult' episodes in each run (episodes when reward is not required for learning). Default=10
				t (float, optional): temperature of the softmax function (t<<1: strong competition; t>=1: weak competition). Default=0.1
				A (float, optional): input normalization constant. Will be used as: (input size)*A. Default=1.2
				n_hid_neurons (int, optional): number of hidden neurons. Default=49
				lim_weights (bool, optional): whether to artificially limit the value of weights. Used during parameter exploration. Default=False
				lr (float, optiona): learning rate. Default=0.01
				noise_std (float, optional): parameter of the standard deviation of the normal distribution from which noise is drawn. Default=0.2
				exploration (bool, optional): whether to take take explorative decisions (True) or not (False). Default=True
				pdf_method (str, optional): method used to approximate the pdf; valid: 'fit', 'subsample', 'full'. Default='fit'
				n_batch (int, optional): mini-batch size. Default=20
				protocol (str, optional): training protocol. Possible values: 'digit' (MNIST classification), 'gabor' (orientation discrimination). Default='digit'
				classifier (str, optional): which classifier to use for performance assessment. Possible values are: 'actionNeurons', 'SVM', 'neuronClass', 'bayesian'. Default='actionNeurons'
				pre_train (str, optional): initialize weights with pre-trained weights saved to file; use '' or 'None' for random initialization. Default=None
				test_each_epi (bool, optional): whether to test the network's performance at each episode. Default=True
				SVM (bool, optional): whether to use an SVM or the number of stimuli that activate a neuron to determine the class of the neuron. Default=False
				save_data (bool, optional): whether to save data to disk. Default=True
				verbose	(bool, optional): whether to create text output. Default=True
				show_W_act (bool, optional): whether to display W_act weights on the weight plots. Default=True
				sort (str, optional): sorting methods for weights when displaying. Valid value: None, 'class', 'tSNE'. Default=None
				target (int, optional): target digit (to be used to color plots). Use None if not desired. Default=None
				
				'dataset'		: 'test'			,# dataset to use; possible values: 'test': MNIST test, 'train': MNIST train, 'grating': orientation discrimination
				'dHigh' 		: 4.5 				,# learning rate increase for unexpected reward																	digit: 4.5	; gabor: 2.0
				'dMid' 			: 0.02 				,# learning rate increase for correct reward prediction															digit: 0.02	; gabor: ---
				'dNeut' 		: -0.1				,# learning rate increase for correct no reward prediction														digit: -0.1	; gabor: ---
				'dLow' 			: -2.0				,# learning rate increase for incorrect reward prediction														digit: -2.0	; gabor: 0.0
				'target_ori' 	: 85. 				,# target orientation around which to discriminate clock-wise vs. counter clock-wise
				'excentricity' 	: 3. 				,# degree range within wich to test the network (on each side of target orientation)
				'noise_crit'	: 0. 				,# noise injected in the gabor filter for the pre-training (critical period)
				'noise_train'	: 0. 				,# noise injected in the gabor filter for the training
				'noise_test'	: 0.2 				,# noise injected in the gabor filter for the testing
				'im_size'		: 28 				,# side of the gabor filter image (total pixels = im_size * im_size)
				'seed' 			: 995, #np.random.randint(1000), 	# seed of the random number generator
		"""
		self.name 			= name
		self.n_run 			= n_run
		self.n_epi_crit		= n_epi_crit				
		self.n_epi_dopa		= n_epi_dopa				
		self.t				= t 						
		self.A 				= A
		self.n_hid_neurons 	= n_hid_neurons
		self.lim_weights	= lim_weights
		self.lr				= lr
		self.noise_std		= noise_std
		self.exploration	= exploration
		self.pdf_method 	= pdf_method
		self.n_batch 		= n_batch
		self.protocol		= protocol
		self.classifier		= classifier
		self.pre_train		= pre_train
		self.test_each_epi	= test_each_epi
		self.SVM 			= SVM
		self.save_data 		= save_data
		self.verbose 		= verbose
		self.show_W_act 	= show_W_act
		self.sort 			= sort
		self.target 		= target


	def train(	self, images, labels, orientations, 
				images_test, labels_test, orientations_test, 
				images_task, labels_task, orientations_task,
				kwargs,
				classes, rActions, dataset, dHigh, dMid, dNeut, dLow, target_ori, excentricity, noise_crit, noise_train, noise_test, im_size, seed):

		""" variable initialization """


		W_in_save = {}
		W_act_save = {}
		perf_save = {}
		nClasses = len(classes)
		nEpiTot = self.n_epi_crit + self.n_epi_dopa
		nImages = np.size(images,0)
		nInpNeurons = np.size(images,1)
		nActNeurons = nClasses
		train_class_layer = False if self.classifier=='bayesian' else True
		self.show_W_act = False if self.classifier=='bayesian' else self.show_W_act
		proba_predict = False if self.classifier!='bayesian' else proba_predict
		nn_input = np.empty((0,2), dtype=float) #input to the regressor neural net; nn_input[:,0] is prediction_error; nn_input[:,1] is tried DA value 
		nn_input_save = np.empty((0,2), dtype=float)

		""" training of the network """
		if self.verbose: 
			print 'run:  ' + self.name
			print '\ntraining hebbian network...'
		for r in range(self.n_run):
			np.random.seed(seed+r)
			if self.verbose: print '\nrun: ' + str(r+1)

			#initialize network variables
			""" load pre-trained weights """
			
			if self.pre_train!='' and self.pre_train is not None:
				f_W_in = open('output/' + self.pre_train + '/W_in', 'r')
				W_in = pickle.load(f_W_in)['000']
				f_W_in.close()

				f_W_act = open('output/' + self.pre_train + '/W_act', 'r')
				W_act = pickle.load(f_W_act)['000']
				f_W_act.close()
			else:
				W_in = np.random.random_sample(size=(nInpNeurons, self.n_hid_neurons)) + 1.0
				W_act = (np.random.random_sample(size=(self.n_hid_neurons, nActNeurons))/1000+1.0)/self.n_hid_neurons
			
			W_in_init = np.copy(W_in)
			W_act_init = np.copy(W_act)
			W_in_since_update = np.copy(W_in)
			perf_track = np.zeros((nActNeurons, 2))

			choice_count = np.zeros((nClasses, nClasses))
			dopa_save = np.array([])
			perf_epi = []
			dW_save=np.array([])

			# pbar_epi = ProgressBar()
			# for e in pbar_epi(range(nEpiTot)):
			for e in range(nEpiTot):
				if self.verbose and e==self.n_epi_crit: print '----------end crit-----------'

				#shuffle input
				if self.protocol=='digit' or (self.protocol=='gabor' and e < self.n_epi_crit):
					rndImages, rndLabels = ex.shuffle([images, labels])
				elif self.protocol=='gabor' and e >= self.n_epi_crit:
					rndImages, rndLabels = ex.shuffle([images_task, labels_task])

				#train network with mini-batches
				for b in range(int(nImages/self.n_batch)):
					#re-compute the pdf for bayesian inference if any weights have changed more than a threshold
					if self.classifier=='bayesian' and (e >= self.n_epi_crit or self.test_each_epi):
						W_mschange = np.sum((W_in_since_update - W_in)**2, 0)
						if (W_mschange/940 > 0.01).any() or (e==0 and b==0): ## > 0.005 
							W_in_since_update = np.copy(W_in)
							pdf_marginals, pdf_evidence, pdf_labels = bc.pdf_estimate(rndImages, rndLabels, W_in, kwargs)
				
					#select batch training images (may leave a few training examples out (< self.n_batch))
					bImages = rndImages[b*self.n_batch:(b+1)*self.n_batch,:]
					bLabels = rndLabels[b*self.n_batch:(b+1)*self.n_batch]

					#initialize batch variables
					dopa = np.ones(self.n_batch)
					dW_in = 0.
					dW_act = 0.
					disinhib_Hid = np.ones(self.n_batch)##np.zeros(self.n_batch)
					disinhib_Act = np.zeros(self.n_batch)
					
					#compute activation of hidden and classification neurons
					bHidNeurons = ex.propL1(bImages, W_in, SM=False)
					if train_class_layer: 
						bActNeurons = ex.propL1(ex.softmax(bHidNeurons, t=self.t), W_act, SM=False)
						bPredictActions = rActions[np.argmax(bActNeurons,1)]
					elif e >= self.n_epi_crit:
						posterior = bc.bayesian_decoder(ex.softmax(bHidNeurons, t=self.t), pdf_marginals, pdf_evidence, pdf_labels, self.pdf_method)
						bPredictActions = rActions[np.argmax(posterior,1)]

					#add noise to activation of hidden neurons and compute lateral inhibition
					if self.exploration and (e >= self.n_epi_crit):
						bHidNeurons += np.random.normal(0, np.std(bHidNeurons)*self.noise_std, np.shape(bHidNeurons))
						bHidNeurons = ex.softmax(bHidNeurons, t=self.t)
						if train_class_layer:
							bActNeurons = ex.propL1(bHidNeurons, W_act, SM=False)
					else:
						bHidNeurons = ex.softmax(bHidNeurons, t=self.t)

					if train_class_layer:
						#adds noise in W_act neurons
						if e < self.n_epi_crit:
							bActNeurons += np.random.normal(0, 4.0, np.shape(bActNeurons))
						bActNeurons = ex.softmax(bActNeurons, t=self.t)
						#take action			
						bActions = rActions[np.argmax(bActNeurons,1)]	
						#compute reward
						bReward = ex.reward_delivery(ex.labels2actionVal(bLabels), bActions)
					elif e >= self.n_epi_crit:
						posterior_noise = bc.bayesian_decoder(bHidNeurons, pdf_marginals, pdf_evidence, pdf_labels, self.pdf_method)
						bActions = rActions[np.argmax(posterior_noise,1)]
						#compute reward
						bReward = ex.reward_delivery(ex.labels2actionVal(bLabels), bActions)

					#determine predicted reward
					if self.classifier!='bayesian':
						predicted_reward = ex.reward_prediction(bPredictActions, bActions, proba_predict)
					elif self.classifier=='bayesian' and e >= self.n_epi_crit:
						predicted_reward = ex.reward_prediction(bPredictActions, bActions, proba_predict, posterior)

					#compute dopa signal and disinhibition based on training period
					if e < self.n_epi_crit and train_class_layer:
						""" critical period; trains class layer """
						# dopa = ex.compute_dopa(predicted_reward, bReward, dHigh=0.0, dMid=0.75, dNeut=0.0, dLow=-0.5) #original param give close to optimal results
						# dopa = ex.compute_dopa(predicted_reward, bReward, dHigh=dHigh, dMid=dMid, dNeut=dNeut, dLow=dLow)
						dopa = ex.compute_dopa(predicted_reward, bReward, dHigh=0.0, dMid=0.2, dNeut=-0.3, dLow=-0.5)

						disinhib_Hid = np.ones(self.n_batch)
						disinhib_Act = dopa

					elif e >= self.n_epi_crit: 
						""" Dopa - perceptual learning """
						dopa = ex.compute_dopa(predicted_reward, bReward, dHigh=dHigh, dMid=dMid, dNeut=dNeut, dLow=dLow)

						disinhib_Hid = dopa
						# disinhib_Act = ex.compute_dopa(bPredictActions, bActions, bReward, dHigh=0.0, dMid=0.75, dNeut=0.0, dLow=-0.5) #continuous learning in L2
						disinhib_Act = np.zeros(self.n_batch) #no learning in L2 during perc_dopa.
						dopa_save = np.append(dopa_save, dopa)
						
					#compute weight updates
					dW_in = ex.learningStep(bImages, bHidNeurons, W_in, lr=self.lr, disinhib=disinhib_Hid)
					if train_class_layer: dW_act = ex.learningStep(bHidNeurons, bActNeurons, W_act, lr=self.lr*1e-4, disinhib=disinhib_Act)

					#update weights
					if e<self.n_epi_crit or not self.lim_weights:
						W_in += dW_in
					elif e>=self.n_epi_crit: #artificially prevents weight explosion; used to dissociate influences in parameter self.exploration
						mask = np.logical_and(np.sum(W_in+dW_in,0)<=940.801, np.min(W_in+dW_in,0)>0.2)
						W_in[:,mask] += dW_in[:,mask]
					if train_class_layer: W_act += dW_act

					W_in = np.clip(W_in, 1e-10, np.inf)
					if train_class_layer: W_act = np.clip(W_act, 1e-10, np.inf)

				""" end of mini-batch """

				#check Wact assignment after each episode:
				if self.protocol=='digit':
					RFproba, _, _ = rf.hist(self, {'000':W_in}, classes, images, labels, self.protocol, SVM=self.SVM, save_data=False, verbose=False)
				elif self.protocol=='gabor':
					pref_ori = gr.preferred_orientations({'000':W_in}, params=kwargs)
					RFproba = np.zeros((1, self.n_hid_neurons, nClasses), dtype=int)
					RFproba[0,:,:][pref_ori['000']<=target_ori] = [1,0]
					RFproba[0,:,:][pref_ori['000']>target_ori] = [0,1]
				same = ex.labels2actionVal(np.argmax(RFproba[0],1)) == rActions[np.argmax(W_act,1)]
				if train_class_layer:
					correct_W_act = 0.
					correct_W_act += np.sum(same)
					correct_W_act/=len(RFproba)

				#check performance after each episode
				if self.verbose:
					if train_class_layer:
						print ('correct action weights: ' + str(int(correct_W_act)) + '/' + str(int(self.n_hid_neurons)) + '; '),
				if self.save_data:
					if r==0 and e==self.n_epi_crit-1:
						if self.protocol=='digit':
							pl.plot_noise_proba(self, W_in, images, kwargs)
						else:
							pl.plot_noise_proba(self, W_in, images_task, kwargs)
				if self.test_each_epi and (self.verbose or self.save_data):
					if self.classifier=='bayesian':
						rdn_idx = np.random.choice(len(labels_test), 1000, replace=False)
						_, perf_tmp = cl.bayesian({'000':W_in}, images, labels, images_test[rdn_idx], labels_test[rdn_idx], kwargs, save_data=False, verbose=False)
					if self.classifier=='actionNeurons':
						_, perf_tmp = cl.actionNeurons(self, {'000':W_in}, {'000':W_act}, images_test, labels_test, kwargs, save_data=False, verbose=False)
					perf_epi.append(perf_tmp[0])
					if self.verbose: print 'performance: ' + str(np.round(perf_tmp[0]*100,1)) + '%'
				elif self.verbose and train_class_layer: print 

			""" end of episode """

			#save weights
			W_in_save[str(r).zfill(3)] = np.copy(W_in)
			W_act_save[str(r).zfill(3)] = np.copy(W_act)
			perf_save[str(r).zfill(3)] = np.copy(perf_epi)

		""" compute histogram of RF classes """
		if self.protocol=='digit':
			RFproba, RFclass, _ = rf.hist(self, W_in_save, classes, images, labels, self.protocol, SVM=self.SVM, save_data=self.save_data, verbose=self.verbose, lr_ratio=1.0, rel_classes=classes[rActions!='0'])

		elif self.protocol=='gabor':
			n_bins = 10
			bin_size = 180./n_bins
			orientations_bin = np.zeros(len(orientations), dtype=int)
			for i in range(n_bins): 
				mask_bin = np.logical_and(orientations >= i*bin_size, orientations < (i+1)*bin_size)
				orientations_bin[mask_bin] = i

			pref_ori = gr.preferred_orientations(W_in_save, params=kwargs)
			RFproba = np.zeros((self.n_run, self.n_hid_neurons, nClasses), dtype=int)
			for r in pref_ori.keys():
				RFproba[int(r),:,:][pref_ori[r]<=target_ori] = [1,0]
				RFproba[int(r),:,:][pref_ori[r]>target_ori] = [0,1]
			_, _, _ = rf.hist(self, W_in_save, range(n_bins), images, orientations_bin, self.protocol, n_bins=n_bins, SVM=self.SVM, save_data=self.save_data, verbose=self.verbose)

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
		if self.save_data:
			if self.show_W_act: W_act_pass=W_act_save
			else: W_act_pass=None
			if self.protocol=='digit':
				rf.plot(self, W_in_save, RFproba, target=self.target, W_act=W_act_pass, sort=self.sort, notsame=notsame, verbose=self.verbose)
				slopes = {}
			elif self.protocol=='gabor':
				rf.plot(self, W_in_save, RFproba, W_act=W_act_pass, notsame=notsame, verbose=self.verbose)
				curves = gr.tuning_curves(W_in_save, params=kwargs, method='no_softmax', plot=True) #basic, no_softmax, with_noise
				slopes = gr.slopes(W_in_save, curves, pref_ori, kwargs)
			if self.test_each_epi:
				pl.perf_progress(self, perf_save, kwargs)

		""" compute network performance """
		if self.classifier=='actionNeurons':	allCMs, allPerf = cl.actionNeurons(self, W_in_save, W_act_save, images_test, labels_test, kwargs, self.save_data, self.verbose)
		if self.classifier=='self.SVM': 			allCMs, allPerf = cl.self.SVM(self, self.name, W_in_save, images, labels, classes, nInpNeurons, dataset, self.save_data, self.verbose)
		if self.classifier=='neuronClass':	allCMs, allPerf = cl.neuronClass(self, self.name, W_in_save, classes, RFproba, nInpNeurons, images_test, labels_test, self.save_data, self.verbose)
		if self.classifier=='bayesian':		allCMs, allPerf = cl.bayesian(self, W_in_save, images, labels, images_test, labels_test, kwargs)

		if self.verbose and train_class_layer: print 'correct action weight assignment:\n' + str(correct_W_act) + ' out of ' + str(self.n_hid_neurons)

		"""" save data """
		if self.save_data: ex.save_data(self, W_in_save, W_act_save, perf_save, slopes)

		if self.verbose: print '\nrun: '+self.name + '\n'

		import pdb; pdb.set_trace()

		return allCMs, allPerf, correct_W_act/self.n_hid_neurons, W_in, W_act, RFproba, nn_input_save





	


















