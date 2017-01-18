"""
Author: Raphael Holca-Lamarre
Date: 23/10/2014

This function trains a Hebbian neural network to learn a representation from the MNIST dataset. It makes use of a reward/relevance signal that increases the learning rate when the network makes a correct state-action pair selection.
"""

import numpy as np
import helper.external as ex
import helper.grating as gr
import helper.bayesian_decoder as bc
import helper.assess_network as an
import time
import pickle
import scipy.special
import os
from pdb import set_trace

ex = reload(ex)
gr = reload(gr)
bc = reload(bc)
an = reload(an)

class Network:
	""" Hebbian neural network with dopamine-inspired learning """

	def __init__(self, dHigh, dMid, dNeut, dLow, dopa_func='discrete', dopa_out_same=True, train_out_dopa=False, dHigh_out=0.0, dMid_out=0.2, dNeut_out=-0.3, dLow_out=-0.5, ach_1=1.0, ach_2=0.0, ach_3=0.0, ach_4=0.0, ach_func='sigmoidal', ach_avg=20, ach_stim=False, ach_uncertainty=True, ach_BvSB=False, ach_approx_class=False, protocol='digit', name='net', dopa_release=True, ach_release=False, n_runs=1, n_epi_crit=20, n_epi_fine=0, n_epi_perc=20, n_epi_post=0, t_hid=1.0, t_out=1.0, A=940., lr_hid=5e-3, lr_out=5e-7, batch_size=50, block_feedback=False, shuffle_datasets=True, n_hid_neurons=49, weight_init='input', init_file=None, lim_weights=False, log_weights='log', epsilon_xplr=0.5, noise_xplr_hid=0.2, noise_xplr_out=2e4, noise_activ=0.2, exploration=True, compare_output=False, pdf_method='fit', classifier='neural_prob', RF_classifier='svm', test_each_epi=False, early_stop=True, verbose=True, save_light=True, seed=None, pypet=False, pypet_name=''):

		"""
		Sets network parameters 

			Args:
				dHigh (float): values of dopamine release for -reward expectation, +reward delivery
				dMid (float): values of dopamine release for +reward expectation, +reward delivery
				dNeut (float): values of dopamine release for -reward expectation, -reward delivery
				dLow (float): values of dopamine release for +reward expectation, -reward delivery
				dopa_func (str, optional): relationship between RPE and DA release. Default: 'discrete'
				dopa_out_same (bool, optional): whether to use the same dopa values in the output layer as in the hidden layer (True) or use the values provided in the 'd_out' variables below. Default: True
				train_out_dopa (bool, optional): whether to train the output layer during the dopa period. Default: True
				dHigh_out (float, optional): values of dopamine release for -reward expectation, +reward delivery for output layer. Default: 0.0
				dMid_out (float, optional): values of dopamine release for +reward expectation, +reward delivery for output layer. Default: 0.2
				dNeut_out (float, optional): values of dopamine release for -reward expectation, -reward delivery for output layer. Default: -0.3
				dLow_out (float, optional): values of dopamine release for +reward expectation, -reward delivery for output layer. Default: -0.5
				ach_1 (float, optional): value of the first parameter of the release function for acetylcholine. Default: 1.0
				ach_2 (float, optional): value of the second parameter of the release function for acetylcholine. Default: 0.0
				ach_3 (float, optional): value of the third parameter of the release function for acetylcholine. Default: 0.0
				ach_4 (float, optional): value of the fourth parameter of the release function for acetylcholine. Default: 0.0
				ach_func (str, optional): release function for acetylcholine. Default: 'sigmoidal'
				ach_avg (int, optional): number of episodes to average over for moving averag of performance. Default: 20 
				ach_stim (bool, optional): whether to average stimulus difficulty over stimuli (True) or over classes (False). Default: False 
				ach_uncertainty (bool, optional): whether to use uncertainty (True) or performance (False) as a measure of task difficulty. Default: True
				ach_BvSB (bool, optional): whether to compare best vs second-best posterior for uncertainty extimate (True) or simply best posterior (False). Default: False
				ach_approx_class (bool, optional): whether to approximate the class of the stimulus using net output (True) or use true class label (False). Default: False
				protocol (str, optional): training protocol. Possible values: 'digit' (MNIST classification), 'gabor' (orientation discrimination). Default: 'digit'
				name (str, optional): name of the folder where to save results. Default: 'net'
				dopa_release (bool, optional): whether DA is release during perceptual learning period
				ach_release (bool, optional): whether ACh is release during perceptual learning period
				n_runs (int, optional): number of runs. Default: 1
				n_epi_crit (int, optional): number of 'critical period' episodes in each run (episodes when reward is not required for learning). Default: 20
				n_epi_fine (int, optional): number of episodes after the critical period to train output layer without learning in the hidden layer (similar to post). Default: 5
				n_epi_perc (int, optional): number of 'adult' episodes in each run (episodes when reward is not required for learning). Default: 20
				n_epi_post (int, optional): number of episodes after the dopa period to train output layer without learning in the hidden layer. Default: 5
				t_hid (float, optional): temperature of the softmax function in the hidden layer (t<<1: strong competition; t>=1: weak competition). Default: 1.0
				t_out (float, optional): temperature of the softmax function in the output layer (t<<1: strong competition; t>=1: weak competition). Default: 1.0
				A (float, optional): input normalization constant for the hidden layer. Default: 940.
				lr_hid (float, optional): learning rate for the hidden layer. Default: 5e-3
				lr_out (float, optional): learning rate for the output layer. Default: 5e-7
				batch_size (int, optional): mini-batch size. Default: 20
				block_feedback (bool, optional): whether to use block feedback (dopa averaged over a batch) or trial feedback (individual dopa for each stimulus). Default: False
				shuffle_datasets (bool, optional): whether to shuffle train and test datasets to create new split of the data for each individual run. Default: True
				n_hid_neurons (int, optional): number of hidden neurons. Default: 49
				weight_init (str, optional): method for initializing weights: 'random', 'input' (based on input statistic), 'file' (load from file). Default: 'input' 
				init_file (str, optional): folder in output directory from which to load network from for weight initialization; use '' or None for random initialization; use 'NO_INIT' to not initialize weights. Default: None
				lim_weights (bool, optional): whether to artificially limit the value of weights. Used during parameter exploration. Default: False
				log_weights (str, optional): transfer function of the weights; possible values: 'lin', 'log', 'linlog'. Default: 'log'
				epsilon_xplr (float, optional): probability of taking an exploratory decision (proba of noise injection). Default: 0.5
				noise_xplr_hid (float, optional): parameter of the standard deviation of the normal distribution from which noise is drawn, for exploration in the hidden layer. Default: 0.2
				noise_xplr_out (float, optional): parameter of the standard deviation of the normal distribution from which noise is drawn, for exploration in the output layer. Default: 2e4
				noise_activ (float, optional): standard deviation of additive noise corrupting the activation of hidden neurons. Default: 0.2
				exploration (bool, optional): whether to take take explorative decisions (True) or not (False). Default: True
				compare_output (bool, optional): whether to compare the value of greedy and taken action to determine if the trial is exploratory. Default: False
				pdf_method (str, optional): method used to approximate the pdf; valid: 'fit', 'subsample', 'full'. Default: 'fit'
				classifier (str, optional): which classifier to use for performance assessment. Possible values are: 'neural_prob', 'neural_dopa', 'bayesian'. Default: 'neural_prob'
				RF_classifier (str, optional): which classifier to use to classifier RFs. Possible values are: 'data', 'svm'. Default: 'svm'
				test_each_epi (bool, optional): whether to test the network's performance at each episode with test data. Default: False
				early_stop (bool, optional): whether to stop training when performance saturates. Default: True
				verbose	(bool, optional): whether to create text output. Default: True
				save_light (bool, optional): whether to save a lighter version of the network (excludes ach_tracker, stim_perf_labels_saved, _idx_shuffle_saved)
				seed (int, optional): seed of the random number generator. Default: None
				pypet (bool, optional): whether the network simulation is part of pypet exploration
				pypet_name (str, optional): name of the directory in which data is saved when doing pypet exploration. Default: ''
		"""
		
		self.dopa_values 		= {'dHigh': dHigh, 'dMid':dMid, 'dNeut':dNeut, 'dLow':dLow}
		self.dopa_func 			= dopa_func
		self.dopa_out_same 		= dopa_out_same
		self.train_out_dopa		= train_out_dopa
		self.dopa_values_out 	= {'dHigh': dHigh_out, 'dMid':dMid_out, 'dNeut':dNeut_out, 'dLow':dLow_out} if not self.dopa_out_same else self.dopa_values.copy()
		self.ach_values 		= {'ach_1': ach_1, 'ach_2': ach_2, 'ach_3': ach_3, 'ach_4': ach_4} 
		self.ach_avg 			= ach_avg
		self.ach_stim 			= ach_stim
		self.ach_uncertainty 	= ach_uncertainty
		self.ach_BvSB 			= ach_BvSB
		self.ach_approx_class 	= ach_approx_class
		self.protocol			= protocol
		self.name 				= name
		self.dopa_release 		= dopa_release
		self.ach_release 		= ach_release
		self.n_runs 			= n_runs
		self.n_epi_crit			= n_epi_crit
		self.n_epi_fine 		= n_epi_fine
		self.n_epi_perc			= n_epi_perc
		self.n_epi_post 		= n_epi_post
		self.t_hid				= t_hid
		self.t_out				= t_out
		self.A 					= A
		self.lr_hid				= lr_hid
		self.lr_out				= lr_out
		self.batch_size 		= batch_size
		self.block_feedback 	= block_feedback
		self.shuffle_datasets 	= shuffle_datasets	
		self.n_hid_neurons 		= n_hid_neurons
		self.weight_init 		= weight_init
		self.init_file			= init_file
		self.lim_weights		= lim_weights
		self.log_weights 		= log_weights
		self.epsilon_xplr 		= epsilon_xplr
		self.noise_xplr_hid		= np.clip(noise_xplr_hid, 1e-20, np.inf)
		self.noise_xplr_out 	= np.clip(noise_xplr_out, 1e-20, np.inf)
		self.exploration		= exploration
		self.compare_output 	= compare_output
		self.noise_activ 		= np.clip(noise_activ, 1e-20, np.inf)
		self.pdf_method 		= pdf_method
		self.classifier			= classifier
		self.RF_classifier 		= RF_classifier
		self.test_each_epi		= test_each_epi
		self.early_stop 		= early_stop
		self.verbose 			= verbose
		self.save_light 		= save_light
		self.seed 				= seed
		self.pypet 				= pypet
		self.pypet_name 		= pypet_name if pypet_name != '' else name
		self._early_stop_cond 	= []
		self.ach_func 			= {'linear':ex.ach_linear, 'exponential':ex.ach_exponential, 'polynomial':ex.ach_polynomial, 'sigmoidal':ex.ach_sigmoidal, 'handmade':ex.ach_handmade, 'preset':'preset', 'labels':'labels', 'labels_reverse':'labels_reverse'}[ach_func]

		np.random.seed(self.seed)
		self._check_parameters()
		if not self.pypet:
			self.name = ex.checkdir(self.name, self.protocol, overwrite=True)

		if noise_activ!=0.0: raise NotImplementedError('corruptove noise is commented-out') ##

	def train(self, images_dict, labels_dict, images_params={}):
		""" 
		Train Hebbian neural network

			Args: 
				images_dict (dict): dictionary of 2D image arrays to test the Network on, with keys: 'train', 'test'.
				labels_dict (dict): dictionary of label arrays of the images.
				images_params (dict, optional): parameters used to create the images
		"""

		self._train_start = time.time()

		images_train, images_test, images_task = images_dict['train'], images_dict['test'], images_dict['task']
		labels_train, labels_test, labels_task = labels_dict['train'], labels_dict['test'], labels_dict['task']

		if self.block_feedback:
			print "***** training using block feedback *****"

		self.images_params = images_params
		self.n_images = images_train.shape[0]
		self.classes = np.sort(np.unique(labels_train))
		self.n_classes = len(self.classes)
		self.n_out_neurons = len(self.classes)
		self.n_inp_neurons = np.size(images_train,1)
		self.n_epi_tot = self.n_epi_crit + self.n_epi_fine + self.n_epi_perc + self.n_epi_post
		self.hid_W_naive = np.zeros((self.n_runs, self.n_inp_neurons, self.n_hid_neurons))
		self.hid_W_trained = np.zeros((self.n_runs, self.n_inp_neurons, self.n_hid_neurons))
		self.out_W_naive = np.zeros((self.n_runs, self.n_hid_neurons, self.n_out_neurons))
		self.out_W_trained = np.zeros((self.n_runs, self.n_hid_neurons, self.n_out_neurons))
		self._idx_shuffle = None
		self._idx_shuffle_saved = np.zeros((self.n_runs, images_train.shape[0] + images_test.shape[0]), dtype=int) if not self.save_light or self.ach_release else None
		self.CM_all = np.zeros((self.n_runs, self.n_classes, self.n_classes))
		self.perf_all = np.zeros(self.n_runs)
		self.perf_train_prog = np.ones((self.n_runs, self.n_epi_tot))*-1
		self.perf_test_prog = np.ones((self.n_runs, self.n_epi_tot))*-1 if self.test_each_epi else None
		self.log_likelihood_prog = np.ones((self.n_runs, self.n_epi_tot))*-1 if self.test_each_epi else None
		self._labels2idx = ex.set_labels2idx(self.classes)
		self._train_class_layer = True if self.classifier=='neural_dopa'  else False
		self._n_batches = int(np.ceil(float(self.n_images)/self.batch_size))
		self._saved_perf_size = (self.n_images, self.ach_avg) if self.ach_stim else (self.n_classes, self.ach_avg)
		self.stim_perf_saved = np.ones((self.n_runs, self._saved_perf_size[0], self._saved_perf_size[1]))*np.nan
		self.stim_perf_labels_saved = np.ones((self.n_runs, self.n_images))*np.nan if not self.save_light else np.zeros(1)
		self.ach_tracker = np.ones((self.n_images, self.n_epi_tot))*np.nan if not self.save_light else np.zeros(self.n_images)
		# self.dopa_tracker = np.array([])
		# self.RP_tracker = np.array([])
		# self.RPE_tracker = np.array([])
		# self.decision_tracker_greedy = np.zeros((self.n_epi_perc, self.n_images), dtype=int) ###
		# self.decision_tracker_explore = np.zeros((self.n_epi_perc, self.n_images), dtype=int) ###
		# self.posterior_tracker_greedy = np.zeros((self.n_epi_perc, self.n_images, self.n_classes)) ###
		# self.posterior_tracker_explore = np.ones((self.n_epi_perc, self.n_images, self.n_classes))*np.nan

		if self.verbose: 
			print 'seed: ' + str(self.seed) + '\n'
			print 'run:  ' + self.name
			print '\ntraining network...'
		""" execute multiple training runs """
		for r in range(self.n_runs):
			self._r = r
			if self.verbose: print '\nrun: %d' %r

			np.random.seed(self.seed+r)
			self._init_weights(images_train)
			self._W_in_since_update = np.copy(self.hid_W)
			if self.protocol=='digit' and self.shuffle_datasets: #shuffle train and test datasets for each independent run
				images_train, images_test, labels_train, labels_test, idx_train, idx_test = ex.shuffle_datasets(images_dict, labels_dict, self._idx_shuffle)
			elif self.protocol=='gabor':
				if r != 0: #reload new training gabor filter
					images_dict_new, labels_dict_new, _, _ = ex.load_images(self.protocol, self.A, self.verbose, gabor_params=self.images_params)
					images_train, images_task = images_dict_new['train'], images_dict_new['task']
					labels_train, labels_task = labels_dict_new['train'], labels_dict_new['task']
				if self.images_params['noise_pixel'] > 0.0:
					gaussian_noise = np.random.normal(0.0, self.images_params['noise_pixel'], size=np.shape(images_train))
				else:
					gaussian_noise = np.zeros(np.shape(images_train))
			elif self.protocol=='toy_data' and not self.pypet:
				images_train, images_test, labels_train, labels_test, idx_train, idx_test = ex.shuffle_datasets(images_dict, labels_dict, self._idx_shuffle)
				an.assess_toy_data(self, images_train, labels_train, os.path.join('.', 'output', self.name, 'result_init'))
			images_rndm, labels_rndm = images_train, labels_train

			""" train network """
			for e in range(self.n_epi_tot):
				self._e = e
				stim_perf_epi = np.empty(0)

				#save weights just after the end of statistical pre-training
				if e == self.n_epi_crit and self.verbose:
					print '----------end crit-----------'
				if e == self.n_epi_crit + self.n_epi_fine:
					self.hid_W_naive[r,:,:] = np.copy(self.hid_W)
					self.out_W_naive[r,:,:] = np.copy(self.out_W)
					if self.verbose: print '----------end fine-----------'
				if e == self.n_epi_crit + self.n_epi_fine + self.n_epi_perc and self.verbose: 
					print '----------end dopa-----------'
				
				#shuffle or create new input images
				if self.protocol=='gabor' and e >= self.n_epi_crit:
					if self.images_params['renew_trainset']: #create new training images
						rnd_orientations= np.random.random(self.images_params['n_train'])*self.images_params['excentricity']*2 + self.images_params['target_ori'] - self.images_params['excentricity']
						images_rndm, labels_rndm = ex.generate_gabors(rnd_orientations, self.images_params['target_ori'], self.images_params['im_size'])
					else: 
						images_rndm, labels_rndm = ex.shuffle([images_task, labels_task])
				else:
					if not self.ach_stim:
						images_rndm, labels_rndm, self.ach_tracker = ex.shuffle([images_rndm, labels_rndm, self.ach_tracker])
					else:
						images_rndm, labels_rndm, self._stim_perf, idx_train, self.ach_tracker = ex.shuffle([images_rndm, labels_rndm, self._stim_perf, idx_train, self.ach_tracker])

				#add noise to gabor filter images
				if self.protocol=='gabor':
					np.random.shuffle(gaussian_noise)
					images_rndm += gaussian_noise
					images_rndm = ex.normalize(images_rndm, self.A)

				#train network with mini-batches
				correct = 0.
				greedy_all = np.array([], dtype=int)
				for b in range(self._n_batches):
					self._b = b
					#update pdf for bayesian inference
					self._update_pdf(images_rndm, labels_rndm)
				
					#select training images for the current batch
					batch_images = images_rndm[b*self.batch_size:(b+1)*self.batch_size,:]
					batch_labels = labels_rndm[b*self.batch_size:(b+1)*self.batch_size]
					
					#propagate images through the network
					greedy, explore_hid, explore_out, posterior, explorative = self._propagate(batch_images)
					greedy_all = np.append(greedy_all, greedy)

					###
					# self.decision_tracker_greedy[self._e, b*self.batch_size:(b+1)*self.batch_size] = np.argmax(self.out_neurons_greedy,1)
					# self.decision_tracker_explore[self._e, b*self.batch_size:(b+1)*self.batch_size] = np.argmax(self.out_neurons_explore,1)
					# self.posterior_tracker_greedy[self._e, b*self.batch_size:(b+1)*self.batch_size, :] = np.copy(self.out_neurons_greedy)
					# self.posterior_tracker_explore[self._e, b*self.batch_size:(b+1)*self.batch_size, :] = np.copy(self.out_neurons_explore)
					###

					#compute reward prediction
					predicted_reward_hid = ex.reward_prediction(explorative, self.compare_output, self.classes, self.out_neurons_greedy, self.out_neurons_explore, self.dopa_func)
					predicted_reward_out = ex.reward_prediction(explorative, self.compare_output, self.classes, self.out_neurons_greedy, self.out_neurons_explore_hid_out, self.dopa_func) if self._train_class_layer else None

					#compute reward
					reward_hid = ex.reward_delivery(batch_labels, explore_hid)
					reward_out = ex.reward_delivery(batch_labels, explore_out) if self._train_class_layer else None

					# self.RP_tracker = np.append(self.RP_tracker, predicted_reward_hid)
					# self.RPE_tracker = np.append(self.RPE_tracker, reward_hid-predicted_reward_hid)
					# self.RPE_tracker = np.append(self.RPE_tracker, reward_hid-ex.reward_prediction(explorative, self.compare_output, self.classes, self.out_neurons_greedy, self.out_neurons_explore, 'linear'))

					#compute dopa signal
					dopa_hid, dopa_out = self._dopa_release_func(predicted_reward_hid, predicted_reward_out, reward_hid, reward_out)
					if not self.dopa_release: dopa_hid = np.ones(len(batch_labels))

					pred_rew_disc = ex.reward_prediction(explorative, self.compare_output, self.classes, self.out_neurons_greedy, self.out_neurons_explore, 'discrete')
					dopa_disc = ex.compute_dopa(pred_rew_disc, reward_hid, {'dHigh': 4.0, 'dMid':0.01, 'dNeut':-0.25, 'dLow':-1.0}, 'discrete')
					dopa_hid[dopa_disc==-1.0] = -1.0

					# pred_rew_exp = ex.reward_prediction(explorative, self.compare_output, self.classes, self.out_neurons_greedy, self.out_neurons_explore, 'exponential')
					# dopa_xplor = ex.compute_dopa(predicted_reward_hid, reward_hid, {'dHigh': 1.844, 'dMid':1.065}, 'exponential')
					# dopa_xploit = ex.compute_dopa(predicted_reward_hid, reward_hid, {'dHigh': 0.909, 'dMid':1.128}, 'exponential')
					# mask_xplr = self.classes[np.argmax(self.out_neurons_greedy,1)] != self.classes[np.argmax(self.out_neurons_explore,1)]
					# dopa_hid[mask_xplr] = dopa_xplor[mask_xplr]
					# dopa_hid[~mask_xplr] = dopa_xploit[~mask_xplr]

					### to have DA only for a specific class
					# dopa_hid[~np.logical_or(batch_labels==4, explore_hid==4)] = 0.0
					# dopa_hid[~(explore_hid==9)] = 0.0
					# set_trace()
					###

					# self.dopa_tracker = np.append(self.dopa_tracker, dopa_hid)

					#compute ACh signal
					if self.ach_uncertainty:
						sorted_post = np.sort(self.out_neurons_explore, axis=1)
						if self.ach_BvSB:
							stim_perf_epi = np.append(stim_perf_epi, sorted_post[:,-1]-sorted_post[:,-2])
						else:
							stim_perf_epi = np.append(stim_perf_epi, sorted_post[:,-1])
					else:
						stim_perf_epi = np.append(stim_perf_epi, reward_hid)
					ach_hid = self._ach_release_func(batch_labels) if self.ach_release else np.ones(self.batch_size) ##<-----actual labels
					# ach_hid = self._ach_release_func(greedy) if self.ach_release else np.ones(self.batch_size) ##<-----estimated labels

					#block feedback
					if self.block_feedback: dopa_hid = np.ones_like(dopa_hid)*np.mean(dopa_hid)

					#set lr_hid=0 during the 'fine' and 'post' periods ##
					if (e >= self.n_epi_crit and e < self.n_epi_crit + self.n_epi_fine) or e >= self.n_epi_crit + self.n_epi_fine + self.n_epi_perc:
						lr_hid = 0.0
						lr_out = 5e-7
					else: 
						lr_hid = self.lr_hid
						lr_out = self.lr_out

					#update weights
					self.hid_W = self._learning_step(batch_images, self.hid_neurons_explore, self.hid_W, lr=lr_hid, dopa=dopa_hid, ach=ach_hid)
					if self._train_class_layer:
						self.out_W = self._learning_step(self.hid_neurons_greedy, self.out_neurons_explore_out, self.out_W, lr=lr_out, dopa=dopa_out)
					#update weights of probabilistic neural classifier
					if self.classifier=='neural_prob' and self._b%100==0:
						self.out_W = self._learn_out_proba(images_rndm, labels_rndm)

					#keep track of training performance
					correct += np.sum(greedy==batch_labels)

					#track ACh release
					if self.ach_release and not self.save_light: self.ach_tracker[b*self.batch_size:(b+1)*self.batch_size, self._e] = ach_hid

				#assess performance
				self._assess_perf_progress(correct/self.n_images, images_train, labels_train, images_test, labels_test)

				#update tracking of performance for ach release
				if self.ach_approx_class:
					self._update_ach_perf_track(stim_perf_epi, greedy_all)
				else:
					self._update_ach_perf_track(stim_perf_epi, labels_rndm)

				#assess early stop
				if self._assess_early_stop(): break

				if (self.protocol=='toy_data' and e%50==0) and not self.pypet:
					an.assess_toy_data(self, images_train, labels_train, os.path.join('.', 'output', self.name, 'results_'+str(e)))

			#save data
			if self.protocol=='toy_data' and self.pypet:
				an.assess_toy_data(self, images_train, labels_train, os.path.join('.', 'output', self.pypet_name, 'results_final_'+self.name+'_run_'+str(r)))
			elif self.protocol=='toy_data':
				an.assess_toy_data(self, images_train, labels_train, os.path.join('.', 'output', self.name, 'results_final_'+str(r)))
			self.hid_W_trained[r,:,:] = np.copy(self.hid_W)
			self.out_W_trained[r,:,:] = np.copy(self.out_W)
			self.stim_perf_saved[r,:,:] = np.copy(self._stim_perf)
			if 'labels_rndm' in locals() and not self.save_light: self.stim_perf_labels_saved[r,:] = np.copy(labels_rndm)
			if not self.save_light or self.ach_release: self._idx_shuffle_saved[r,:] = np.concatenate((idx_train, idx_test))
			self.test(images_test, labels_test, end_of_run=True)
			if not self.pypet: ex.save_net(self)

		self._train_stop = time.time()
		self.runtime = self._train_stop - self._train_start

		# set_trace()

	def test(self, images, labels, during_training=False, end_of_run=False):
		""" 
		Test Hebbian convolutional neural network

			Args: 
				images (numpy array): 2D image arrays to test the Network on.
				labels (numpy array): corresponding labels of the images.
				during_training (bool, optional): whether testing error is assessed during training of the network (is True, less information is computed)
				end_of_run (bool, optional): whether this is the last testing of a run (if True, perf results are saved)

			returns:
				(dict): confusion matrix and performance of the network for all runs
		"""

		#add noise to gabor filter images
		if self.protocol=='gabor':
			if self.images_params['noise_pixel']>0.0:
				images += np.random.normal(0.0, self.images_params['noise_pixel'], size=np.shape(images)) #add Gaussian noise
				# if self.classifier=='bayesian':
				# 	images_train += np.random.normal(0.0, self.images_params['noise_pixel'], size=np.shape(images_train)) #add Gaussian noise
			images = ex.normalize(images, self.A)

		if self.verbose and not during_training: print "\ntesting network..."

		""" variable initialization """
		n_runs = 1 if during_training or end_of_run else self.n_runs
		if not during_training and not end_of_run: 
			CM_all=np.zeros((self.n_runs, self.n_classes, self.n_classes))
			perf_all=np.zeros(self.n_runs)

		for iw in range(n_runs):
			if during_training or end_of_run:
				hid_W = np.copy(self.hid_W)
				out_W = np.copy(self.out_W)
			else:
				if self.verbose: print 'run: ' + str(iw+1)
				hid_W = self.hid_W_trained[iw,:,:]
				out_W = self.out_W_trained[iw,:,:]

			""" testing of the classifier """
			if self.classifier=='neural_dopa':
				hidNeurons = ex.propagate_layerwise(images, hid_W, SM=False, log_weights=self.log_weights) 
				# hidNeurons += np.random.normal(0, self.noise_activ, np.shape(hidNeurons))## corruptive noise
				hidNeurons = ex.softmax(hidNeurons, t=self.t_hid)

				actNeurons = ex.propagate_layerwise(hidNeurons, out_W, log_weights=self.log_weights)
				classIdx = np.argmax(actNeurons, 1)
				classResults = self.classes[classIdx]
			elif self.classifier=='neural_prob':
				hidNeurons = ex.propagate_layerwise(images, hid_W, SM=False, log_weights=self.log_weights) 
				# hidNeurons += np.random.normal(0, self.noise_activ, np.shape(hidNeurons))## corruptive noise
				hidNeurons = ex.softmax(hidNeurons, t=self.t_hid)

				out_W_normed = out_W/np.sum(out_W, 1)[:,np.newaxis]
				actNeurons = np.dot(hidNeurons, out_W_normed)
				classIdx = np.argmax(actNeurons, 1)
				classResults = self.classes[classIdx]
			elif self.classifier=='bayesian':
				raise NotImplementedError('bayesian classifier not implemented')
				# pdf_marginals, pdf_evidence, pdf_labels = bc.pdf_estimate(images_train, labels_train, hid_W, self.pdf_method, self.t_hid)
				# hidNeurons = ex.propagate_layerwise(images, hid_W, t=self.t_hid, log_weights=self.log_weights)
				# posterior = bc.bayesian_decoder(hidNeurons, pdf_marginals, pdf_evidence, pdf_labels, self.pdf_method)
				# classIdx = np.argmax(posterior, 1)
				# classResults = self.classes[classIdx]
			correct_classif = float(np.sum(classResults==labels))/len(labels)
			
			""" compute classification matrix """
			if not during_training or end_of_run:
				CM = np.zeros((len(self.classes), len(self.classes)))
				for ilabel,label in enumerate(self.classes):
					for iclassif, classif in enumerate(self.classes):
						classifiedAs = np.sum(np.logical_and(labels==label, classResults==classif))
						overTot = np.sum(labels==label)
						CM[ilabel, iclassif] = float(classifiedAs)/overTot

			if not during_training and not end_of_run: 
				CM_all[iw,:,:] = CM
				perf_all[iw] = correct_classif
			if end_of_run:
				self.CM_all[self._r,:,:] = CM
				self.perf_all[self._r] = correct_classif

		if during_training:
			return correct_classif
		elif end_of_run:
			return
		elif not during_training and not end_of_run: 
			return CM_all, perf_all
		
	def _init_weights(self, images=None):
		""" initialize weights of the network, either by loading saved weights from file or by random initialization """
		if self.init_file == 'NO_INIT':
			pass
		if self.init_file != '' and self.init_file != None:
			self._init_weights_file()
		elif self.weight_init == 'random':
			self._init_weights_random()
		elif self.weight_init == 'input' and images is not None:
			self._init_weights_input(images)
		else:
			raise ValueError ('wrong weitgh initialization method: %s' % self.weight_init)

	def _init_weights_file(self):
		""" initialize weights of the network by loading saved weights from file """
		if not os.path.exists(os.path.join('output', self.init_file)):
			raise IOError, "weight file \'%s\' not found" % self.init_file
		f_net = open(os.path.join('output', self.init_file, 'Network'), 'r')
		saved_net = pickle.load(f_net)

		#randomly choose weights from one of the saved runs
		run_to_load = self._r % saved_net.n_runs 
		saved_hid_W = saved_net.hid_W_trained[run_to_load, :, :]
		saved_out_W = saved_net.out_W_trained[run_to_load, :, :]

		if (self.n_inp_neurons, self.n_hid_neurons) != np.shape(saved_hid_W):
			raise ValueError, "Hidden weights loaded from file are not of the same shape as those of the current network"
		if (self.n_hid_neurons, self.n_out_neurons) != np.shape(saved_out_W):
			raise ValueError, "Output weights loaded from file are not of the same shape as those of the current network"

		self.hid_W = np.copy(saved_hid_W)
		self.out_W = np.copy(saved_out_W)
		self._idx_shuffle = np.copy(saved_net._idx_shuffle_saved[run_to_load, :]).astype(int)
		self._stim_perf = np.copy(saved_net.stim_perf_saved[run_to_load, :, :])
		if saved_net.stim_perf_saved[run_to_load, :, :].shape != self._saved_perf_size:
			raise ValueError('loaded stim_perf_saved not the same size as current network\'s')
		self._stim_perf_weights = (np.arange(saved_net.ach_avg, dtype=float)+1)[::-1]
		self._stim_perf_avg = ex.weighted_sum(self._stim_perf, self._stim_perf_weights)
		f_net.close()

	def _init_weights_random(self):
		""" initialize weights of the network randomly or by loading saved weights from file """
		# self.hid_W = np.random.random_sample(size=(self.n_inp_neurons, self.n_hid_neurons)) + 1.0
		# self.out_W = (np.random.random_sample(size=(self.n_hid_neurons, self.n_out_neurons))/1000+1.0)/self.n_hid_neurons

		###
		self.hid_W = np.random.random_sample(size=(self.n_inp_neurons, self.n_hid_neurons))
		# self.hid_W = ex.normalize(self.hid_W.T, self.A).T
		self.hid_W = ex.normalize(self.hid_W.T, self.A*1.1).T
		# self.hid_W = ex.normalize(self.hid_W.T, self.A*1.5*self.A/1e3).T

		self.out_W = (np.random.random_sample(size=(self.n_hid_neurons, self.n_out_neurons))/1000+1.0)/self.n_hid_neurons
		# self.out_W = np.random.random_sample(size=(self.n_hid_neurons, self.n_out_neurons))
		# self.out_W *= 1./np.sum(self.out_W,0) * 2.0
		###
	
		self._stim_perf = np.ones(self._saved_perf_size)*np.nan
		self._stim_perf_weights = (np.arange(self._saved_perf_size[1], dtype=float)+1)[::-1]
		self._stim_perf_avg = np.ones(self._saved_perf_size[0])

	def _init_weights_input(self, images):
		""" initialize weights by using the input statistics """

		m_d = np.zeros_like(images[0])
		for i in xrange(images.shape[0]):
			m_d += images[i]
		m_d /= images.shape[0]
    
		v_d = np.zeros_like(images[0])
		for i in xrange(images.shape[0]):
			v_d += (images[i] - m_d) ** 2
		v_d /= images.shape[0]

		self.hid_W = np.zeros(shape=(self.n_inp_neurons, self.n_hid_neurons), dtype=float)
		for i in xrange(self.n_hid_neurons):
			self.hid_W[:,i] = m_d + 2.*v_d*np.random.random_sample(self.n_inp_neurons)

		self.out_W = (np.random.random_sample(size=(self.n_hid_neurons, self.n_out_neurons))/1000+1.0)/self.n_hid_neurons

		# self.out_W = np.random.random_sample(size=(self.n_hid_neurons, self.n_out_neurons))
		# self.out_W *= 1./np.sum(self.out_W,0) * 2.0

		self._stim_perf = np.ones(self._saved_perf_size)*np.nan
		self._stim_perf_weights = (np.arange(self._saved_perf_size[1], dtype=float)+1)[::-1]
		self._stim_perf_avg = np.ones(self._saved_perf_size[0])

	def _check_parameters(self):
		""" checks if parameters of the Network object are correct """
		if self.name=='':
			raise ValueError('No name provided for the Network object')
		if self.classifier not in ['neural_prob', 'neural_dopa', 'bayesian']:
			raise ValueError( '\'' + self.classifier +  '\' not a legal classifier value. Legal values are: \'neural_dopa\', \'neural_prob\' and \'bayesian\'.')
		if self.protocol not in ['digit', 'gabor', 'toy_data']:
			raise ValueError( '\'' + self.protocol +  '\' not a legal protocol value. Legal values are: \'digit\' and \'gabor\'.')
		if self.pdf_method not in ['fit', 'subsample', 'full']:
			raise ValueError( '\'' + self.pdf_method +  '\' not a legal pdf_method value. Legal values are: \'fit\', \'subsample\' and \'full\'.')

	def _update_pdf(self, images_rndm, labels_rndm, threshold=0.01):
		""" re-compute the pdf for bayesian inference if any weights have changed more than a threshold """
		if self.classifier=='bayesian' and (self._e >= self.n_epi_crit + self.n_epi_fine or self.test_each_epi):
			W_mschange = np.sum((self._W_in_since_update - self.hid_W)**2, 0)
			if (W_mschange/940 > threshold).any() or (self._e==0 and self._b==0):
				self._W_in_since_update = np.copy(self.hid_W)
				self._pdf_marginals, self._pdf_evidence, self._pdf_labels = bc.pdf_estimate(images_rndm, labels_rndm, self.hid_W, self.pdf_method, self.t_hid)

	def _propagate(self, batch_images):
		""" propagate input images through the network, either with a layer of neurons on top or with a bayesian decoder """
		if self.classifier == 'bayesian':
			greedy, explore_hid, explore_out, posterior, explorative = self._propagate_bayesian(batch_images)
		elif self.classifier == 'neural_dopa':
			greedy, explore_hid, explore_out, posterior, explorative = self._propagate_neural_dopa(batch_images)
		elif self.classifier == 'neural_prob':
			greedy, explore_hid, explore_out, posterior, explorative = self._propagate_neural_prob(batch_images)

		return greedy, explore_hid, explore_out, posterior, explorative

	def _propagate_neural_dopa(self, batch_images):
		""" propagate input images through the network with a layer of neurons on top """
		#reset activity (important for cases in which no noise is added)
		self.hid_neurons_greedy = None
		self.hid_neurons_explore = None
		self.out_neurons_greedy = None
		self.out_neurons_explore_hid = None
		self.out_neurons_explore_out = None

		#determine which trial will be explorative (e-greedy)
		self.batch_explorative = ex.exploration(self.epsilon_xplr, batch_images.shape[0])

		#compute activation of hidden neurons
		hid_activ = ex.propagate_layerwise(batch_images, self.hid_W, SM=False, log_weights=self.log_weights) 
		hid_activ_std = np.std(hid_activ)
		# hid_activ += np.random.normal(0, self.noise_activ, np.shape(hid_activ))## corruptive noise

		#add noise to activation of hidden neurons for exploration
		if self.exploration and self._e >= self.n_epi_crit + self.n_epi_fine and self._e < self.n_epi_crit + self.n_epi_fine + self.n_epi_perc and self.dopa_release:
			self.hid_neurons_explore = hid_activ + np.random.normal(0, hid_activ_std*self.noise_xplr_hid, np.shape(hid_activ))*self.batch_explorative[:,np.newaxis]
			self.hid_neurons_explore = ex.softmax(self.hid_neurons_explore, t=self.t_hid)
			self.out_neurons_explore_hid = ex.propagate_layerwise(self.hid_neurons_explore, self.out_W, SM=True, t=self.t_hid, log_weights=self.log_weights)

		#softmax and normalize hidden neurons
		self.hid_neurons_greedy = ex.softmax(hid_activ, t=self.t_hid)

		#compute activation of class neurons in greedy case
		out_activ = ex.propagate_layerwise(self.hid_neurons_greedy, self.out_W, SM=False, log_weights=self.log_weights)

		#adds noise in out_W neurons
		if (self._e < self.n_epi_crit + self.n_epi_fine or self._e >= self.n_epi_crit + self.n_epi_fine + self.n_epi_perc or self.train_out_dopa) and self.exploration:
			self.out_neurons_explore_out = out_activ + np.random.normal(0, np.clip(np.std(out_activ)*self.noise_xplr_out, 1e-10, np.inf), np.shape(out_activ))*self.batch_explorative[:,np.newaxis]
			self.out_neurons_explore_out = ex.softmax(self.out_neurons_explore_out, t=self.t_out)

		#softmax output neurons
		self.out_neurons_greedy = ex.softmax(out_activ, t=self.t_out)
		
		#set activation values for neurons when no exploration
		if self.hid_neurons_explore is None: self.hid_neurons_explore = np.copy(self.hid_neurons_greedy)
		if self.out_neurons_explore_hid is None: self.out_neurons_explore_hid = np.copy(self.out_neurons_greedy)
		if self.out_neurons_explore_out is None: self.out_neurons_explore_out = np.copy(self.out_neurons_greedy)

		#set return variables
		greedy = self.classes[np.argmax(self.out_neurons_greedy,1)]
		explore_hid = self.classes[np.argmax(self.out_neurons_explore_hid,1)]
		explore_out = self.classes[np.argmax(self.out_neurons_explore_out,1)]
		return greedy, explore_hid, explore_out, None, self.batch_explorative

	def _propagate_neural_prob(self, batch_images):
		""" propagate input images through the network, with statistical inference performed at the top layer """
		self.hid_neurons_explore = None

		#determine which trial will be explorative (e-greedy)
		self.batch_explorative = ex.exploration(self.epsilon_xplr, batch_images.shape[0])

		#compute activation of hidden neurons
		hid_activ = ex.propagate_layerwise(batch_images, self.hid_W, SM=False, log_weights=self.log_weights) 
		hid_activ_std = np.std(hid_activ)
		# hid_activ += np.random.normal(0, self.noise_activ, np.shape(hid_activ))## corruptive noise

		#add noise to activation of hidden neurons for exploration
		if self.exploration and self._e >= self.n_epi_crit + self.n_epi_fine and self._e < self.n_epi_crit + self.n_epi_fine + self.n_epi_perc and self.dopa_release:
			self.hid_neurons_explore = hid_activ + np.random.normal(0, hid_activ_std*self.noise_xplr_hid, np.shape(hid_activ))*self.batch_explorative[:,np.newaxis]
			self.hid_neurons_explore = ex.softmax(self.hid_neurons_explore, t=self.t_hid)

		#softmax and normalize hidden neurons
		self.hid_neurons_greedy = ex.softmax(hid_activ, t=self.t_hid)

		if self.hid_neurons_explore is None: self.hid_neurons_explore = np.copy(self.hid_neurons_greedy)

		#compute activation of output neurons
		out_W_normed = self.out_W/np.sum(self.out_W, 1)[:,np.newaxis]
		self.out_neurons_explore = np.dot(self.hid_neurons_explore, out_W_normed)
		self.out_neurons_greedy = np.dot(self.hid_neurons_greedy, out_W_normed)

		#set return variables
		greedy = self.classes[np.argmax(self.out_neurons_greedy,1)]
		explore = self.classes[np.argmax(self.out_neurons_explore,1)]

		return greedy, explore, None, None, self.batch_explorative

	def _learn_out_proba(self, images, labels):
		""" learn output weights """

		hid_activ = ex.propagate_layerwise(images, self.hid_W, SM=True, t=self.t_hid, log_weights=self.log_weights)
		for ic, c in enumerate(self.classes):
			self.out_W[:,ic] = np.mean(hid_activ[labels==c,:],0)

		return self.out_W

	def _propagate_bayesian(self, batch_images):
		""" propagate input images through the network with a bayesian decoder on top """
		raise NotImplementedError('bayesian method out of date since compare_output')
		# #reset activity (important for cases in which no noise is added)
		# self.hid_neurons_greedy = None
		# self.hid_neurons_explore = None

		# #compute activation of hidden neurons
		# hid_activ = ex.propagate_layerwise(batch_images, self.hid_W, SM=False, log_weights=self.log_weights)
		
		# #add noise to activation of hidden neurons (exploration)
		# if self.exploration and self._e >= self.n_epi_crit + self.n_epi_fine:
		# 	self.hid_neurons_explore = hid_activ + np.random.normal(0, np.std(hid_activ)*self.noise_xplr_hid, np.shape(hid_activ))
		# 	self.hid_neurons_explore = ex.softmax(self.hid_neurons_explore, t=self.t_hid)

		# #softmax hidden neurons
		# self.hid_neurons_greedy = ex.softmax(hid_activ, t=self.t_hid)
		
		# #set activation values for neurons when no exploration
		# if self.hid_neurons_explore is None: self.hid_neurons_explore = np.copy(self.hid_neurons_greedy)

		# #compute posteriors of the bayesian decoder in greedy and explorative cases
		# if self._e >= self.n_epi_crit + self.n_epi_fine:
		# 	posterior_greedy = bc.bayesian_decoder(self.hid_neurons_greedy, self._pdf_marginals, self._pdf_evidence, self._pdf_labels, self.pdf_method)
		# 	greedy = self.classes[np.argmax(posterior_greedy,1)]
			
		# 	posterior_explore = bc.bayesian_decoder(self.hid_neurons_explore, self._pdf_marginals, self._pdf_evidence, self._pdf_labels, self.pdf_method)
		# 	explore = self.classes[np.argmax(posterior_explore,1)]
		# else:
		# 	posterior_greedy = None
		# 	greedy = None
		# 	explore = None		

		# return greedy, explore, None, posterior_greedy

	def _dopa_release_func(self, predicted_reward_hid, predicted_reward_out, reward_hid, reward_out):
		""" compute dopa release based on predicted and delivered reward """
		if (self._e < self.n_epi_crit + self.n_epi_fine or self._e >= self.n_epi_crit + self.n_epi_fine + self.n_epi_perc) and self._train_class_layer:
			""" Critical and Post period """
			dopa_hid = np.ones(len(reward_hid))
			dopa_out = ex.compute_dopa(predicted_reward_out, reward_out, self.dopa_values_out, self.dopa_func)
	
		elif self._e >= self.n_epi_crit + self.n_epi_fine and self._e < self.n_epi_crit + self.n_epi_fine + self.n_epi_perc: 
			""" Perceptual learning period """
			dopa_hid = ex.compute_dopa(predicted_reward_hid, reward_hid, self.dopa_values, self.dopa_func)
			## add parallel out training here
			dopa_out = np.zeros(len(reward_hid))
		else:
			dopa_hid = np.ones(len(reward_hid))
			dopa_out = np.ones(len(reward_hid))

		return dopa_hid, dopa_out

	def _ach_release_func(self, labels=None):
		""" compute ach realease based on average performance for each stimulus """
		if self._e >= self.n_epi_crit + self.n_epi_fine and self._e < self.n_epi_crit + self.n_epi_fine + self.n_epi_perc and self.ach_release: #ACh starts at perc
		# if self.ach_release: #ACh starts at crit
			if hasattr(self, 'ach_stim') and (self.ach_stim and self.ach_uncertainty):
				self._stim_perf_avg = 0.80 ##np.nanmean(self._stim_perf) ##<--------set mean perf; only to speed-up computations
			else:
				self._stim_perf_avg = ex.weighted_sum(self._stim_perf, self._stim_perf_weights)

			if self.ach_func=='preset': #uses preset values (only valid of 1-4-9)
				ach = np.ones_like(labels, dtype=float)
				ach[labels==1] = 0.17 #0.17 #1.00 #1.0
				ach[labels==4] = 1.82 #1.82 #10.64 #9.0
				ach[labels==9] = 1.02 #1.02 #5.98 #4.0
			elif self.ach_func=='labels': #uses the labels for the value of ACh; used for 'motivation' or 'relevance' aspect of ACh
				ach = np.ones_like(labels, dtype=float)
				for l in np.unique(labels):
					ach[labels==l] = l+1
			elif self.ach_func=='labels_reverse': #uses the reverse labels for the value of ACh; used for 'motivation' or 'relevance' aspect of ACh
				ach = np.ones_like(labels, dtype=float)
				for l in np.unique(labels):
					ach[labels==l] = abs(l-(self.n_classes))
			else:
				if self.ach_stim: #average over stimuli
					if self.ach_uncertainty: #uses uncertainty of current stimulus
						rel_perf = np.nanmean(self._stim_perf[self._b*self.batch_size:(self._b+1)*self.batch_size, :],1)/self._stim_perf_avg ##averaged over 20 episodes
						# rel_perf = np.max(self.out_neurons_explore, axis=1)/self._stim_perf_avg ##single stimuli
					else:
						perf_avg = self._stim_perf_avg[self._b*self.batch_size:(self._b+1)*self.batch_size]
						rel_perf = perf_avg/np.mean(self._stim_perf_avg)
				else: #average over classes
					rel_perf_classes = self._stim_perf_avg/np.mean(self._stim_perf_avg)
					rel_perf = rel_perf_classes[self._labels2idx[labels]]
				ach = self.ach_func(rel_perf, **self.ach_values)

		else:
			ach = np.ones_like(labels)
		return ach

	def _update_ach_perf_track(self, stim_perf_epi, labels):
		""" updates the tracking of performance for ACh release """
		self._stim_perf = np.roll(self._stim_perf, 1, axis=1)
		#average over stimuli
		if self._saved_perf_size[0]==self.n_images: 
			self._stim_perf[:,0] = stim_perf_epi
		#average over classes
		elif self._saved_perf_size[0]==self.n_classes: 
			for c in range(self.n_classes):
				self._stim_perf[c,0] = np.mean(stim_perf_epi[labels==self.classes[c]])
		else:
			raise ValueError('save performance matrix of wrong shape')

	def _learning_step(self, pre_neurons, post_neurons, W, lr, dopa=None, ach=None, numba=True):
		"""
		One learning step for the hebbian network

		Args:
			pre_neurons (numpy array): activation of the pre-synaptic neurons
			post_neurons (numpy array): activation of the post-synaptic neurons
			W (numpy array): weight matrix
			lr (float): learning rate
			dopa (numpy array, optional): learning rate increase for the effect of acetylcholine and dopamine

		returns:
			numpy array: change in weight; must be added to the weight matrix W
		"""
		if dopa is None or dopa.shape[0]!=post_neurons.shape[0]: dopa=np.ones(post_neurons.shape[0])
		if ach is None or ach.shape[0]!=post_neurons.shape[0]: ach=np.ones(post_neurons.shape[0])

		if numba:
			postNeurons_lr = ex.disinhibition(post_neurons, lr, dopa, ach, np.zeros_like(post_neurons))
			dot = np.dot(pre_neurons.T, postNeurons_lr)
			dW = ex.regularization(dot, postNeurons_lr, W, np.zeros(postNeurons_lr.shape[1]))
		else:
			postNeurons_lr = post_neurons * (lr * dopa[:,np.newaxis] * ach[:,np.newaxis]) #adds the effect of dopamine and acetylcholine to the learning rate  
			dW = (np.dot(pre_neurons.T, postNeurons_lr) - np.sum(postNeurons_lr, 0)*W)

		#update weights		
		if self.lim_weights and self._e>=self.n_epi_crit + self.n_epi_fine and pre_neurons.shape[1]==784: #prevents weight change for entire hidden neuron if any weight of this neuron would become negative
			mask = np.all(W+dW>0.0, axis=0)
		else:
			mask = np.ones(np.size(W,1), dtype=bool)

		W[:,mask] += dW[:,mask]
		# W = np.clip(W, 1e-10, np.inf) ##no clipping
		
		return W

	def _assess_early_stop(self):
		""" assesses whether to stop training if performance saturates after a given number of episodes; returns True to stop and False otherwise """
		if self.early_stop:
			#check if performance is maximal
			if self._e>=2:
				cond_train = (self.perf_train_prog[self._r, self._e-1:self._e+1]==1.0).all()
				if self.test_each_epi:
					cond_test = (self.perf_test_prog[self._r, self._e-1:self._e+1]==1.0).all()
				else:
					cond_test = True
				if np.logical_and(cond_train, cond_test):
					print "----------early stop condition reached: performance reached 100.0%----------"
					self._early_stop_cond.append({'epi':self._e, 'epi_cond':'max_perf', 'threshold_cond':'max_perf'})
					return True

			#check if performance is minimal
			cond_train = self.perf_train_prog[self._r, self._e] < 1./self.n_out_neurons+1e-5
			if self.test_each_epi:
				cond_test = self.perf_test_prog[self._r, self._e] < 1./self.n_out_neurons+1e-5
			else:
				cond_test = True
			if np.logical_and(cond_train, cond_test):
				print "----------early stop condition reached: performance reached chance level of %.2f%%----------" %((1./self.n_out_neurons)*100.)
				self._early_stop_cond.append({'epi':self._e, 'epi_cond':'min_perf', 'threshold_cond':'min_perf'})
				return True

			#check if perfmance is decreasing
			n_epi=5
			if self._e>=n_epi:
				perf = self.perf_train_prog[self._r, self._e-n_epi:self._e]
				cond_train = ((np.roll(perf,-1)-perf)[:-1]<0).all()
				if self.test_each_epi:
					perf = self.perf_test_prog[self._r, self._e-n_epi:self._e]
					cond_test = ((np.roll(perf,-1)-perf)[:-1]<0).all()
				else:
					cond_test = True
				if np.logical_and(cond_train, cond_test):
					print "----------early stop condition reached: performance decreased for %d episodes----------" %n_epi
					self._early_stop_cond.append({'epi':self._e, 'epi_cond':'max_perf', 'threshold_cond':'max_perf'})
					return True

			#check if performance reached a plateau
			n_epi 		= [10, 		20]
			threshold 	= [0.0001,	0.0005]
			for e, t in zip(n_epi, threshold):
				if self._e>=e:
					#condition for training performance
					p_range_train = self.perf_train_prog[self._r, self._e-e:self._e]
					cond_train = np.max(p_range_train)-np.min(p_range_train) <= t
					#condition for testing performance
					if self.test_each_epi:
						p_range_test = self.perf_test_prog[self._r, self._e-e:self._e]
						cond_test = np.max(p_range_test)-np.min(p_range_test) <= t
					else:
						cond_test = True
					if np.logical_and(cond_train, cond_test):
						print "----------early stop condition reached: %d episodes with equal or less than %.2f%% change in performance----------" %(e, t*100.)
						self._early_stop_cond.append({'epi':self._e, 'epi_cond':e, 'threshold_cond': t})
						return True
		return False

	def _assess_perf_progress(self, perf_train, images_train, labels_train, images_test, labels_test):
		""" assesses progression of performance of network as it is being trained """
		
		print_perf = 'epi ' + str(self._e) + ': '
		if self.test_each_epi and self._train_class_layer: ##remove neural_prob... 
			correct_out_W = self._check_out_W(images_train, labels_train)
			print_perf += 'correct out weights: %d/%d ; ' %(correct_out_W, self.n_hid_neurons)
		if self.test_each_epi and False: ## remove bool flag to measure likelihood at each episode
			log_likelihood = self._assess_loglikelihood(images_train[::100,:], labels_train[::100])
			print_perf += 'log-likelihood: %.2f ; ' %(log_likelihood)
			self.log_likelihood_prog[self._r, self._e] = log_likelihood
		if self.classifier=='neural_dopa' or self.classifier=='neural_prob' or self._e>=self.n_epi_crit + self.n_epi_fine:
			print_perf += 'train performance: %.2f%%' %(perf_train*100)
		else:
			print_perf += 'train performance: ' + '-N/A-'
		if self.test_each_epi:
			perf_test = self.test(images_test, labels_test, during_training=True)
			print_perf += ' ; test performance: %.2f%%' %(perf_test*100)
			self.perf_test_prog[self._r, self._e] = perf_test
		if self.verbose: print print_perf

		self.perf_train_prog[self._r, self._e] = perf_train

		#save weights just after the end of statistical pre-training
		if self._e==self.n_epi_crit+self.n_epi_fine-1:
			self.hid_W_naive[self._r,:,:] = np.copy(self.hid_W)
			self.out_W_naive[self._r,:,:] = np.copy(self.out_W)

	def _assess_loglikelihood(self, images, labels):
		""" assesses log-likelihood of the data under the model """

		W_1 = self.hid_W
		W_2 = self.out_W
		N = images.shape[0]
		C = self.n_hid_neurons
		D = self.n_inp_neurons
		K = self.n_out_neurons if W_2 is not None else None

		if W_2 is None:
			ones = np.ones(shape=(D,C), dtype=float)
			log_likelihood = 0.
			for npicture in xrange(N):
				log_poisson = -W_1 + ones*images[npicture,:][:, np.newaxis] * np.log(W_1) - np.log(scipy.special.gamma(ones*images[npicture,:][:, np.newaxis]+1))
				sum_log_poisson = np.sum(log_poisson,axis=0)
				a = np.min(sum_log_poisson)
				log_likelihood += -np.log(C) + a + np.log(np.sum(np.exp(sum_log_poisson - a)))
		
		elif W_2 is not None and labels is not None:
			ones = np.ones(shape=(D,C), dtype=float)
			log_likelihood = 0.
			for npicture in xrange(N):
				log_poisson = -W_1 + ones*images[npicture,:][:, np.newaxis] * np.log(W_1) - np.log(scipy.special.gamma(ones*images[npicture,:][:, np.newaxis]+1))
				sum_log_poisson = np.sum(log_poisson,axis=0)
				a = np.min(sum_log_poisson)
				if (labels[npicture] is None):
					log_likelihood += a + np.log(np.sum(np.exp(sum_log_poisson - a)*np.sum(W_2,axis=1)/float(K)))
				else:
					log_likelihood += a + np.log(np.sum(np.exp(sum_log_poisson - a)*np.sum(W_2[:,np.argwhere(self.classes==labels[npicture])][0][0])/float(K)))

		return log_likelihood

	def _check_out_W(self, images, labels, RFproba=None):
		""" check out_W assignment after each episode """
		if RFproba is None:
			if self.protocol=='digit':
				RFproba = an.hist(self, images, labels, verbose=False)['RFproba']
			elif self.protocol=='gabor':
				_, pref_ori = gr.tuning_curves(self.hid_W[np.newaxis,:,:], self.t_hid, self.A, self.images_params, self.name, curve_method='no_softmax', plot=False, log_weights=self.log_weights)
				RFproba = np.zeros((1, self.n_hid_neurons, self.n_out_neurons), dtype=int)
				RFproba[0,:,:][pref_ori[0,:] <= 0] = [1,0]
				RFproba[0,:,:][pref_ori[0,:] > 0] = [0,1]
			elif self.protocol=='toy_data':
				RFproba=np.zeros((1,self.n_hid_neurons,3)) ##temporary fix
		same = np.argmax(RFproba[0],1) == self.classes[np.argmax(self.out_W,1)]
		correct_out_W = 0.
		correct_out_W += np.sum(same)
		
		return correct_out_W/len(RFproba)






