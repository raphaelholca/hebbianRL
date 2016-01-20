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
import pickle
import os

ex = reload(ex)
gr = reload(gr)
bc = reload(bc)
an = reload(an)

class Network:
	""" Hebbian neural network with dopamine-inspired learning """

	def __init__(self, dHigh, dMid, dNeut, dLow, name='net', n_runs=1, n_epi_crit=10, n_epi_dopa=10, t=0.1, A=1.2, n_hid_neurons=49, lim_weights=False, lr=0.01, noise_std=0.2, exploration=True, pdf_method='fit', batch_size=20, protocol='digit', classifier='neural', init_file=None, test_each_epi=False, verbose=True, seed=None):

		"""
		Sets network parameters 

			Args:
				dHigh (float): values of dopamine release for -reward expectation, +reward delivery
				dMid (float): values of dopamine release for +reward expectation, +reward delivery
				dNeut (float): values of dopamine release for -reward expectation, -reward delivery
				dLow (float): values of dopamine release for +reward expectation, -reward delivery
				name (str, optional): name of the folder where to save results. Default: 'net'
				n_runs (int, optional): number of runs. Default: 1
				n_epi_crit (int, optional): number of 'critical period' episodes in each run (episodes when reward is not required for learning). Default: 10
				n_epi_dopa (int, optional): number of 'adult' episodes in each run (episodes when reward is not required for learning). Default: 10
				t (float, optional): temperature of the softmax function (t<<1: strong competition; t>=1: weak competition). Default: 0.1
				A (float, optional): input normalization constant. Will be used as: (input size)*A. Default: 1.2
				n_hid_neurons (int, optional): number of hidden neurons. Default: 49
				lim_weights (bool, optional): whether to artificially limit the value of weights. Used during parameter exploration. Default: False
				lr (float, optiona): learning rate. Default: 0.01
				noise_std (float, optional): parameter of the standard deviation of the normal distribution from which noise is drawn. Default: 0.2
				exploration (bool, optional): whether to take take explorative decisions (True) or not (False). Default: True
				pdf_method (str, optional): method used to approximate the pdf; valid: 'fit', 'subsample', 'full'. Default: 'fit'
				batch_size (int, optional): mini-batch size. Default: 20
				protocol (str, optional): training protocol. Possible values: 'digit' (MNIST classification), 'gabor' (orientation discrimination). Default: 'digit'
				classifier (str, optional): which classifier to use for performance assessment. Possible values are: 'neural', 'bayesian'. Default: 'neural'
				init_file (str, optional): initialize weights with pre-trained weights saved to file; use '' or 'None' for random initialization. Default: None
				test_each_epi (bool, optional): whether to test the network's performance at each episode with test data. Default: False
				verbose	(bool, optional): whether to create text output. Default: True
				seed (int, optional): seed of the random number generator. Default: None
		"""
		
		self.dopa_values 	= {'dHigh': dHigh, 'dMid':dMid, 'dNeut':dNeut, 'dLow':dLow}
		self.name 			= name
		self.n_runs 		= n_runs
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
		self.batch_size 	= batch_size
		self.protocol		= protocol
		self.classifier		= classifier
		self.init_file		= init_file
		self.test_each_epi	= test_each_epi
		self.verbose 		= verbose
		self.seed 			= seed
	
		self._check_parameters()
		np.random.seed(self.seed)

	def train(self, images_dict, labels_dict, images_params={}):
		""" 
		Train Hebbian neural network

			Args: 
				images_dict (dict): dictionary of 2D image arrays to test the Network on, with keys: 'train', 'test'.
				labels_dict (dict): dictionary of label arrays of the images.
				images_params (dict, optional): parameters used to create the images
		"""

		images, images_task = images_dict['train'], images_dict['task']
		labels, labels_task = labels_dict['train'], labels_dict['task']

		self.images_params = images_params
		self.classes = np.sort(np.unique(labels))
		self.n_out_neurons = len(self.classes)
		self.n_inp_neurons = np.size(images,1)
		self.n_epi_tot = self.n_epi_crit + self.n_epi_dopa
		self.hid_W_naive = np.zeros((self.n_runs, self.n_inp_neurons, self.n_hid_neurons))
		self.hid_W_trained = np.zeros((self.n_runs, self.n_inp_neurons, self.n_hid_neurons))
		self.out_W_naive = np.zeros((self.n_runs, self.n_hid_neurons, self.n_out_neurons))
		self.out_W_trained = np.zeros((self.n_runs, self.n_hid_neurons, self.n_out_neurons))
		self.perf_train_prog = np.ones((self.n_runs, self.n_epi_tot))*-1
		self.perf_test_prog = np.ones((self.n_runs, self.n_epi_tot))*-1 if self.test_each_epi else None
		self._train_class_layer = False if self.classifier=='bayesian' else True
		n_images = np.size(images,0)
		n_batches = int(np.ceil(float(n_images)/self.batch_size))

		if self.verbose: 
			print 'seed: ' + str(self.seed) + '\n'
			print 'run:  ' + self.name
			print '\ntraining network...'
		
		""" execute multiple training runs """
		for r in range(self.n_runs):
			if self.verbose: print '\nrun: ' + str(r+1)

			np.random.seed(self.seed+r)
			self._init_weights()
			self._W_in_since_update = np.copy(self.hid_W)

			""" train network """
			for e in range(self.n_epi_tot):
				if self.verbose and e==self.n_epi_crit: print '----------end crit-----------'

				#shuffle input images
				if self.protocol=='digit' or (self.protocol=='gabor' and e < self.n_epi_crit):
					rnd_images, rnd_labels = ex.shuffle([images, labels])
				elif self.protocol=='gabor' and e >= self.n_epi_crit:
					rnd_images, rnd_labels = ex.shuffle([images_task, labels_task])

				correct = 0.

				#train network with mini-batches
				for b in range(n_batches):
					
					#update pdf for bayesian inference
					self._update_pdf(e, b, rnd_images, rnd_labels)
				
					#select training images for the current batch
					b_images = rnd_images[b*self.batch_size:(b+1)*self.batch_size,:]
					b_labels = rnd_labels[b*self.batch_size:(b+1)*self.batch_size]
					
					#propagate images through the network
					out_greedy, out_explore, posterior = self._propagate(b_images, e)

					#compute reward prediction
					predicted_reward = ex.reward_prediction(out_greedy, out_explore, posterior=posterior)

					#compute reward
					reward = ex.reward_delivery(b_labels, out_explore)

					#compute dopa signal
					dopa_hid, dopa_out = self._dopa_release(e, predicted_reward, reward)
						
					#update weights
					hid_W = self._learning_step(b_images, self.hid_neurons, self.hid_W, lr=self.lr, dopa=dopa_hid)
					if self._train_class_layer: 
						out_W = self._learning_step(self.hid_neurons, self.out_neurons, self.out_W, lr=self.lr*1e-4, dopa=dopa_out)

					correct += np.sum(out_greedy == b_labels)

				#assess performance
				self._assess_perf_progress(correct/n_images, r, e, images_dict, labels_dict)

			#save data
			self.hid_W_trained[r,:,:] = np.copy(self.hid_W)
			self.out_W_trained[r,:,:] = np.copy(self.out_W)

	def test(self, images_dict, labels_dict, during_training=False):
		""" 
		Test Hebbian convolutional neural network

			Args: 
				images_dict (dict): dictionary of 2D image arrays to test the Network on, with keys: 'train', 'test'.
				labels_dict (dict): dictionary of label arrays of the images.
				during_training (bool, optional): whether testing error is assessed during training of the network (is True, less information is computed)

			returns:
				(dict): confusion matrix and performance of the network for all runs
		"""

		images_train, images_test = images_dict['train'], images_dict['test']
		labels_train, labels_test = labels_dict['train'], labels_dict['test']

		if self.verbose and not during_training: print "\ntesting network..."

		""" variable initialization """
		CM_all = []
		perf_all = []
		n_runs = self.n_runs if not during_training else 1

		for iw in range(n_runs):
			if not during_training:
				if self.verbose: print 'run: ' + str(iw+1)
				hid_W = self.hid_W_trained[iw,:,:]
				out_W = self.out_W_trained[iw,:,:]
			else:
				hid_W = np.copy(self.hid_W)
				out_W = np.copy(self.out_W)

			""" testing of the classifier """
			if self.classifier=='neural':
				hidNeurons = ex.propagate_layerwise(images_test, hid_W, t=self.t)
				actNeurons = ex.propagate_layerwise(hidNeurons, out_W)
				classIdx = np.argmax(actNeurons, 1)
				classResults = self.classes[classIdx]
			elif self.classifier=='bayesian':
				pdf_marginals, pdf_evidence, pdf_labels = bc.pdf_estimate(images_train, labels_train, hid_W, self.pdf_method, self.t)
				hidNeurons = ex.propagate_layerwise(images_test, hid_W, t=self.t)
				posterior = bc.bayesian_decoder(hidNeurons, pdf_marginals, pdf_evidence, pdf_labels, self.pdf_method)
				classIdx = np.argmax(posterior, 1)
				classResults = self.classes[classIdx]
			correct_classif = float(np.sum(classResults==labels_test))/len(labels_test)
			perf_all.append(correct_classif)
			
			""" compute classification matrix """
			if not during_training:
				CM = np.zeros((len(self.classes), len(self.classes)))
				for ilabel,label in enumerate(self.classes):
					for iclassif, classif in enumerate(self.classes):
						classifiedAs = np.sum(np.logical_and(labels_test==label, classResults==classif))
						overTot = np.sum(labels_test==label)
						CM[ilabel, iclassif] = float(classifiedAs)/overTot
				CM_all.append(CM)

		""" create classification results to save """
		if not during_training:
			CM_avg = np.mean(CM_all,0)
			CM_ste = np.std(CM_all,0)/np.sqrt(np.shape(CM_all)[0])
			perf_avg = np.mean(perf_all)
			perf_ste = np.std(perf_all)/np.sqrt(len(perf_all))
			self.perf_dict = {'CM_all':CM_all, 'CM_avg':CM_avg, 'CM_ste':CM_ste, 'perf_all':perf_all, 'perf_avg':perf_avg, 'perf_ste':perf_ste}

			return self.perf_dict
		else:
			return correct_classif

	def _init_weights(self):
		""" initialize weights of the network, either by loading saved weights from file or by random initialization """
		if self.init_file != '' and self.init_file != None:
			self._init_weights_file()
		else:
			self._init_weights_random()

	def _init_weights_file(self):
		""" initialize weights of the network by loading saved weights from file """
		if not os.path.exists(self.init_file):
			raise IOError, "weight file \'%s\' not found" % self.init_file

		f_W_in = open('output/' + self.init_file + '/hid_W', 'r')
		self.hid_W = pickle.load(f_W_in)['000']
		f_W_in.close()

		f_W_act = open('output/' + self.init_file + '/out_W', 'r')
		self.out_W = pickle.load(f_W_act)['000']
		f_W_act.close()

	def _init_weights_random(self):
		""" initialize weights of the network randomly or by loading saved weights from file """
		self.hid_W = np.random.random_sample(size=(self.n_inp_neurons, self.n_hid_neurons)) + 1.0
		self.out_W = (np.random.random_sample(size=(self.n_hid_neurons, self.n_out_neurons))/1000+1.0)/self.n_hid_neurons
	
	def _check_parameters(self):
		""" checks if parameters of the Network object are correct """
		if self.classifier not in ['neural', 'bayesian']:
			raise ValueError( '\'' + self.classifier +  '\' not a legal classifier value. Legal values are: \'neural\' and \'bayesian\'.')
		if self.protocol not in ['digit', 'gabor']:
			raise ValueError( '\'' + self.protocol +  '\' not a legal protocol value. Legal values are: \'digit\' and \'gabor\'.')
		if self.pdf_method not in ['fit', 'subsample', 'full']:
			raise ValueError( '\'' + self.pdf_method +  '\' not a legal pdf_method value. Legal values are: \'fit\', \'subsample\' and \'full\'.')

	def _update_pdf(self, e, b, rnd_images, rnd_labels, threshold=0.01):
		""" re-compute the pdf for bayesian inference if any weights have changed more than a threshold """
		if self.classifier=='bayesian' and (e >= self.n_epi_crit or self.test_each_epi):
			W_mschange = np.sum((self._W_in_since_update - self.hid_W)**2, 0)
			if (W_mschange/940 > threshold).any() or (e==0 and b==0):
				self._W_in_since_update = np.copy(self.hid_W)
				self._pdf_marginals, self._pdf_evidence, self._pdf_labels = bc.pdf_estimate(rnd_images, rnd_labels, self.hid_W, self.pdf_method, self.t)

	def _propagate(self, b_images, e):
		""" propagate input images through the network, either with a layer of neurons on top or with a bayesian decoder """
		if self.classifier == 'bayesian':
			out_greedy, out_explore, posterior = self._propagate_bayesian(b_images, e)
		else:
			out_greedy, out_explore, posterior = self._propagate_neural(b_images, e)

		return out_greedy, out_explore, posterior

	def _propagate_neural(self, b_images, e):
		""" propagate input images through the network with a layer of neurons on top """
		#compute activation of hidden neurons
		self.hid_neurons = ex.propagate_layerwise(b_images, self.hid_W, SM=False)
		
		#compute activation of class neurons in greedy case
		self.out_neurons = ex.propagate_layerwise(ex.softmax(self.hid_neurons, t=self.t), self.out_W, SM=False)
		out_greedy = self.classes[np.argmax(self.out_neurons,1)]

		#add noise to activation of hidden neurons (exploration)
		if self.exploration and e >= self.n_epi_crit:
			self.hid_neurons += np.random.normal(0, np.std(self.hid_neurons)*self.noise_std, np.shape(self.hid_neurons))
			self.hid_neurons = ex.softmax(self.hid_neurons, t=self.t)
			self.out_neurons = ex.propagate_layerwise(self.hid_neurons, self.out_W, SM=False)
		else:
			self.hid_neurons = ex.softmax(self.hid_neurons, t=self.t)

		#adds noise in out_W neurons
		if e < self.n_epi_crit:
			self.out_neurons += np.random.normal(0, 4.0, np.shape(self.out_neurons))
		
		#compute activation of class neurons in explorative case
		self.out_neurons = ex.softmax(self.out_neurons, t=self.t)
		out_explore = self.classes[np.argmax(self.out_neurons,1)]	

		return out_greedy, out_explore, None

	def _propagate_bayesian(self, b_images, e):
		""" propagate input images through the network with a bayesian decoder on top """
		#compute activation of hidden neurons
		self.hid_neurons = ex.propagate_layerwise(b_images, self.hid_W, SM=False)
		
		#compute posterior of the bayesian decoder in greedy case
		if e >= self.n_epi_crit:
			posterior = bc.bayesian_decoder(ex.softmax(self.hid_neurons, t=self.t), self._pdf_marginals, self._pdf_evidence, self._pdf_labels, self.pdf_method)
			out_greedy = self.classes[np.argmax(posterior,1)]
		else:
			posterior = None
			out_greedy = None

		#add noise to activation of hidden neurons (exploration)
		if self.exploration and e >= self.n_epi_crit:
			self.hid_neurons += np.random.normal(0, np.std(self.hid_neurons)*self.noise_std, np.shape(self.hid_neurons))
			self.hid_neurons = ex.softmax(self.hid_neurons, t=self.t)
		else:
			self.hid_neurons = ex.softmax(self.hid_neurons, t=self.t)

		#compute posterior of the bayesian decoder in explorative case
		if e >= self.n_epi_crit:
			posterior_noise = bc.bayesian_decoder(self.hid_neurons, self._pdf_marginals, self._pdf_evidence, self._pdf_labels, self.pdf_method)
			out_explore = self.classes[np.argmax(posterior_noise,1)]
		else: 
			out_explore = None

		return out_greedy, out_explore, posterior

	def _dopa_release(self, e, predicted_reward, reward):
		""" compute dopa release based on predicted and delivered reward """
		if e < self.n_epi_crit and self._train_class_layer:
			""" critical period; train class layer """
			# dopa_release = ex.compute_dopa(predicted_reward, reward, dHigh=0.0, dMid=0.75, dNeut=0.0, dLow=-0.5) #original param give close to optimal results
			# dopa_release = ex.compute_dopa(predicted_reward, reward, dHigh=dHigh, dMid=dMid, dNeut=dNeut, dLow=dLow)
			dopa_release = ex.compute_dopa(predicted_reward, reward, {'dHigh':0.0, 'dMid':0.2, 'dNeut':-0.3, 'dLow':-0.5})

			dopa_hid = np.ones(self.batch_size)
			dopa_out = dopa_release

		elif e >= self.n_epi_crit: 
			""" Dopa - perceptual learning """
			dopa_release = ex.compute_dopa(predicted_reward, reward, self.dopa_values)

			dopa_hid = dopa_release
			# dopa_out = ex.compute_dopa(out_greedy, out_explore, reward, dHigh=0.0, dMid=0.75, dNeut=0.0, dLow=-0.5) #continuous learning in L2
			dopa_out = np.zeros(self.batch_size)
		else:
			dopa_hid = None
			dopa_out = None

		return dopa_hid, dopa_out

	def _learning_step(self, pre_neurons, post_neurons, W, lr, dopa=None, numba=True):
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

		if numba:
			postNeurons_lr = ex.disinhibition(post_neurons, lr, dopa, np.zeros_like(post_neurons))
			dot = np.dot(pre_neurons.T, postNeurons_lr)
			dW = ex.regularization(dot, postNeurons_lr, W, np.zeros(postNeurons_lr.shape[1]))
		else:
			postNeurons_lr = post_neurons * (lr * dopa[:,np.newaxis]) #adds the effect of dopamine and acetylcholine to the learning rate  
			dW = (np.dot(pre_neurons.T, postNeurons_lr) - np.sum(postNeurons_lr, 0)*W)

		#update weights		
		if self.lim_weights and e>=self.n_epi_crit: #artificially prevents weight explosion; used to dissociate influences in parameter self.exploration
			mask = np.logical_and(np.sum(self.hid_W+hid_dW,0)<=940.801, np.min(self.hid_W+hid_dW,0)>0.2)
		else:
			mask = np.ones(np.size(W,1), dtype=bool)

		W[:,mask] += dW[:,mask]
		W = np.clip(W, 1e-10, np.inf)
		
		return W

	def _assess_perf_progress(self, perf_train, r, e, images_dict, labels_dict):
		""" assesses progression of performance of network as it is being trained """
		if self.verbose:
			print_perf = ''
			if self._train_class_layer:
				correct_out_W = self._check_out_W(images_dict['train'], labels_dict['train'])
				print_perf += 'correct out weights: ' + str(int(correct_out_W)) + '/' + str(int(self.n_hid_neurons)) + '; '
			if self.classifier=='neural' or e>=self.n_epi_crit:
				print_perf += 'train performance: ' + str(np.round(perf_train*100,1)) + '%'
			else:
				print_perf += 'train performance: ' + '-N/A-'
			if self.test_each_epi:
				perf_test = self.test(images_dict, labels_dict, during_training=True)
				print_perf += ' ; test performance: ' + str(np.round(perf_test*100,1)) + '%'
			print print_perf

		self.perf_train_prog[r, e] = perf_train
		if self.test_each_epi: self.perf_test_prog[r, e] = perf_test

		#save weights just after the end of statistical pre-training
		if e==self.n_epi_crit-1:
			self.hid_W_naive[r,:,:] = np.copy(self.hid_W)
			self.out_W_naive[r,:,:] = np.copy(self.out_W)

	def _check_out_W(self, images, labels, RFproba=None):
		""" check out_W assignment after each episode """
		if RFproba is None:
			if self.protocol=='digit':
				RFproba, _ = an.hist(self.name, self.hid_W[np.newaxis,:,:], self.classes, images, labels, save_data=False, verbose=False)
			elif self.protocol=='gabor':
				pref_ori = gr.preferred_orientations(self.hid_W[np.newaxis,:,:], self.t, self.images_params['target_ori'], self.name)
				RFproba = np.zeros((1, self.n_hid_neurons, self.n_out_neurons), dtype=int)
				RFproba[0,:,:][pref_ori[0,:] <= self.images_params['target_ori']] = [1,0]
				RFproba[0,:,:][pref_ori[0,:] > self.images_params['target_ori']] = [0,1]
		
		same = np.argmax(RFproba[0],1) == self.classes[np.argmax(self.out_W,1)]
		correct_out_W = 0.
		correct_out_W += np.sum(same)
		
		return correct_out_W/len(RFproba)






