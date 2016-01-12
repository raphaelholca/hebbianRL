"""
Author: Raphael Holca-Lamarre
Date: 23/10/2014

This function trains a hebbian neural network to learn a representation from the MNIST dataset. It makes use of a reward/relevance signal that increases the learning rate when the network makes a correct state-action pair selection.
"""

import numpy as np
import matplotlib.pyplot as pyplot
import helper.external as ex
import helper.plots as pl
import helper.classifier as cl
import helper.assess_network as an
import helper.grating as gr
import helper.bayesian_decoder as bc
import pickle

ex = reload(ex)
pl = reload(pl)
cl = reload(cl)
an = reload(an)
gr = reload(gr)
bc = reload(bc)

class Network:

	def __init__(self, dopa_values, name, n_run=1, n_epi_crit=10, n_epi_dopa=10, t=0.1, A=1.2, n_hid_neurons=49, lim_weights=False, lr=0.01, noise_std=0.2, exploration=True, pdf_method='fit', n_batch=20, protocol='digit', classifier='actionNeurons', init_file=None, test_each_epi=True, SVM=False, save_data=True, verbose=True, show_W_act=True, sort=None, target=None, seed=None):

		"""
		Sets network parameters 

			Args:
				dopa_values (dict): values of dopamine release for different reward prediction error scenarios
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
				init_file (str, optional): initialize weights with pre-trained weights saved to file; use '' or 'None' for random initialization. Default=None
				test_each_epi (bool, optional): whether to test the network's performance at each episode. Default=True
				SVM (bool, optional): whether to use an SVM or the number of stimuli that activate a neuron to determine the class of the neuron. Default=False
				save_data (bool, optional): whether to save data to disk. Default=True
				verbose	(bool, optional): whether to create text output. Default=True
				show_W_act (bool, optional): whether to display out_W weights on the weight plots. Default=True
				sort (str, optional): sorting methods for weights when displaying. Valid value: None, 'class', 'tSNE'. Default=None
				target (int, optional): target digit (to be used to color plots). Use None if not desired. Default=None
				seed (int, optional): seed of the random number generator. Default=None
		"""
		
		self.dopa_values 	= dopa_values
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
		self.init_file		= init_file
		self.test_each_epi	= test_each_epi
		self.SVM 			= SVM
		self.save_data 		= save_data
		self.verbose 		= verbose
		self.show_W_act 	= show_W_act
		self.sort 			= sort
		self.target 		= target
		self.seed 			= seed

		if self.save_data: 
			self.name = ex.checkdir(self, OW_bool=True)
		else:
			print "!!! ----- not saving data ----- !!! \n"
		
		self._check_parameters()
		np.random.seed(self.seed)


	def train(self, images, labels, orientations, images_test, labels_test, orientations_test, images_task, labels_task, orientations_task):

		""" variable initialization """
		self.classes = np.sort(np.unique(labels))
		self.n_out_neurons = len(self.classes)
		self.n_inp_neurons = np.size(images,1)
		self.n_epi_tot = self.n_epi_crit + self.n_epi_dopa
		self.W_hid_save = {}
		self.W_out_save = {}
		self.perf_save = {}
		self.show_W_act = False if self.classifier=='bayesian' else self.show_W_act
		self._train_class_layer = False if self.classifier=='bayesian' else True
		n_images = np.size(images,0)

		""" training the network """
		if self.verbose: 
			print 'seed: ' + str(self.seed) + '\n'
			print 'run:  ' + self.name
			print '\ntraining hebbian network...'
		
		for r in range(self.n_run):
			np.random.seed(self.seed+r)
			if self.verbose: print '\nrun: ' + str(r+1)

			""" initialize network variables """
			if self.init_file != '' and self.init_file != None:
				self._init_weights_file()
			else:
				self._init_weights_random()

			W_in_since_update = np.copy(self.hid_W)
			perf_epi = []

			for e in range(self.n_epi_tot):
				if self.verbose and e==self.n_epi_crit: print '----------end crit-----------'

				#shuffle input
				if self.protocol=='digit' or (self.protocol=='gabor' and e < self.n_epi_crit):
					rnd_images, rnd_labels = ex.shuffle([images, labels])
				elif self.protocol=='gabor' and e >= self.n_epi_crit:
					rnd_images, rnd_labels = ex.shuffle([images_task, labels_task])

				#train network with mini-batches
				for b in range(int(n_images/self.n_batch)):
					
					#re-compute the pdf for bayesian inference if any weights have changed more than a threshold
					if self.classifier=='bayesian' and (e >= self.n_epi_crit or self.test_each_epi):
						W_mschange = np.sum((W_in_since_update - self.hid_W)**2, 0)
						if (W_mschange/940 > 0.01).any() or (e==0 and b==0): ## > 0.005 
							W_in_since_update = np.copy(self.hid_W)
							pdf_marginals, pdf_evidence, pdf_labels = bc.pdf_estimate(rnd_images, rnd_labels, self.hid_W)
				
					#select batch training images (may leave a few training examples out (< self.n_batch))
					b_images = rnd_images[b*self.n_batch:(b+1)*self.n_batch,:]
					b_labels = rnd_labels[b*self.n_batch:(b+1)*self.n_batch]

					#initialize batch variables
					dopa_release = np.ones(self.n_batch)
					hid_dW = 0.
					out_dW = 0.
					dopa_hid = np.ones(self.n_batch)
					dopa_out = np.zeros(self.n_batch)
					
					#compute activation of hidden and classification neurons
					out_greedy, out_explore = self._propagate(b_images, e)

					#determine predicted reward
					if self.classifier!='bayesian':
						predicted_reward = ex.reward_prediction(out_greedy, out_explore)
					elif self.classifier=='bayesian' and e >= self.n_epi_crit:
						predicted_reward = ex.reward_prediction(out_greedy, out_explore, posterior)

					#compute reward
					reward = ex.reward_delivery(b_labels, out_explore)

					#compute dopa signal
					dopa_hid, dopa_out = self._dopa_release(e, predicted_reward, reward)
						
					#compute weight updates
					hid_W = self._learning_step(b_images, self.hid_neurons, self.hid_W, lr=self.lr, disinhib=dopa_hid)
					if self._train_class_layer: 
						out_W = self._learning_step(self.hid_neurons, self.out_neurons, self.out_W, lr=self.lr*1e-4, disinhib=dopa_out)

				""" end of mini-batch """

				#check Wact assignment after each episode:
				if self.protocol=='digit':
					RFproba, _, _ = an.hist(self, {'000':self.hid_W}, self.classes, images, labels, self.protocol, SVM=self.SVM, save_data=False, verbose=False)
				elif self.protocol=='gabor':
					pref_ori = gr.preferred_orientations({'000':self.hid_W})
					RFproba = np.zeros((1, self.n_hid_neurons, self.n_out_neurons), dtype=int)
					RFproba[0,:,:][pref_ori['000']<=target_ori] = [1,0]
					RFproba[0,:,:][pref_ori['000']>target_ori] = [0,1]
				same = np.argmax(RFproba[0],1) == self.classes[np.argmax(self.out_W,1)]
				if self._train_class_layer:
					correct_W_act = 0.
					correct_W_act += np.sum(same)
					correct_W_act/=len(RFproba)

				#check performance after each episode
				if self.verbose:
					if self._train_class_layer:
						print ('correct action weights: ' + str(int(correct_W_act)) + '/' + str(int(self.n_hid_neurons)) + '; '),
				if self.save_data:
					if r==0 and e==self.n_epi_crit-1:
						if self.protocol=='digit':
							pl.plot_noise_proba(self, self.hid_W, images)
						else:
							pl.plot_noise_proba(self, self.hid_W, images_task)
				if self.test_each_epi and (self.verbose or self.save_data):
					if self.classifier=='bayesian':
						rdn_idx = np.random.choice(len(labels_test), 1000, replace=False)
						_, perf_tmp = cl.bayesian({'000':self.hid_W}, images, labels, images_test[rdn_idx], labels_test[rdn_idx], save_data=False, verbose=False)
					if self.classifier=='actionNeurons':
						_, perf_tmp = cl.actionNeurons(self, {'000':self.hid_W}, {'000':self.out_W}, images_test, labels_test, False, False, self.classes)
					perf_epi.append(perf_tmp[0])
					if self.verbose: print 'performance: ' + str(np.round(perf_tmp[0]*100,1)) + '%'
				elif self.verbose and self._train_class_layer: print 

			""" end of episode """
			#save weights
			self.W_hid_save[str(r).zfill(3)] = np.copy(self.hid_W)
			self.W_out_save[str(r).zfill(3)] = np.copy(self.out_W)
			self.perf_save[str(r).zfill(3)] = np.copy(perf_epi)

		if self.verbose and self._train_class_layer: print 'correct action weight assignment:\n' + str(correct_W_act) + ' out of ' + str(self.n_hid_neurons)

		if self.verbose: print '\nrun: '+self.name + '\n'

	def test(self, images, labels):
		""" compute network performance """

		self.classes = np.sort(np.unique(labels))
		if self.classifier=='actionNeurons':	allCMs, allPerf = cl.actionNeurons(self, self.W_hid_save, self.W_out_save, images, labels, self.save_data, self.verbose, self.classes)
		# if self.classifier=='self.SVM': 		allCMs, allPerf = cl.self.SVM(self, self.name, self.W_hid_save, images, labels, self.classes, self.n_inp_neurons, dataset, self.save_data, self.verbose)
		# if self.classifier=='neuronClass':		allCMs, allPerf = cl.neuronClass(self, self.name, self.W_hid_save, self.classes, RFproba, self.n_inp_neurons, images, labels, self.save_data, self.verbose)
		if self.classifier=='bayesian':			allCMs, allPerf = cl.bayesian(self, self.W_hid_save, images, labels, images_test, labels_test) ##

		return allCMs, allPerf, None ##, correct_W_act/self.n_hid_neurons

	def assess(self, images, labels):
		""" compute histogram of RF self.classes """
		if self.protocol=='digit':
			RFproba, RFclass, _ = an.hist(self, self.W_hid_save, self.classes, images, labels, self.protocol, SVM=self.SVM, save_data=self.save_data, verbose=self.verbose)

		elif self.protocol=='gabor':
			n_bins = 10
			bin_size = 180./n_bins
			orientations_bin = np.zeros(len(orientations), dtype=int)
			for i in range(n_bins): 
				mask_bin = np.logical_and(orientations >= i*bin_size, orientations < (i+1)*bin_size)
				orientations_bin[mask_bin] = i

			pref_ori = gr.preferred_orientations(self.W_hid_save)
			RFproba = np.zeros((self.n_run, self.n_hid_neurons, self.n_out_neurons), dtype=int)
			for r in pref_ori.keys():
				RFproba[int(r),:,:][pref_ori[r]<=target_ori] = [1,0]
				RFproba[int(r),:,:][pref_ori[r]>target_ori] = [0,1]
			_, _, _ = hist(self, self.W_hid_save, range(n_bins), images, orientations_bin, self.protocol, n_bins=n_bins, SVM=self.SVM, save_data=self.save_data, verbose=self.verbose)

		""" compute correct weight assignment in the ouput layer """
		if self._train_class_layer:
			correct_W_act = 0.
			notsame = {}
			for k in self.W_out_save.keys():
				same = np.argmax(RFproba[int(k)],1) == self.classes[np.argmax(self.W_out_save[k],1)]
				notsame[k] = np.argwhere(~same)
				correct_W_act += np.sum(same)
			correct_W_act/=len(RFproba)
		else:
			notsame = None
			correct_W_act = 0.

		""" plot weights """
		if self.save_data:
			if self.show_W_act: W_act_pass=self.W_out_save
			else: W_act_pass=None
			if self.protocol=='digit':
				an.plot(self, self.W_hid_save, RFproba, target=self.target, W_act=W_act_pass, sort=self.sort, notsame=notsame, verbose=self.verbose)
				slopes = {}
			elif self.protocol=='gabor':
				an.plot(self, self.W_hid_save, RFproba, W_act=W_act_pass, notsame=notsame, verbose=self.verbose)
				curves = gr.tuning_curves(self.W_hid_save, method='no_softmax', plot=True) #basic, no_softmax, with_noise
				slopes = gr.slopes(self.W_hid_save, curves, pref_ori)
			if self.test_each_epi:
				pl.perf_progress(self, self.perf_save)

	def save(self):
		"""" save data """
		if self.save_data: ex.save_data(self, self.W_hid_save, self.W_out_save, self.perf_save)

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
		if self.classifier not in ['neuronClass', 'SVM', 'actionNeurons', 'bayesian']:
			raise ValueError( '\'' + self.classifier +  '\' not a legal classifier value. Legal values are: \'neuronClass\', \'SVM\', \'actionNeurons\' and \'bayesian\'.')
		if self.protocol not in ['digit', 'gabor']:
			raise ValueError( '\'' + self.protocol +  '\' not a legal protocol value. Legal values are: \'digit\' and \'gabor\'.')
		if self.pdf_method not in ['fit', 'subsample', 'full']:
			raise ValueError( '\'' + self.pdf_method +  '\' not a legal pdf_method value. Legal values are: \'fit\', \'subsample\' and \'full\'.')

	def _propagate(self, b_images, e):
		if self.classifier == 'bayesian':
			out_greedy, out_explore = self._propagate_bayesian(b_images, e)
		else:
			out_greedy, out_explore = self._propagate_neural(b_images, e)

		return out_greedy, out_explore

	def _propagate_neural(self, b_images, e):
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

		return out_greedy, out_explore

	def _propagate_bayesian(self, b_images, e):
		#compute activation of hidden neurons
		self.hid_neurons = ex.propagate_layerwise(b_images, self.hid_W, SM=False)
		
		#compute posterior of the bayesian decoder in greedy case
		if e >= self.n_epi_crit:
			posterior = bc.bayesian_decoder(ex.softmax(self.hid_neurons, t=self.t), pdf_marginals, pdf_evidence, pdf_labels, self.pdf_method)
			out_greedy = self.classes[np.argmax(posterior,1)]

		#add noise to activation of hidden neurons (exploration)
		if self.exploration and e >= self.n_epi_crit:
			self.hid_neurons += np.random.normal(0, np.std(self.hid_neurons)*self.noise_std, np.shape(self.hid_neurons))
			self.hid_neurons = ex.softmax(self.hid_neurons, t=self.t)
		else:
			self.hid_neurons = ex.softmax(self.hid_neurons, t=self.t)

		#compute posterior of the bayesian decoder in explorative case
		if e >= self.n_epi_crit:
			posterior_noise = bc.bayesian_decoder(self.hid_neurons, pdf_marginals, pdf_evidence, pdf_labels, self.pdf_method)
			out_explore = self.classes[np.argmax(posterior_noise,1)]

		return out_greedy, out_explore

	def _dopa_release(self, e, predicted_reward, reward):
		if e < self.n_epi_crit and self._train_class_layer:
			""" critical period; train class layer """
			# dopa_release = ex.compute_dopa(predicted_reward, reward, dHigh=0.0, dMid=0.75, dNeut=0.0, dLow=-0.5) #original param give close to optimal results
			# dopa_release = ex.compute_dopa(predicted_reward, reward, dHigh=dHigh, dMid=dMid, dNeut=dNeut, dLow=dLow)
			dopa_release = ex.compute_dopa(predicted_reward, reward, {'dHigh':0.0, 'dMid':0.2, 'dNeut':-0.3, 'dLow':-0.5})

			dopa_hid = np.ones(self.n_batch)
			dopa_out = dopa_release

		elif e >= self.n_epi_crit: 
			""" Dopa - perceptual learning """
			dopa_release = ex.compute_dopa(predicted_reward, reward, self.dopa_values)

			dopa_hid = dopa_release
			# dopa_out = ex.compute_dopa(out_greedy, out_explore, reward, dHigh=0.0, dMid=0.75, dNeut=0.0, dLow=-0.5) #continuous learning in L2
			dopa_out = np.zeros(self.n_batch)

		return dopa_hid, dopa_out

	def _learning_step(self, preNeurons, postNeurons, W, lr, disinhib=None, numba=True):
		"""
		One learning step for the hebbian network

		Args:
			preNeurons (numpy array): activation of the pre-synaptic neurons
			postNeurons (numpy array): activation of the post-synaptic neurons
			W (numpy array): weight matrix
			lr (float): learning rate
			disinhib (numpy array, optional): learning rate increase for the effect of acetylcholine and dopamine

		returns:
			numpy array: change in weight; must be added to the weight matrix W
		"""
		if disinhib is None or disinhib.shape[0]!=postNeurons.shape[0]: disinhib=np.ones(postNeurons.shape[0])

		if numba:
			postNeurons_lr = ex.disinhibition(postNeurons, lr, disinhib, np.zeros_like(postNeurons))
			dot = np.dot(preNeurons.T, postNeurons_lr)
			dW = ex.regularization(dot, postNeurons_lr, W, np.zeros(postNeurons_lr.shape[1]))
		else:
			postNeurons_lr = postNeurons * (lr * disinhib[:,np.newaxis]) #adds the effect of dopamine and acetylcholine to the learning rate  
			dW = (np.dot(preNeurons.T, postNeurons_lr) - np.sum(postNeurons_lr, 0)*W)

		#update weights		
		if self.lim_weights and e>=self.n_epi_crit: #artificially prevents weight explosion; used to dissociate influences in parameter self.exploration
			mask = np.logical_and(np.sum(self.hid_W+hid_dW,0)<=940.801, np.min(self.hid_W+hid_dW,0)>0.2)
		else:
			mask = np.ones(np.size(W,1), dtype=bool)

		W[:,mask] += dW[:,mask]
		W = np.clip(W, 1e-10, np.inf)
		
		return W








