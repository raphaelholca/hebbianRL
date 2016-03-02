"""
Author: Raphael Holca-Lamarre
Date: 23/10/2014

This code creates a hebbian neural network object and trains it on the MNIST dataset. The learning rule is a hebbian learning rule augmented with a learning mechanism inspired from dopamine signalling in animal cortex.
"""

import os
import matplotlib
if 'Documents' in os.getcwd(): matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import hebbian_net
import helper.external as ex
import helper.assess_network as an

hebbian_net = reload(hebbian_net)
ex = reload(ex)
an = reload(an)

net_name = 'gabor_test_all_ori_2'
# ori_to_tests = np.array([-60., -20., -10., -5., 0., +5., +10., +20., +60.])
# ori_to_tests = np.array([0., +1., +2. +5., +10., +20., +30.])
ori_to_tests = np.array([0.])

#load net from file
net = pickle.load(open(os.path.join('output', net_name, 'Network'), 'r'))

#re-initialize output weights
net.out_W = (np.random.random_sample(size=(net.n_hid_neurons, net.n_out_neurons))/1000+1.0)/net.n_hid_neurons

n_runs_network = net.n_runs 
#set training variables:
net.n_runs 			= 1
net.n_epi_crit		= 5
net.n_epi_dopa		= 0
# net.lr_hid 			= 0
net.lr_out 			= 0
net.test_each_epi 	= False
net.verbose 	 	= True

perf_at_ori = np.zeros((n_runs_network, len(ori_to_tests)))

#test each orientation
for i_ori, ori in enumerate(ori_to_tests):

	#load and pre-process training and testing images
	gabor_params = net.images_params
	gabor_params['target_ori'] = net.images_params['target_ori'] - ori	
	images_dict, labels_dict, images_params = ex.load_images(	protocol 		= net.protocol,
																A 				= net.A,
																verbose 		= net.verbose,
																gabor_params 	= gabor_params
																)

	#test each run
	for i_run in range(n_runs_network):
###prevent network from re-initalizing weight!###

		#load weights for each runs
		net.hid_W = net.hid_W_trained[i_run,:,:]

		net.train(images_dict, labels_dict, images_params)

		perf_at_ori[i_run, i_ori] = net.test(images_dict, labels_dict, during_training=True)

print perf_at_ori

""" plot of performance for different orientations """
fig, ax = plt.subplots()
			
ax.scatter(ori_to_tests, np.mean(perf_at_ori,0), lw=2)

# ax.vlines(0, 0, 1, colors=u'k', linewidth=1.5, linestyle=':')

fig.patch.set_facecolor('white')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xlabel('angle from target (deg)', fontsize=18)
ax.set_ylabel('% correct', fontsize=18)
ax.set_xlim([-90,90])
ax.tick_params(axis='both', which='major', direction='out', labelsize=16)
plt.tight_layout()

plt.savefig(os.path.join('output', net_name, 'perf_all_ori.pdf'))




















