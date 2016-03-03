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
ori_to_tests = np.array([-60., -30., -20., -10., -5., -2., -1., 0., +1., +2., +5., +10., +20., +30., +60.])
# ori_to_tests = np.array([-20., 0.])
verbose = True

#load net from file
net = pickle.load(open(os.path.join('output', net_name, 'Network'), 'r'))

#save original variables:
n_runs_ori 			= np.copy(net.n_runs)
target_ori_ori 		= np.copy(net.images_params['target_ori'])
hid_W_trained_ori 	= np.copy(net.hid_W_trained)
#set training variables:
net.n_runs 			= 1
net.n_epi_crit		= 1
net.n_epi_dopa		= 0
net.lr_hid 			= 0
net.lr_out 			*= 100
net.init_file 		= 'NO_INIT'
net.test_each_epi 	= True
net.verbose 	 	= False

perf_at_ori = np.zeros((n_runs_ori, len(ori_to_tests)))

#test each orientation
for i_ori, ori in enumerate(ori_to_tests):
	if net.verbose: print "\n----------------------------------------------------------------------------------------"
	else: print
	
	#load and pre-process training and testing images
	gabor_params = net.images_params
	gabor_params['target_ori'] = target_ori_ori + ori	
	if verbose: print "test orientation: " + str(ori) + " (" + str(gabor_params['target_ori']) + ")"
	
	np.random.seed(0)
	images_dict, labels_dict, images_params = ex.load_images(	protocol 		= net.protocol,
																A 				= net.A,
																verbose 		= net.verbose,
																gabor_params 	= gabor_params
																)

	#test each run
	for i_run in range(n_runs_ori):
		#re-initialize output weights
		np.random.seed(0)
		net.out_W = (np.random.random_sample(size=(net.n_hid_neurons, net.n_out_neurons))/1000+1.0)/net.n_hid_neurons

		#load weights for each runs
		net.hid_W = hid_W_trained_ori[i_run,:,:]

		net.train(images_dict, labels_dict, images_params)

		perf_at_ori[i_run, i_ori] = net.test(images_dict, labels_dict, during_training=True)

		if verbose: print "performance: " + str(perf_at_ori[i_run, i_ori])

print
print perf_at_ori

""" plot of performance for different orientations """
fig, ax = plt.subplots()
			
ax.scatter(np.ones_like(perf_at_ori)*ori_to_tests, perf_at_ori, alpha=0.2)
ax.errorbar(ori_to_tests, np.mean(perf_at_ori,0), yerr=np.std(perf_at_ori,0)/np.sqrt(n_runs_ori), marker='o', ms=5, ls='-', lw=1.5, c='r', mfc='r', mec='r', ecolor='r', mew=1)
# ax.plot(ori_to_tests, np.mean(perf_at_ori,0), )

# ax.scatter(ori_to_tests, np.mean(perf_at_ori,0), facecolor='r')

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




















