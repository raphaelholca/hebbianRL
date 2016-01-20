"""
Author: Raphael Holca-Lamarre
Date: 23/10/2014

This code uses PyPet to explore the parameters of the hebbian neural network object.
"""

import os
import matplotlib
if 'Documents' in os.getcwd():
	matplotlib.use('Agg') #to avoid sending plots to screen when working on the servers

import numpy as np
import pypet
import pickle
import time
import hebbian_net
import helper.external as ex

hebbian_net = reload(hebbian_net)
ex = reload(ex)

def launch_exploration(traj, images_dict, labels_dict, images_params):
	parameter_dict = traj.parameters.f_to_dict(short_names=True, fast_access=True)

	try:
		test_perf = launch_one_exploration(parameter_dict, images_dict, labels_dict, images_params)
	except ValueError:
		test_perf = [-1.]

	traj.f_add_result('test_perf', perf=test_perf)

def launch_one_exploration(parameter_dict, images_dict, labels_dict, images_params):
	""" create Hebbian neural network """
	net = hebbian_net.Network(**parameter_dict)

	net.train(images_dict, labels_dict, images_params)

	perf_dict = net.test(images_dict, labels_dict)

	return perf_dict['perf_all']

def add_parameters(traj, parameter_dict):
	for k in parameter_dict.keys():
		traj.f_add_parameter(k, parameter_dict[k])

def set_run_names(explore_dict, name):
	nXplr = len(explore_dict[explore_dict.keys()[0]])
	runName_list = [name for _ in range(nXplr)]
	for n in range(nXplr):
		for k in explore_dict.keys():
			runName_list[n] += '_'
			runName_list[n] += k
			runName_list[n] += str(explore_dict[k][n]).replace('.', ',')
	return runName_list

parameter_dict = {	'dHigh' 		: 4.5,
					'dMid' 			: 0.02,
					'dNeut' 		: -0.1,
					'dLow' 			: -2.0,
					'protocol'		: 'digit',
					'name' 			: 'digit_long_highExplr',
					'n_runs' 		: 1,		
					'n_epi_crit'	: 5,				
					'n_epi_dopa'	: 5,				
					't'				: 0.1, 							
					'A' 			: 1.2,
					'lr'			: 0.01,	#0.005
					'batch_size' 	: 20,
					'n_hid_neurons'	: 49,
					'init_file'		: '',	
					'lim_weights'	: False,
					'noise_std'		: 0.2,
					'exploration'	: True,
					'pdf_method' 	: 'fit',
					'classifier'	: 'neural',
					'test_each_epi'	: True,
					'verbose'		: True,
					'seed' 			: 995 #np.random.randint(1000)
					}

""" parameters for exploration """
explore_dict = {'dMid':[0.02, 0.03], 'dLow':[-2.0, -1.0]}

""" load and pre-process images """
images_dict, labels_dict, images_params = ex.load_images(	protocol 		= parameter_dict['protocol'],
															A 				= parameter_dict['A'],
															verbose 		= parameter_dict['verbose'],
															digit_params 	= {	'classes' 		: np.array([ 4 ], dtype=int),
																				'dataset_train'	: 'test',
																				'dataset_path' 	: '/Users/raphaelholca/Documents/data-sets/MNIST',
																				},
															gabor_params 	= {	'n_train' 		: 10000,
																				'n_test' 		: 1000,
																				'target_ori' 	: 85.,
																				'excentricity' 	: 3.,
																				'noise_crit'	: 0.,
																				'noise_train'	: 0.,
																				'noise_test'	: 0.2,
																				'im_size'		: 28,
																				}
															)

""" launch simulation with pypet for parameter exploration """
env = pypet.Environment(trajectory 		= 'explore_perf',
						log_stdout		= False,
						add_time 		= False,
						multiproc 		= True,
						ncores 			= 1,
						filename		='output/' + 'test_pypet' + '/perf.hdf5',
						overwrite_file	= True)

traj = env.v_trajectory
add_parameters(traj, parameter_dict)

explore_dict = pypet.cartesian_product(explore_dict, tuple(explore_dict.keys())) #if not all entry of dict need be explored through cartesian product replace tuple(.) only with relevant dict keys in tuple
explore_dict['name'] = set_run_names(explore_dict, parameter_dict['name'])
traj.f_explore(explore_dict)

#run the simuation
tic = time.time()
env.f_run(launch_exploration, images_dict, labels_dict, images_params)
toc = time.time()

print '\nstart time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(tic))
print 'end time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(toc))
print 'train time:\t' + time.strftime("%H:%M:%S", time.gmtime(toc-tic))











































