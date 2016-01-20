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
import time
import shutil
import hebbian_net
import helper.external as ex
import helper.pypet_helper as pp

hebbian_net = reload(hebbian_net)
ex = reload(ex)
pp = reload(pp)

parameter_dict = {	'dHigh' 		: 4.5,
					'dMid' 			: 0.02,
					'dNeut' 		: -0.1,
					'dLow' 			: -2.0,
					'protocol'		: 'digit',
					'name' 			: 'test_pypet_0',
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
					'exploration'	: False,
					'pdf_method' 	: 'fit',
					'classifier'	: 'neural',
					'test_each_epi'	: False,
					'verbose'		: False,
					'seed' 			: 995 #np.random.randint(1000)
					}

""" parameters for exploration """
explore_dict = {	'dMid'			: [0.02], 
					'dLow'			: [-2.0]
				}

""" load and pre-process images """
images_dict, labels_dict, images_params = ex.load_images(	protocol 		= parameter_dict['protocol'],
															A 				= parameter_dict['A'],
															verbose 		= parameter_dict['verbose'],
															digit_params 	= {	'classes' 		: np.array([ 4, 7, 9 ], dtype=int),
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
save_path = 'output/' + parameter_dict['name']
if os.path.exists(os.path.join(save_path)):
	shutil.rmtree(save_path)
	os.mkdir(save_path)
	os.mkdir(os.path.join(save_path, 'networks'))
pp.print_params(parameter_dict, explore_dict, save_path)
env = pypet.Environment(trajectory 		= 'explore_perf',
						log_stdout		= False,
						add_time 		= False,
						multiproc 		= True,
						ncores 			= 2,
						filename		=  os.path.join(save_path, 'explore_perf.hdf5'),
						overwrite_file	= True)

traj = env.v_trajectory
pp.add_parameters(traj, parameter_dict)

explore_dict = pypet.cartesian_product(explore_dict, tuple(explore_dict.keys())) #if not all entry of dict need be explored through cartesian product replace tuple(.) only with relevant dict keys in tuple
explore_dict['name'] = pp.set_run_names(explore_dict, parameter_dict['name'])
traj.f_explore(explore_dict)

#run the exploration
tic = time.time()
env.f_run(pp.launch_exploration, images_dict, labels_dict, images_params, save_path)
toc = time.time()

print '\nstart time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(tic))
print 'end time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(toc))
print 'train time:\t' + time.strftime("%H:%M:%S", time.gmtime(toc-tic))







































