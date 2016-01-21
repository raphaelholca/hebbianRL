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
import helper.external as ex
import helper.pypet_helper as pp

ex = reload(ex)
pp = reload(pp)

""" static parameters """
parameter_dict = {	'dHigh' 		: 4.5,
					'dMid' 			: 0.02,
					'dNeut' 		: -0.1,
					'dLow' 			: -2.0,
					'protocol'		: 'gabor',
					'name' 			: 'pypet_noExplr_gabor_1',
					'n_runs' 		: 10,		
					'n_epi_crit'	: 15,				
					'n_epi_dopa'	: 15,				
					't'				: 0.1, 							
					'A' 			: 1.2,
					'lr'			: 0.005,	#0.01
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

""" explored parameters """
explore_dict = {	'dMid'			: [0.05, 0.10, 0.15, 0.20, 0.25],
					'dLow'			: [-7.0, -6.0, -5.0, -4.0, -3.0]
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

""" create pypet environment """
save_path = os.path.join('output', parameter_dict['name'])
env = pypet.Environment(trajectory 		= 'explore_perf',
						log_stdout		= False,
						add_time 		= False,
						multiproc 		= True,
						ncores 			= 10,
						filename		=  os.path.join(save_path, 'explore_perf.hdf5'),
						overwrite_file	= False)

pp.print_params(parameter_dict, explore_dict, save_path)

traj = env.v_trajectory
pp.add_parameters(traj, parameter_dict)

explore_dict = pypet.cartesian_product(explore_dict, tuple(explore_dict.keys())) #if not all entry of dict need be explored through cartesian product replace tuple(.) only with relevant dict keys in tuple
explore_dict['name'] = pp.set_run_names(explore_dict, parameter_dict['name'])
traj.f_explore(explore_dict)

""" launch simulation with pypet for parameter exploration """
tic = time.time()
env.f_run(pp.launch_exploration, images_dict, labels_dict, images_params, save_path)
toc = time.time()

print "\n\nplotting results"
pp.plot_results(folder_path=save_path)

print '\nstart time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(tic))
print 'end time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(toc))
print 'train time:\t' + time.strftime("%H:%M:%S", time.gmtime(toc-tic))







































