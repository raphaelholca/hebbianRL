"""
Author: Raphael Holca-Lamarre
Date: 23/10/2014

This code uses PyPet to explore the parameters of the hebbian neural network object.
"""

import os
import matplotlib
if 'Documents' in os.getcwd(): matplotlib.use('Agg') #to avoid sending plots to screen when working on the servers
import numpy as np
import time
import pypet
import helper.external as ex
import helper.pypet_helper as pp
np.random.seed(0)

ex = reload(ex)
pp = reload(pp)

""" static parameters """
parameter_dict = {	'dHigh' 		: 0.0,
					'dMid' 			: 0.0,
					'dNeut' 		: 0.0,
					'dLow' 			: 0.0,
					'protocol'		: 'gabor',
					'name' 			: 'pypet_gabor_noise_0-05_exc_3_lr_0-1',
					'n_runs' 		: 5,		
					'n_epi_crit'	: 0,				
					'n_epi_dopa'	: 50,				
					't'				: 0.1,						
					'A' 			: 1.2,
					'lr'			: 0.001,		#0.01
					'batch_size' 	: 20,
					'n_hid_neurons'	: 16,
					'init_file'		: 'gabor_pretrained_noNoise',
					'lim_weights'	: False,
					'noise_std'		: 0.2,
					'exploration'	: True,
					'pdf_method' 	: 'fit',
					'classifier'	: 'neural',
					'test_each_epi'	: False,
					'verbose'		: False,
					'seed' 			: 977 #np.random.randint(1000)
					}

""" explored parameters """
explore_dict = {	'dHigh'			: [0.000, 0.400, 0.800, 1.200, 1.600],
					'dNeut'			: [-0.15, -0.10, -0.05, -0.01, 0.000],
					
					'dMid'			: [0.000, 0.005, 0.010, 0.050, 0.100],
					'dLow'			: [0.000, -0.40, -0.80, -1.20, -1.60]
				}

""" load and pre-process images """
images_dict, labels_dict, images_params = ex.load_images(	protocol 		= parameter_dict['protocol'],
															A 				= parameter_dict['A'],
															verbose 		= parameter_dict['verbose'],
															digit_params 	= {	'classes' 		: np.array([0, 1, 2, 3, 4, 5, 6, 7, 8 , 9 ], dtype=int),
																				'dataset_train'	: 'train',
																				'dataset_path' 	: '/Users/raphaelholca/Documents/data-sets/MNIST',
																				'shuffle'		: False
																				},
															gabor_params 	= {	'n_train' 		: 10000,
																				'n_test' 		: 10000,
																				'target_ori' 	: 87.,
																				'excentricity' 	: 3.0,
																				'noise'			: 0.05,
																				'im_size'		: 28
																				}
															)

""" create directory to save data """
save_path = os.path.join('output', parameter_dict['name'])
pp.check_dir(save_path, overwrite=False)
print_dict = parameter_dict.copy()
print_dict.update(explore_dict)
print_dict.update({'images_params':images_params})
save_file = os.path.join(save_path, parameter_dict['name'] + '_params.txt')
ex.print_params(print_dict, save_file)

""" create pypet environment """
env = pypet.Environment(trajectory 		= 'explore_perf',
						log_stdout		= False,
						add_time 		= False,
						multiproc 		= True,
						ncores 			= 10,
						filename		=  os.path.join(save_path, 'explore_perf.hdf5'))

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
pp.faceting(save_path)
name_best = pp.plot_results(folder_path=save_path)
pp.launch_assess(save_path, parameter_dict['name']+name_best, images_dict['train'], labels_dict['train'])

print '\nrun name:\t' + parameter_dict['name']
print 'start time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(tic))
print 'end time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(toc))
print 'train time:\t' + time.strftime("%H:%M:%S", time.gmtime(toc-tic))







































