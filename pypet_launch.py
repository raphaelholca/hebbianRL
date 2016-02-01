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

ex = reload(ex)
pp = reload(pp)

""" static parameters """
parameter_dict = {	'dHigh' 		: 0.0,
					'dMid' 			: 0.2,
					'dNeut' 		: 0.0,
					'dLow' 			: -4.0,
					'protocol'		: 'gabor',
					'name' 			: 'pypet_test',
					'n_runs' 		: 1,		
					'n_epi_crit'	: 0,				
					'n_epi_dopa'	: 0,				
					't'				: 0.1, 							
					'A' 			: 1.2,
					'lr'			: 0.001,	#0.005
					'batch_size' 	: 20,
					'n_hid_neurons'	: 16,
					'init_file'		: '',	
					'lim_weights'	: False,
					'noise_std'		: 0.2,
					'exploration'	: False,
					'pdf_method' 	: 'fit',
					'classifier'	: 'neural',
					'test_each_epi'	: False,
					'verbose'		: False,
					'seed' 			: 976 #np.random.randint(1000)
					}

""" explored parameters """
explore_dict = {	'dMid'			: [0.1],#[0.00, 0.0125, 0.025, 0.05, 0.300],
					'dLow'			: [0.2]#[-2.0, -4.000, -6.00, -8.0, -10.0]
				}

""" load and pre-process images """
images_dict, labels_dict, images_params = ex.load_images(	protocol 		= parameter_dict['protocol'],
															A 				= parameter_dict['A'],
															verbose 		= parameter_dict['verbose'],
															digit_params 	= {	'classes' 		: np.array([0, 1, 2, 3, 4, 5, 6, 7, 8 , 9 ], dtype=int),
																				'dataset_train'	: 'train',
																				'dataset_path' 	: '/Users/raphaelholca/Documents/data-sets/MNIST',
																				},
															gabor_params 	= {	'n_train' 		: 10000,
																				'n_test' 		: 10000,
																				'target_ori' 	: 87.,
																				'excentricity' 	: 90.,
																				'noise_crit'	: 0.,
																				'noise_train'	: 0.,
																				'noise_test'	: 0.,
																				'im_size'		: 28
																				}
															)

""" create directory to save data """
save_path = os.path.join('output', parameter_dict['name'])
pp.check_dir(save_path, overwrite=True)
print_dict = parameter_dict.copy()
print_dict.update(explore_dict)
print_dict.update(images_params)
save_file = os.path.join(save_path, parameter_dict['name'] + '_params.txt')
ex.print_params(print_dict, save_file)

""" create pypet environment """
env = pypet.Environment(trajectory 		= 'explore_perf',
						log_stdout		= False,
						add_time 		= False,
						multiproc 		= True,
						ncores 			= 1,
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
name_best = pp.plot_results(folder_path=save_path)
pp.launch_assess(save_path, parameter_dict['name']+name_best, images_dict['train'], labels_dict['train'])

print '\nrun name:\t' + parameter_dict['name']
print 'start time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(tic))
print 'end time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(toc))
print 'train time:\t' + time.strftime("%H:%M:%S", time.gmtime(toc-tic))







































