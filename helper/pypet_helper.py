import os
import numpy as np
import matplotlib.pyplot as plt
import helper.assess_network as an
import grating as gr
import hebbian_net
import pypet
import pickle
import shutil
import time
from scipy import stats

an = reload(an)
gr = reload(gr)
hebbian_net = reload(hebbian_net)

def launch_exploration(traj, images_dict, labels_dict, images_params, save_path):
	""" launch all the exploration of the parameters """
	parameter_dict = traj.parameters.f_to_dict(short_names=True, fast_access=True)
	try:
		test_perf = launch_one_exploration(parameter_dict, images_dict, labels_dict, images_params, save_path)
	except ValueError:
		test_perf = [-1.]	

	traj.f_add_result('test_perf', perf=test_perf)

def launch_one_exploration(parameter_dict, images_dict, labels_dict, images_params, save_path):
	""" launch one instance of the network """
	net = hebbian_net.Network(**parameter_dict)

	net.train(images_dict, labels_dict, images_params)

	perf_dict = net.test(images_dict, labels_dict)

	p_file = open(os.path.join(save_path, 'networks', net.name), 'w')
	pickle.dump(net, p_file)
	p_file.close()

	return perf_dict['perf_all']

def add_parameters(traj, parameter_dict):
	for k in parameter_dict.keys():
		traj.f_add_parameter(k, parameter_dict[k])

def set_run_names(explore_dict, name):
	nXplr = len(explore_dict[explore_dict.keys()[0]])
	runName_list = [name for _ in range(nXplr)]
	for n in range(nXplr):
		for k in sorted(explore_dict.keys()):
			runName_list[n] += '_'
			runName_list[n] += k
			runName_list[n] += str(explore_dict[k][n]).replace('.', ',')
	return runName_list

def check_dir(save_path, overwrite=False):
	if os.path.isdir(save_path) and overwrite==True:
		shutil.rmtree(save_path)
	elif os.path.isdir(save_path) and overwrite==False:
		raise RuntimeError("trajectory already exists and will not be overwritten")
	if not os.path.isdir(save_path):
		os.mkdir(save_path)
	if not os.path.isdir(os.path.join(save_path, 'networks')):
		os.mkdir(os.path.join(save_path, 'networks'))

def plot_results(folder_path=''):
	if folder_path=='':
		folder_path = '/Users/raphaelholca/Dropbox/hebbian_net/output/test_pypet_0/'
		# folder_path = '/Users/raphaelholca/Mountpoint/hebbianRL/output/proba_two_lin/'

	traj_name = 'explore_perf'
	traj = pypet.load_trajectory(traj_name, filename=os.path.join(folder_path, 'explore_perf.hdf5'), force=True)
	traj.v_auto_load = True

	perf_all = []
	for run in traj.f_iter_runs():
		perf_all.append(traj.results[run].test_perf['perf'])
	perf_all = np.array(perf_all)
	perf = np.mean(perf_all,1)

	param_traj = traj.f_get_explored_parameters()
	param = {}
	for k in param_traj:
		if k[11:] != 'name':
			xplr_values = np.array(param_traj[k].f_get_range())
			param[k[11:]] = xplr_values

	arg_best = np.argmax(perf)

	best_param = {}

	print 'best parameters:'
	print '================'
	for k in param.keys():
		best_param[k] = param[k][arg_best]
		print k + ' : ' + str(param[k][arg_best]) + '\t\t' + str(np.round(np.unique(param[k]),3))
	print "\nbest performance: " + str(np.round(np.max(perf)*100,2)) + "\n"

	keys = param.keys()
	for ik in range(len(keys)):
		if len(keys)==1: ik=-1
		for k in keys[ik+1:]:
			others = keys[:]
			if len(keys)>1: 
				others.remove(keys[ik])
				others.remove(k)
			
			mask = np.ones_like(param[k], dtype=bool)
			if len(param)>2:
				for o in others:
					mask = np.logical_and(mask, param[o]==best_param[o])
			pX = param[keys[ik]][mask]
			pY = param[k][mask]
			rC = np.hstack(perf)[mask]

			if True: #True: non-linear representation of results; False: linear representation 
				ipX = np.zeros(len(pX))
				ipY = np.zeros(len(pY))
				for i in range(len(pX)):
					ipX[i] = np.argwhere(pX[i]==np.sort(np.unique(pX)))
					ipY[i] = np.argwhere(pY[i]==np.sort(np.unique(pY)))
			else:
				ipX = np.copy(pX)
				ipY = np.copy(pY)

			fig = plt.figure()
			fig.patch.set_facecolor('white')
			
			plt.scatter(ipX, ipY, c=rC, cmap='CMRmap', vmin=np.min(perf)-0.1, vmax=np.max(perf), s=1000, marker='s')
			# plt.scatter(param[keys[ik]][arg_best], param[k][arg_best], c='r', s=50, marker='x')
			for i in range(len(pX)):
				if pX[i]==param[keys[ik]][arg_best] and pY[i]==param[k][arg_best]:
					plt.text(ipX[i], ipY[i], str(np.round(rC[i]*100,1)), horizontalalignment='center', verticalalignment='center', weight='bold', bbox=dict(facecolor='red', alpha=0.5))
				else:
					plt.text(ipX[i], ipY[i], str(np.round(rC[i]*100,1)), horizontalalignment='center', verticalalignment='center')
			plt.xticks(ipX, pX)
			plt.yticks(ipY, pY)
			plt.xlabel(keys[ik], fontsize=25)
			plt.ylabel(k, fontsize=25)
			plt.tick_params(axis='both', which='major', labelsize=18)
			plt.tight_layout()
			plt.savefig(os.path.join(folder_path, keys[ik] + '_' + k + '.pdf'))

			plt.close(fig)

	name_best = ''
	for k in sorted(param.keys()):
		name_best += '_'
		name_best += k
		name_best += str(best_param[k]).replace('.', ',')

	return name_best

def plot_all_slope_diffs(save_path):
	""" plot slope difference for all networks """
	net_path = os.path.join(save_path, 'networks')
	plot_path = os.path.join(save_path, 'slope_diffs')
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)
	for n in os.listdir(net_path):
		n_file = open(os.path.join(net_path, n), 'r')
		net = pickle.load(n_file)

		name = net.name
		hid_W_naive = net.hid_W_naive
		hid_W_trained = net.hid_W_trained
		t = net.t
		target_ori = net.images_params['target_ori']

		#compute RFs info for the naive network
		curves_naive = gr.tuning_curves(hid_W_naive, t, target_ori, name, method='no_softmax', plot=False, save_path=plot_path)
		pref_ori_naive = gr.preferred_orientations(hid_W_naive, t, target_ori, name, curves_naive)
		slopes_naive = gr.slopes(hid_W_naive, curves_naive, pref_ori_naive, t, target_ori, name, plot=False, save_path=plot_path)

		#compute RFs info for the trained network
		curves = gr.tuning_curves(hid_W_trained, t, target_ori, name, method='no_softmax', plot=False, save_path=plot_path)
		pref_ori = gr.preferred_orientations(hid_W_trained, t, target_ori, name, curves)
		slopes = gr.slopes(hid_W_trained, curves, pref_ori, t, target_ori, name, plot=False, save_path=plot_path)
		
		gr.slope_difference(slopes_naive['all_dist_from_target'], slopes_naive['all_slope_at_target'], slopes['all_dist_from_target'], slopes['all_slope_at_target'], name, plot=True, save_path=plot_path)

def launch_assess(save_path, file_name, images, labels):
	net_file = open(os.path.join(save_path, 'networks', file_name), 'r')
	best_net = pickle.load(net_file)
	an.assess(best_net, images, labels, save_path=os.path.join(save_path, 'best_net'))

def import_traj(folder_path, file_name, order_face, traj_name='explore_perf'):
	print "importing data..."
	traj = pypet.load_trajectory(traj_name, filename=os.path.join(folder_path, file_name+'.hdf5'), force=True)
	traj.v_auto_load = True

	perc_correct = np.array([])
	perc_correct_all = []
	skipped_count = 0
	ok_runs = []
	for run in traj.f_iter_runs():
		# import pdb; pdb.set_trace()
		perc_correct = np.append(perc_correct, np.mean(traj.results[run].test_perf['perf']))
		perc_correct_all.append(traj.results[run].test_perf['perf'])
		ok_runs.append(int(run[4:]))

	param_traj = traj.f_get_explored_parameters()
	param = {}
	for k in param_traj:
		if k[11:] != 'name':
			xplr_values = np.array(param_traj[k].f_get_range())[ok_runs]
			if len(np.unique(xplr_values)) >1:
				param[k[11:]] = xplr_values

	# #removes explored trajectories other than the dopamine ones
	# for k in param.keys():
	# 	if k not in order_face:
	# 		print str(k) + ": " + str(np.unique(param[k]))
	# 		select = np.unique(param[k])[0]	##<-------------change index to change which value of removed parameter to use-------------
	# 		for n_k in param.keys():
	# 			if n_k!=k:
	# 				param[n_k] = param[n_k][param[k]==select]
	# 		perc_correct = perc_correct[param[k]==select]
	# 		del param[k]

	return perc_correct, np.array(perc_correct_all), param

def faceting(folder_path):
	file_name = 'explore_perf'

	# trans = {'dMid': '+pred +rew', 'dHigh': '-pred +rew', 'dNeut': '-pred -rew', 'dLow': '+pred -rew'}
	trans = {'dMid': 'dMid', 'dHigh': 'dHigh', 'dNeut': 'dNeut', 'dLow': 'dLow'}

	threshold = 0.002
	t_threshold = 0.1
	vmin=0.92#1
	vmax=0.96#0.97
	font_large = 20
	font_small = 18

	order_face = ['dNeut', 'dHigh', 'dMid', 'dLow'] #order is x1, y1, x2, y2

	perc_correct, perc_correct_all, param = import_traj(folder_path, file_name, order_face)

	n_mesure = len(np.unique(param['dHigh']))

	#find best parameters combination
	arg_best = np.argmax(perc_correct)
	best_param = {}
	print 'best parameters:'
	print '================'
	for k in param.keys():
		best_param[k] = param[k][arg_best]
		print k + ' : ' + str(param[k][arg_best]) + '\t\t' + str(np.round(np.unique(param[k]),2))
	best_perf_str = str(int(np.max(perc_correct)*1000)/10.) + '%'
	print '\nbest performance: ' + best_perf_str

	invert=True if np.argwhere(param[order_face[0]]!= param[order_face[0]][0])[0] > np.argwhere(param[order_face[1]]!= param[order_face[1]][0])[0] else False

	#find similarly good parameters...
	if np.size(perc_correct_all,1)>1 :#...using statistical significance testing
		arg_best_all = np.array([], dtype=int)
		best_perf = perc_correct_all[arg_best, :]
		for arg in range(np.size(perc_correct_all,0)):
			t, prob = stats.ttest_ind(best_perf, perc_correct_all[arg, :]) #two-sided t-test with independent samples
			if prob > t_threshold: #not statistically significantly different
				arg_best_all = np.append(arg_best_all, arg)
		best_param_all = {}
		for k in param.keys():
			best_param_all[k] = param[k][arg_best_all]
		best_perf_all = perc_correct[arg_best_all]

	else: #...within [threshold]% of best performance
		arg_best_all = np.argwhere(perc_correct >= np.max(perc_correct)-threshold*np.max(perc_correct))
		best_param_all = {}
		for k in param.keys():
			best_param_all[k] = param[k][arg_best_all]
		best_perf_all = perc_correct[arg_best_all]

	""" faceting plot """
	fig, ax = plt.subplots(n_mesure,n_mesure, figsize=(8,7))#, sharex=True, sharey=True)
	fig.patch.set_facecolor('white')

	for x2_i, x2 in enumerate(np.sort(np.unique(param[order_face[2]]))):
		for y2_i, y2 in enumerate(np.sort(np.unique(param[order_face[3]]))[::-1]):
			x1 = np.unique(param[order_face[0]][np.logical_and(x2==param[order_face[2]] , y2==param[order_face[3]])])
			y1 = np.unique(param[order_face[1]][np.logical_and(x2==param[order_face[2]] , y2==param[order_face[3]])])
			z = perc_correct[np.logical_and(x2==param[order_face[2]] , y2==param[order_face[3]])]
			z_square = np.reshape(z, (n_mesure, n_mesure))
			if invert: z_square=z_square.T

			ax[y2_i, x2_i].imshow(z_square, origin='lower', interpolation='nearest', cmap='CMRmap', vmin=vmin, vmax=vmax)

			# indicate all params that give performance within [threshold]% of best performance
			if x2 in best_param_all[order_face[2]][y2==best_param_all[order_face[3]]]:
				mask_best = np.logical_and(x2==best_param_all[order_face[2]], y2==best_param_all[order_face[3]])
				x1_dots = []
				y1_dots = []
				for x in best_param_all[order_face[0]][mask_best]: x1_dots.append(np.argwhere(x==x1))
				for y in best_param_all[order_face[1]][mask_best]: y1_dots.append(np.argwhere(y==y1))
				ax[y2_i, x2_i].scatter(x1_dots, y1_dots, s=10, c='k')

			# indicate params that give best performance
			if x2==best_param[order_face[2]] and y2==best_param[order_face[3]]:
				x1_dot = np.argwhere(best_param[order_face[0]] == x1)
				y1_dot = np.argwhere(best_param[order_face[1]] == y1)
				ax[y2_i, x2_i].scatter(x1_dot, y1_dot, s=100, c='r', marker='*')

			if y2_i==0: #ticks for top row
				ax[y2_i,x2_i].set_yticks([])
				ax[y2_i,x2_i].set_xticks([n_mesure/2])
				ax[y2_i,x2_i].xaxis.set_ticks_position('top')
				ax[y2_i,x2_i].set_xticklabels([str(x2)], fontsize=font_small)
				ax[y2_i,x2_i].tick_params(axis='both', which='major', direction='out')
			if x2_i==n_mesure-1: #ticks for right row
				if y2_i!=0:
					ax[y2_i,x2_i].set_xticks([])
				ax[y2_i,x2_i].set_yticks([n_mesure/2])
				ax[y2_i,x2_i].yaxis.set_ticks_position('right')
				ax[y2_i,x2_i].set_yticklabels([str(y2)], fontsize=font_small)
				ax[y2_i,x2_i].tick_params(axis='both', which='major', direction='out')
			elif y2_i!=0:
				ax[y2_i,x2_i].set_xticks([])
				ax[y2_i,x2_i].set_yticks([])

			ax[y2_i,x2_i].set_xlim(-.5, n_mesure-.5)
			ax[y2_i,x2_i].set_ylim(-.5, n_mesure-.5)

	ax[n_mesure-1,0].set_xticks(range(n_mesure)[::2])
	ax[n_mesure-1,0].xaxis.set_ticks_position('bottom')
	# ax[n_mesure-1,0].set_xticklabels(x1[::2], fontsize=font_small)
	# ax[n_mesure-1,0].set_xticklabels([i.replace('0.', '.') for i in map(str,x1)], fontsize=font_small)
	ax[n_mesure-1,0].set_xticklabels([i.replace('0.', '.') for i in map(str,x1)[::2]], fontsize=font_small)

	ax[n_mesure-1,0].set_yticks(range(n_mesure)[::2])
	ax[n_mesure-1,0].yaxis.set_ticks_position('left')
	ax[n_mesure-1,0].set_yticklabels(y1[::2], fontsize=font_small)
	ax[n_mesure-1,0].tick_params(axis='both', which='major', direction='out')

	ax[n_mesure-1,0].set_xlabel(trans[order_face[0]], fontsize=font_large)
	ax[n_mesure-1,0].set_ylabel(trans[order_face[1]], fontsize=font_large)
		
	fig.text(0.5, 0.96, trans[order_face[2]], ha='center', fontsize=font_large)
	fig.text(0.965, 0.5, trans[order_face[3]], va='center', rotation=90, fontsize=font_large)
	fig.text(0.025, 0.975, 'best perf: '+best_perf_str, va='center', fontsize=font_small, weight='semibold')

	fig.savefig(os.path.join(folder_path, 'faceting.pdf'), format='pdf')














