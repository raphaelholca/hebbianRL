""" Support functions for the gabor experimental protocol.  """

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import external as ex
from scipy import stats

ex = reload(ex)

def gabor(size=28, lambda_freq=5, theta=0, sigma=5, phase=0, noise=0):
	"""
	Creates a Gabor patch

	Args:

		size (int): Image side
		lambda_freq (int or float): Spatial frequency (pixels per cycle) 
		theta (int, float, list or numpy array): Grating orientation in degrees (if list or array, a patch is created for each value)
		sigma (int or float): gaussian standard deviation (in pixels)
		phase (float, list or numpy array): phase of the filter; range: [0, 1]
		noise (int): noise level to add to Gabor patch; represents the standard deviation of the Gaussian distribution from which noise is drawn; range: (0, inf

	Returns:
		(1D or 2D numpy array): 1D or 2D Gabor patch (n images * n pixels)
	"""
	#normalize input parameters
	noise = np.clip(noise, 1e-10, np.inf)
	if type(theta) == int or type(theta) == float: theta = np.array([theta])
	elif type(theta) == list: theta = np.array(theta)
	if type(phase)==float or type(phase)==int: phase = np.array([phase])
	n_gratings = len(theta)

	# make linear ramp
	X0 = (np.linspace(1, size, size) / size) - .5

	# Set wavelength and phase
	freq = size / float(lambda_freq)
	phaseRad = phase * 2 * np.pi

	# Make 2D grating
	Xm, Ym = np.meshgrid(X0, X0)
	Xm = np.tile(Xm, (n_gratings, 1, 1))
	Ym = np.tile(Ym, (n_gratings, 1, 1))

	# Change orientation by adding Xm and Ym together in different proportions
	thetaRad = (theta / 360.) * 2 * np.pi
	Xt = Xm * np.cos(thetaRad)[:,np.newaxis,np.newaxis]
	Yt = Ym * np.sin(thetaRad)[:,np.newaxis,np.newaxis]

	# 2D Gaussian distribution
	gauss = np.exp(-((Xm ** 2) + (Ym ** 2)) / (2 * (sigma / float(size)) ** 2))

	gratings = np.sin(((Xt + Yt) * freq * 2 * np.pi) + phaseRad[:,np.newaxis,np.newaxis])
	gratings *= gauss #add Gaussian
	gratings += np.random.normal(0.0, noise, size=np.shape(gratings)) #add Gaussian noise
	gratings -= np.min(gratings)

	gratings = np.reshape(gratings, (n_gratings, size**2))

	return gratings

def tuning_curves(W, t, images_params, name, curve_method='basic', plot=True, save_path=''):
	"""
	compute the tuning curve of the neurons

	Args:
		W (dict): dictionary of weight matrices (each element of the dictionary is a weight matrix from an individual run)
		t (float): temperature of the softmax function used during training
		images_params (dict): dictionary of image parameters
		name (str): name of the network, used for saving figures
		curve_method (str, optional): way of computing the tuning curves. Can be: 'basic' (w/o noise, w/ softmax), 'no_softmax' (w/o noise, w/o softmax), 'with_noise' (w/ noise, w/ softmax)
		plot (bool, optional): whether or not to create plots
		save_path (str, optional): path to save plots

	returns:
		(dict): the tuning curves for each neuron of each run
	"""

	# t=0.2 ##<----------uses different t as the one used during training-------------------

	if plot:
		if save_path=='': save_path=os.path.join('output', name)
		if not os.path.exists(os.path.join(save_path, 'TCs')):
			os.makedirs(os.path.join(save_path, 'TCs'))

	if curve_method not in ['basic', 'no_softmax', 'with_noise']:
		print '!!! invalid method - using \'basic\' method !!!'
		curve_method='basic'

	noise = 0.0 #images_params['noise']
	noise_trial = 10#100
	ori_step = 0.1
	n_input = int(180/ori_step)
	n_runs = np.size(W,0)
	im_size = int(np.sqrt(np.size(W,1)))
	n_neurons = np.size(W,2)

	orientations = np.arange(-90.+images_params['target_ori'], 90.+images_params['target_ori'], ori_step)
	SM = False if curve_method=='no_softmax' else True
	if curve_method != 'with_noise':
		test_input = [gabor(size=im_size, lambda_freq=im_size/5., theta=orientations, sigma=im_size/5., phase=0.25, noise=0.0)]
	else:
		test_input = []
		for _ in range(noise_trial):
			test_input.append(gabor(size=im_size, lambda_freq=im_size/5., theta=orientations, sigma=im_size/5., phase=0.25, noise=noise))

	curves = np.zeros((n_runs, n_input, n_neurons))
	pref_ori = np.zeros((n_runs, n_neurons))
	for r in range(n_runs):
		if plot: 
			fig, ax = plt.subplots()
			plt.gca().set_color_cycle(cm.Paired(i) for i in np.linspace(0,0.8,10))
		for i in range(len(test_input)):
			curves[r,:,:] += ex.propagate_layerwise(test_input[i], W[r], SM=SM, t=t)/len(test_input)
			pref_ori[r, :] = orientations[np.argmax(curves[r,:,:],0)]
			pref_ori[r, :] = ex.relative_orientations(pref_ori[r, :], images_params['target_ori'])

		if plot:
			pref_ori_sorter = pref_ori[r, :].argsort()
			
			ax.plot(np.arange(-90., 90., ori_step), curves[r,:,:][:,pref_ori_sorter], lw=2)
			ax.vlines(0, 0, np.max(curves[r,:,:])*1.2, colors=u'k', linewidth=1.5, linestyle=':')

			fig.patch.set_facecolor('white')
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.xaxis.set_ticks_position('bottom')
			ax.yaxis.set_ticks_position('left')
			ax.set_xlabel('angle from target (deg)', fontsize=18)
			ax.set_ylabel('response', fontsize=18)
			ax.set_xlim([-90,90])
			if curve_method=='no_softmax' and False: ax.set_ylim([119,138])
			else: ax.set_ylim([np.min(curves[r,:,:])-(np.max(curves[r,:,:])-np.min(curves[r,:,:]))*.1, np.max(curves[r,:,:])+(np.max(curves[r,:,:])-np.min(curves[r,:,:]))*.1])
			ax.tick_params(axis='both', which='major', direction='out', labelsize=16)
			plt.tight_layout()
		
			plt.savefig(os.path.join(save_path, 'TCs', 'TCs_' + name + '_' + str(r).zfill(3) + '.pdf'))
			plt.close(fig)

	return curves, pref_ori

def slopes(W, curves, pref_ori, t, target_ori, name, plot=False, save_path=''):
	"""
	compute slope of tuning curves at target orientation

	Args:
		W (dict): dictionary of weight matrices (each element of the dictionary is a weight matrix from an individual run)
		curves (dict): the tuning curves for each neuron of each run; *!!* for now does not support curves if computed with 'with_noise' method)
		pref_ori (dict): the preferred orientation of all neurons in all runs
		t (float): temperature of the softmax function (t<<1: strong competition; t>=1: weak competition)
		target_ori (float): target orientation on side of which to discrimate the gabor patches
		name (str): name of the network, used for saving figures
		plot (bool, optional): whether or not to create plots
		save_path (str, optional): path to save plots

	returns:
		slopes (dict): slopes of the tuning curves for each individual neuron *!!* should be plotted so that slope value is alligned between two measurement points
		all_slopes (dict): all slopes collapsed in a single list, with their order matching degrees away from preferred orientation stores in all_deg
		all_deg (dict): degrees away from preferred orientation matching the the slopes stored in all_slopes 
	"""

	if plot:
		if save_path=='': save_path=os.path.join('output', name)
		if not os.path.exists(os.path.join(save_path, 'TCs')):
			os.makedirs(os.path.join(save_path, 'TCs'))

	n_runs = np.size(curves,0)
	n_input = np.size(curves,1)
	n_neurons = np.size(curves,2)

	slopes = np.zeros((n_runs, n_input, n_neurons))
	all_slopes = []
	all_deg = []
	all_dist_from_target = np.empty((n_runs, n_neurons))
	all_slope_at_target = np.empty((n_runs, n_neurons))

	for r in range(n_runs):
		slopes[r,:,:] = np.abs(curves[r] - np.roll(curves[r], 1, axis=0))
		all_slopes.append([])
		all_deg.append([])

		dist_target = pref_ori[r]
		slope_at_target = slopes[r,:,:][n_input/2, np.arange(n_neurons)]

		all_dist_from_target[r, :] = dist_target
		all_slope_at_target[r, :] = slope_at_target

		if plot: 
			fig, ax = plt.subplots()
		
		for o in np.arange(0,180,10):
			deg_idx = int(o * (n_input/180))
		
			deg = pref_ori[r]-o
			deg[deg>90]-=180
			deg[deg<-90]+=180
			y = slopes[r,:,:][deg_idx, :]
			
			all_slopes[r].append(y)
			all_deg[r].append(deg)

			if plot: 
				plt.scatter(deg, y)

		if plot:
			""" plot of slope w.r.t. distance from preferred orientation """
			fig.patch.set_facecolor('white')
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.xaxis.set_ticks_position('bottom')
			ax.yaxis.set_ticks_position('left')
			ax.set_xlabel('degrees away from preferred', fontsize=18)
			ax.set_ylabel('slope', fontsize=18)
			# ax.set_ylim([0, np.max(curves[r])*1.1])
			ax.tick_params(axis='both', which='major', direction='out')
			plt.tight_layout()
		
			plt.savefig(os.path.join(save_path, 'TCs', 'slopes_' + name + '_' + str(r).zfill(3) + '.pdf'))
			plt.close(fig)

			""" plot of slope at target orientation """
			fig, ax = plt.subplots()
			ax.scatter(all_dist_from_target, all_slope_at_target)

			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			ax.xaxis.set_ticks_position('bottom')
			ax.yaxis.set_ticks_position('left')
			ax.set_xlabel('preferred orientation-trained orientation (degrees)', fontsize=18)
			ax.set_ylabel('slope at target orientation', fontsize=18)
			ax.tick_params(axis='both', which='major', direction='out')
			plt.tight_layout()

			plt.savefig(os.path.join(save_path, 'TCs', 'slopes_at_target.pdf'))
			plt.close(fig)

	return {'slopes':slopes, 'all_slopes':np.array(all_slopes), 'all_deg':np.array(all_deg), 'all_dist_from_target':all_dist_from_target, 'all_slope_at_target':all_slope_at_target}


def slope_difference(pre_dist, pre_slopes, post_dist, post_slopes, name, plot=True, slope_binned=False, save_path='', bin_width=8):
	""" compute and plot the slope at training orientation as a function of the difference between preferred and trained orienation both before and after training """

	if save_path=='': save_path=os.path.join('output', name)

	# bin_width = 8 #degrees, to follow plotting in fig 2b of Schoups01, make bin_width=8
	bin_num = int(np.ceil(180./bin_width))
	stat_threshold = 0.05

	if slope_binned:
		bin_edges = np.arange(-90 - bin_width/2 + (180%bin_width)/2, 90+bin_width, bin_width)
		bin_centers = np.arange(-90 + (180%bin_width)/2, bin_edges[-1], bin_width)
		pre_slopes_binned = []
		pre_slopes_mean = np.zeros(len(bin_edges)-1)
		pre_slopes_ste = np.zeros(len(bin_edges)-1)
		post_slopes_binned = []
		post_slopes_mean = np.zeros(len(bin_edges)-1)
		post_slopes_ste = np.zeros(len(bin_edges)-1)

		for ib in range(len(bin_edges)-1):
			values = pre_slopes[np.logical_and(pre_dist>=bin_edges[ib], pre_dist<bin_edges[ib+1])]
			pre_slopes_binned.append(values)
			pre_slopes_mean[ib] = np.mean(values) ##returns NaN for empty bin_edges
			pre_slopes_ste[ib] = np.std(values)/np.sqrt(len(values)) ##returns NaN for empty bin_edges

			values = post_slopes[np.logical_and(post_dist>=bin_edges[ib], post_dist<bin_edges[ib+1])]
			post_slopes_binned.append(values)
			post_slopes_mean[ib] = np.mean(values) ##returns NaN for empty bin_edges
			post_slopes_ste[ib] = np.std(values)/np.sqrt(len(values)) ##returns NaN for empty bin_edges

		""" check whether slopes of trained neurons are significantly greater than slopes of untrained neurons """
		mid_bin = ((bin_num+1)/2)-1
		stat_diff = np.zeros(bin_num)
		t = np.zeros(bin_num)

		for b in range(bin_num):
			if post_slopes_binned[b].any() and pre_slopes_binned[b].any():
				t[b], stat_diff[b] = stats.ttest_ind(post_slopes_binned[b], pre_slopes_binned[b], equal_var=False) #two-sided t-test with independent samples
			else:
				t[b], stat_diff[b] = np.nan, np.nan
		stat_diff*=np.sign(t) #stat_diff is negative if post slope is smaller than pre slope
		stat_signif = np.logical_and(stat_diff>0, stat_diff/2.<stat_threshold) #makes the two-sided t-test a one-sided one
	else:
		stat_diff=np.ones(bin_num)

	""" plot of slope at target orientation """
	if plot:
		fig, ax = plt.subplots(figsize=(8,4.5))
		fig.patch.set_facecolor('white')
		if slope_binned:
			ax.plot(bin_centers, pre_slopes_mean, ls='--', lw=3, c='b', label='statistical')
			ax.errorbar(bin_centers, pre_slopes_mean, yerr=pre_slopes_ste, marker='o', ms=10, ls=' ', lw=3, c='b', mfc='w', mec='b', ecolor='b', mew=2)
			ax.errorbar(bin_centers, post_slopes_mean, yerr=post_slopes_ste, marker='o', ms=10, ls='-', lw=3, c='r', mfc='r', mec='r', ecolor='r', mew=2, label='reward-based')
			
			#marker of statistical significance
			# Y = np.ones(np.sum(stat_signif))*np.nanmax(post_slopes_mean)*1.20
			# ax.scatter(bin_centers[stat_signif], Y, marker='*', c='k')
		else:
			ax.scatter(pre_dist, pre_slopes, c='b')
			ax.scatter(post_dist, post_slopes, c='r')

		# ax.spines['right'].set_visible(False)
		# ax.spines['top'].set_visible(False)
		ax.xaxis.set_ticks_position('bottom')
		ax.yaxis.set_ticks_position('left')
		if slope_binned: ax.set_xticks(bin_centers)
		ax.set_xlim([-50,50])
		if slope_binned:
			ax.set_ylim([np.nanmax(post_slopes_mean)*-0.20, np.nanmax(post_slopes_mean)*1.20])
		else:
			ax.set_ylim([np.nanmax(post_slopes)*-0.20, np.nanmax(post_slopes)*1.20])
		ax.set_xlabel('preferred orientation-trained orientation (degrees)', fontsize=18)
		ax.set_ylabel('slope at TO', fontsize=18)
		ax.tick_params(axis='both', which='major', direction='out', labelsize=16)
		# plt.legend(loc='lower center')
		plt.tight_layout()

		plt.savefig(os.path.join(save_path, name + '_slope_diffs.pdf'))
		plt.close(fig)

	return stat_diff




































