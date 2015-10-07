import numpy as np
import matplotlib.pyplot as plt
import external as ex
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

def tuning_curves(W, params, method='basic', plot=True):
	"""
	compute the tuning curve of the neurons

	Args:
		W (dict): dictionary of weight matrices (each element of the dictionary is a weight matrix from an individual run)
		params (dict): parameters of the main simulation (kwargs)
		method (str, optional): way of computing the tuning curves. Can be: 'basic' (w/o noise, w/ softmax), 'no_softmax' (w/o noise, w/o softmax), 'with_noise' (w/ noise, w/ softmax)
		plot (bool, optional): whether or not to create plots

	returns:
		(dict): the tuning curves for each neuron of each run
	"""

	if method not in ['basic', 'no_softmax', 'with_noise']:
		print '!!! invalid method - using \'basic\' method !!!'
		method='basic'

	t = params['t_hid']
	target_ori = params['target_ori']
	runName = params['runName']

	noise = 1.0
	noise_trial = 100
	ori_step = 0.1
	n_input = int(180/ori_step)
	im_size = int(np.sqrt(np.size(W[W.keys()[0]],0)))
	n_neurons = int(np.size(W[W.keys()[0]],1))

	orientations = np.arange(0,180,ori_step)
	SM = False if method=='no_softmax' else True
	if method != 'with_noise':
		test_input = [gabor(size=im_size, lambda_freq=im_size/5., theta=orientations, sigma=im_size/5., phase=0.25, noise=0.0)]
	else:
		test_input = []
		for _ in range(noise_trial):
			test_input.append(gabor(size=im_size, lambda_freq=im_size/5., theta=orientations, sigma=im_size/5., phase=0.25, noise=noise))

	curves = {}
	for r in W.keys():
		if plot: 
			fig, ax = plt.subplots()
		curves[r] = np.zeros((n_input, n_neurons))
		for i in range(len(test_input)):
			curves[r] += ex.propL1(test_input[i], W[r], SM=SM, t=t)/len(test_input)
			if plot:
				ax.plot(orientations, curves[r])
				ax.vlines(target_ori, 0, np.max(curves[r])*1.05, colors=u'r', linestyles=u'dashed')

				fig.patch.set_facecolor('white')
				ax.spines['right'].set_visible(False)
				ax.spines['top'].set_visible(False)
				ax.xaxis.set_ticks_position('bottom')
				ax.yaxis.set_ticks_position('left')
				ax.set_xlabel('angle (deg)', fontsize=18)
				ax.set_ylabel('response', fontsize=18)
				ax.set_ylim([np.min(curves[r]), np.max(curves[r])+(np.max(curves[r])-np.min(curves[r]))*.1])
				# import pdb; pdb.set_trace()
				ax.tick_params(axis='both', which='major', direction='out')
				plt.tight_layout()
		
		if plot:
			plt.savefig('output/' + runName + '/TCs/' + 'TCs_'  +runName+ '_' + str(r).zfill(3))
			plt.close(fig)

	return curves

def preferred_orientations(W, params):
	"""
	compute the preferred orientation of neurons

	Args:
		W (dict): dictionary of weight matrices (each element of the dictionary is a weight matrix from an individual run)
		params (dict): parameters of the main simulation (kwargs)

	returns:
		the preferred orientation of all neurons in all runs
	"""

	curves = tuning_curves(W, params, method='no_softmax', plot=False)


	n_input = np.size(curves[curves.keys()[0]],0)

	pref_ori = {}
	for r in curves.keys():
		pref_ori[r] = np.argmax(curves[r],0) * (180./n_input) ##180??

	return pref_ori


def slopes(W, curves, pref_ori, params, plot=True):
	"""
	compute slope of tuning curves at target orientation

	Args:
		W (dict): dictionary of weight matrices (each element of the dictionary is a weight matrix from an individual run)
		curves (dict): the tuning curves for each neuron of each run; *!!* for now does not support curves if computed with 'with_noise' method)
		pref_ori (dict): the preferred orientation of all neurons in all runs
		params (dict): parameters of the main simulation (kwargs)
		plot (bool, optional): whether or not to create plots

	returns:
		slopes (dict): slopes of the tuning curves for each individual neuron *!!* should be plotted so that slope value is alligned between two measurement points
		all_slopes (dict): all slopes collapsed in a single list, with their order matching degrees away from preferred orientation stores in all_deg
		all_deg (dict): degrees away from preferred orientation matching the the slopes stored in all_slopes 
	"""

	t = params['t_hid']
	target_ori = params['target_ori']
	runName = params['runName']
	nHidNeurons = params['nHidNeurons']

	n_input = np.size(curves[curves.keys()[0]],0)

	slopes = {}
	all_slopes = {}
	all_deg = {}
	all_dist_from_target = []
	all_slope_at_target = []

	for r in W.keys():
		slopes[r] = np.abs(curves[r] - np.roll(curves[r], 1, axis=0))
		all_slopes[r] =[]
		all_deg[r] =[]

		dist_target = pref_ori[r] - target_ori
		dist_target[dist_target>90]-=180
		dist_target[dist_target<-90]+=180
		target_idx = int(target_ori * (n_input/180))
		slope_at_target = slopes[r][target_idx, np.arange(nHidNeurons)]

		all_dist_from_target.append(dist_target)
		all_slope_at_target.append(slope_at_target)

		if plot: 
			fig, ax = plt.subplots()
		
		for o in np.arange(0,180,10):
			deg_idx = int(o * (n_input/180))
		
			deg = pref_ori[r]-o
			deg[deg>90]-=180
			deg[deg<-90]+=180
			y = slopes[r][deg_idx, :]
			
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
		
			plt.savefig('output/' + runName + '/TCs/' + 'slopes_' + runName+ '_' + str(r).zfill(3))
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

			plt.savefig('output/' + runName + '/TCs/' + 'slopes_at_target')
			plt.close(fig)

	return {'slopes':slopes, 'all_slopes':all_slopes, 'all_deg':all_deg, 'all_dist_from_target':all_dist_from_target, 'all_slope_at_target':all_slope_at_target}








































