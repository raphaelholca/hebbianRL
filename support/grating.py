import numpy as np
import matplotlib.pyplot as plt
import external as ex
ex = reload(ex)

def gabor(size=28, lambda_freq=5, theta=0, sigma=5, phase=0, noise=0):
	"""Creates a Gabor patch

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
	gratings *= gauss #add Gaussian trim
	gratings += np.random.normal(0.0, noise, size=(size, size)) #add Gaussian noise
	gratings -= np.min(gratings)

	gratings = np.reshape(gratings, (n_gratings, size**2))

	return gratings

# np.random.seed(100)

# im_side = 28
# im_number = 10

# im_cycles = 2.6*2.3 #(deg*cycle/deg) from Schoups et al., 2001
# im_freq = np.round(im_side/im_cycles) #spatial frequency of the grating (pixel per cycle)
# noise_level = 0.

# orientations = np.random.random(im_number)*180 #orientations of gratings (in degrees)

# #create gratings
# gratings = gabor(size=im_side, lambda_freq=im_freq, theta=orientations, sigma=im_side/5., phase=np.random.random(im_number), noise=noise_level)
# A=940.8
# gratings = ex.normalize(gratings, A)

# print orientations

# for i in range(im_number):
# 	plt.figure()
# 	plt.imshow(np.reshape(gratings[i,:], (im_side, im_side)), interpolation='nearest', cmap='Greys')


# plt.show(block=False)







































