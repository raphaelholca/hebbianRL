"""
Author: Raphael Holca-Lamarre
Date: 26/05/2015

This function creates a convolutional hebbian neural network
"""

import numpy as np
import matplotlib.pyplot as plt
import recon as rc
import sys
import pickle
import numba
import time
import support.accel as accel
import support.mnist as mnist
import support.external as ex
from progressbar import ProgressBar

ex = reload(ex)
rc = reload(rc)

""" support functions """

def add_padding(images, pad_size, pad_value=0.):
	""" 
	Adds padding of value 1 around images and reshapes images vector to 3D (1D all images + 2D spatial dimensions).

	Args:
		images (2D numpy array): images to add the padding to; size = (images_num x nPixels_1D)
		pad_size (int): amount of padding to add on each 4 sides of the 2D image
		pad_value (float, optional): value of the padding to add

	returns:
		3D numpy array; size = (images_num x nPixels_2D x nPixels_2D)
	"""

	images_num = np.size(images, 0)
	sqrtPixels = int(np.sqrt(np.size(images,1)))
	images = np.reshape(images, (images_num, sqrtPixels, sqrtPixels))
	images_padded = np.ones((images_num, sqrtPixels+pad_size*2, sqrtPixels+pad_size*2))*pad_value

	images_padded[:, pad_size-1:sqrtPixels+pad_size-1, pad_size-1:sqrtPixels+pad_size-1] = images

	return images_padded

@numba.njit
def get_conv_input(image, input_conv, conv_side):
	"""
	Gets the input to the convolving weight matrix

	Args:
		image (2D numpy array): image to get the input from; size = (images_side x images_side)
		input_conv (2D numpy array): empty input array to be filled; size = (conv_neuronNum x conv_side**2)
		conv_side (int): size of the convolutional filter

	returns:
		2D numpy array; size = (conv_neuronNum x conv_side**2)
	"""
	images_side = image.shape[0]

	im=0
	for i in range(images_side-conv_side+1):
		for j in range(images_side-conv_side+1):
			select = image[i:i+conv_side, j:j+conv_side]
			km=0
			for k in range(conv_side):
				for l in range(conv_side):
					input_conv[im,km] = select[k,l]
					km+=1
			im+=1
	return input_conv	

def subsampling(FM_lin, L1_conv_mapSide, L1_mapNum, L1_subs_mapSide):
	FM = np.reshape(FM_lin, (L1_conv_mapSide, L1_conv_mapSide, L1_mapNum))
	SSM = np.zeros((L1_subs_mapSide, L1_subs_mapSide, L1_mapNum))
	ite = np.arange(0, L1_conv_mapSide, 2)
	SSM = subsampling_numba(FM, SSM, ite)
	SSM = ex.softmax( np.reshape(SSM, (L1_subs_mapSide**2, L1_mapNum) ) )
	SSM = np.reshape(SSM, (L1_subs_mapSide, L1_subs_mapSide, L1_mapNum))
	SSM_lin = np.reshape(SSM, (-1))[np.newaxis,:]

	return SSM_lin

@numba.njit
def subsampling_numba(FM, SSM, ite):
	"""
	Subsamples the convolutional feature maps

	Args:
		FM (3D numpy array): feature maps; size = (cMaps_side x cMaps_side x nFeatureMaps)
		SSM (3D numpy array): empty subsampled feature maps to be filled; size = (cMaps_side/2 x cMaps_side/2 x nFeatureMaps)
		ite (1D numpy array): iterator used over FM (contains np.arange(0, cMaps_side, 2))

	returns:
		3D numpy array; subsampled feature maps; size = (cMaps_side/2 x cMaps_side/2 nFeatureMaps)
	"""
	for f in range(FM.shape[2]):
		for im in range(ite.shape[0]):
			i=ite[im]
			for jm in range(ite.shape[0]):
				j=ite[jm]
				select = FM[i:i+2,j:j+2,f]
				tmp_sum=0
				for k in range(2):
					for l in range(2):
						tmp_sum += select[k,l]
				SSM[im,jm,f] = tmp_sum
	return SSM

def shuffle_images(images, labels):
	"""
	Shuffles images and labels
	"""

	rdnIndex = range(images.shape[0])
	np.random.shuffle(rdnIndex)
	rndImages = images[rdnIndex,:,:]
	rndLabels = labels[rdnIndex]

	return rndImages, rndLabels

def load_images(classes, dataset, imPath, pad_size):
	"""
	Loads and pads images
	"""

	print "importing and pre-processing " + dataset + " images..."
	images, labels = mnist.read_images_from_mnist(classes = classes, dataset = dataset, path = imPath)
	images, labels = ex.evenLabels(images, labels, classes)
	# images = ex.normalize_numba(images, 1080)
	images = add_padding(images, pad_size=pad_size, pad_value=0)
	# images+=5. ## testing the effect of lower contrast on long training...
	# images[images==0.]=5. ## testing the effect of non-zero back-ground on training...
	images += 1e-5 #to avoid division by zero error when the convolving filter takes as input a patch of images that is filled with 0s

	return images, labels

def propagate(image, L1_conv_W, L2_feedf_W, L3_class_W, A, t, size_params):
	"""
	propagates a single image through the network and return its classification
	"""
	#get params from size_params dict
	L1_conv_neuronNum = size_params['L1_conv_neuronNum']
	L1_conv_filterSide = size_params['L1_conv_filterSide']
	L1_conv_mapSide = size_params['L1_conv_mapSide']
	L1_mapNum = size_params['L1_mapNum']
	L1_subs_mapSide = size_params['L1_subs_mapSide']

	#get input to the convolutional filter
	input_conv = np.zeros((L1_conv_neuronNum, L1_conv_filterSide**2))
	input_conv = get_conv_input(image, input_conv, L1_conv_filterSide)
	input_conv = ex.normalize_numba(input_conv, A)

	#activate convolutional feature maps
	FM_lin = ex.propL1(input_conv, L1_conv_W, SM=True, t=t)

	#subsample feature maps
	SSM_lin = subsampling(FM_lin, L1_conv_mapSide, L1_mapNum, L1_subs_mapSide)

	#activate feedforward layer
	FF_lin = ex.propL1(SSM_lin, L2_feedf_W, SM=True, t=t)
	
	#activate classification layer
	class_lin = ex.propL1(FF_lin, L3_class_W, SM=True, t=0.001)

	return np.argmax(class_lin)

""" define parameters """
classes 	= np.array([ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
# classes 	= np.array([0,1], dtype=int)
# classes 	= np.array([4,7,9], dtype=int)
nClasses = len(classes)

runName 			= 't_1'
nEpi 				= 1
nCrit 				= 6
DA 					= True
A 					= 200.
lr 					= 1e-5
dataset 			= 'train'
nBatch 				= 196  #196 112 49
L1_mapNum 			= 12
L1_conv_filterSide	= 5
L2_feedf_neuronNum 	= 49
L3_class_neuronNum 	= nClasses
seed 				= 970#np.random.randint(1000) #970

np.random.seed(seed)
print '\n' + 'run name: ' + runName + ' -- seed: ' + str(seed) + '\n'

""" load and pre-process data """
pad_size = (L1_conv_filterSide-1)/2
imPath = '/Users/raphaelholca/Documents/data-sets/MNIST'
images, labels = load_images(classes, dataset, imPath, pad_size)

##reduce size of training set
images, labels = shuffle_images(images, labels)
images = images[:10000, :,:]
labels = labels[:10000]
##

##duplicate size of test dataset to test effect of size
# images_ =np.zeros((images.shape[0]*6, images.shape[1], images.shape[2]))
# labels_ =np.zeros(labels.shape[0]*6, dtype=int)
# for i in range(6):
# 	images_[images.shape[0]*i:images.shape[0]*(i+1), :, :] = images
# 	labels_[labels.shape[0]*i:labels.shape[0]*(i+1)] = labels
# images = images_
# labels = labels_
##

if dataset=='train':
	images_test, labels_test = load_images(classes, 'test', imPath, pad_size)
else: 
	images_test, labels_test = load_images(classes, 'train', imPath, pad_size)
step = labels_test.shape[0]/100
images_test_short = images_test[::step]
labels_test_short = labels_test[::step]

""" initialize variables """
#input
images_num = np.size(images, 0)
images_side = np.size(images,2)

#layer 1 (convolutional feature maps)
L1_conv_neuronNum = (images_side-L1_conv_filterSide+1)**2
L1_conv_mapSide = int(np.sqrt(L1_conv_neuronNum))
L1_subs_mapSide = L1_conv_mapSide/2

size_params = {'L1_conv_neuronNum':L1_conv_neuronNum, 'L1_conv_filterSide':L1_conv_filterSide, 'L1_conv_mapSide':L1_conv_mapSide, 'L1_mapNum':L1_mapNum, 'L1_subs_mapSide':L1_subs_mapSide}

#weights
if False: #load pre-trained weights
	# f = open('w_corners', 'r')
	# L1_conv_W = pickle.load(f)
	# f.close()

	f = open('weights_long', 'r')
	weights = pickle.load(f)
	f.close()

	L1_conv_W = weights['L1_conv_W']
	L1_subs_W = weights['L1_subs_W']
	L2_feedf_W = weights['L2_feedf_W']
	L3_class_W = weights['L3_class_W']
else:
	L1_conv_W = np.random.random_sample(size=(L1_conv_filterSide**2, L1_mapNum)) + A/(L1_conv_filterSide**2) + 2.5 ##0.5
	L1_subs_W = np.random.random_sample(size=((L1_subs_mapSide**2)*L1_mapNum, (L1_subs_mapSide**2)*L1_mapNum)) + 1.5 ##for now, no trainable weights between cMaps and sMaps...
	L2_feedf_W = np.random.random_sample(size=((L1_subs_mapSide**2)*L1_mapNum, L2_feedf_neuronNum))/1000 + float(L1_subs_mapSide**2)/((L1_subs_mapSide**2)*L1_mapNum) + 0.6
	L3_class_W = (np.random.random_sample(size=(L2_feedf_neuronNum, L3_class_neuronNum))/1000+1.0)/L2_feedf_neuronNum

imcount = 0

""" training network """
print "training network..."
for e in range(nEpi):
	print '\ntrain epi: ' + str(e+1) + '/' + str(nEpi)
	rndImages, rndLabels = shuffle_images(images, labels)

	correct = 0.
	last_neuron_class = np.zeros((L2_feedf_neuronNum, L3_class_neuronNum))
	dopa_save = []

	pbar_epi = ProgressBar()
	for i in pbar_epi(range(rndImages.shape[0])):
		imcount+=1
		#get input to the convolutional filter
		input_conv = np.zeros((L1_conv_neuronNum, L1_conv_filterSide**2))
		input_conv = get_conv_input(rndImages[i,:,:], input_conv, L1_conv_filterSide)
		input_conv = ex.normalize_numba(input_conv, A)

		#activate convolutional feature maps
		FM_lin = ex.propL1(input_conv, L1_conv_W, SM=True, t=0.01)

		#subsample feature maps
		SSM_lin = subsampling(FM_lin, L1_conv_mapSide, L1_mapNum, L1_subs_mapSide)

		#activate feedforward layer
		FF_lin = ex.propL1(SSM_lin, L2_feedf_W, SM=False)
		
		#add noise in FF_lin
		FF_lin_noise = np.copy(FF_lin)
		if DA and e>=nCrit and e<nEpi: FF_lin_noise += np.random.uniform(0, 50, np.shape(FF_lin)) ##param explore, optimize
		FF_lin_noise = ex.softmax(FF_lin_noise, t=0.01)
		FF_lin = ex.softmax(FF_lin, t=0.01)

		dopa = None

		#activate classification layer
		class_lin = ex.propL1(FF_lin, L3_class_W, SM=True, t=0.001)

		if DA and e>=nCrit: #DOPA
			#activate classification layer with noise
			class_lin_noise = ex.propL1(FF_lin_noise, L3_class_W, SM=True, t=0.001)

			#compute reward and dopamine signal
			reward = ex.compute_reward(ex.label2idx(classes, [rndLabels[i]]), np.argmax(class_lin_noise))

			# dopa = ex.compute_dopa([np.argmax(class_lin)], [np.argmax(class_lin_noise)], reward, dHigh=7.0, dMid=0.01, dNeut=-0.0, dLow=-2.0) #parameters from old network
			dopa = ex.compute_dopa([np.argmax(class_lin)], [np.argmax(class_lin_noise)], reward, dHigh=2.0, dMid=0.00, dNeut=-0.02, dLow=-0.5) #OK paramters for feedforward layer alone
			# dopa = ex.compute_dopa([np.argmax(class_lin)], [np.argmax(class_lin_noise)], reward, dHigh=2.0, dMid=1.00, dNeut=-0.02, dLow=-0.5) #testing parameters for convolutional layer

			dopa_save.append(dopa[0])

		last_neuron_class[np.argmax(FF_lin), np.argwhere(rndLabels[i]==classes)] += 1

		# learn weights...
		if imcount<60000:
			for b in range(L1_conv_neuronNum/nBatch):
				#...of the convolutional matrices
				dW_conv = ex.learningStep(input_conv[b*nBatch:(b+1)*nBatch-1, :], FM_lin[b*nBatch:(b+1)*nBatch-1, :], L1_conv_W, lr=lr*0.1, disinhib=dopa)
				# if e>= 8: dW_conv[:,[0,2,3,4,7,8,11]] = np.zeros((L1_conv_filterSide**2,7)) ##prevents learning of 1 weight...
				L1_conv_W += dW_conv
				L1_conv_W = np.clip(L1_conv_W, 1e-10, np.inf)

		#...of the feedforward layer
		dW_FF = ex.learningStep(SSM_lin, FF_lin_noise, L2_feedf_W, lr=lr*3600, disinhib=dopa)
		L2_feedf_W += dW_FF
		L2_feedf_W = np.clip(L2_feedf_W, 1e-10, np.inf)

		#...of the classification layer
		dW_class = ex.learningStep(FF_lin_noise, class_lin, L3_class_W, lr=0.005)
		
		if np.argmax(class_lin) == ex.label2idx(classes, [rndLabels[i]]):
			dW_class *= 0.75
			correct+=1
		else:
			dW_class *= -0.5

		L3_class_W += dW_class
		L3_class_W = np.clip(L3_class_W, 1e-10, np.inf)

		if np.isnan(L3_class_W).any(): import pdb; pdb.set_trace()

	print 'train error: ' + str(np.round((1.-correct/rndImages.shape[0])*100,2)) +'%'

	correct_test = 0.
	for i in range(images_test_short.shape[0]):
		classif = propagate(images_test_short[i,:,:], L1_conv_W, L2_feedf_W, L3_class_W, A, 0.01, size_params)
		if classif == labels_test_short[i]: correct_test += 1.
	print 'test error: ' + str(np.round((1.-correct_test/images_test_short.shape[0])*100,2)) +'%'

	correct_Wout = np.sum(np.argmax(last_neuron_class,1)==np.argmax(L3_class_W,1))
	print 'correct W_out assignment: ' + str(correct_Wout) + '/' + str(L2_feedf_neuronNum)

""" test network """
print '\ntest epi'
step = 8920/images_test.shape[0]
images_test=images_test[::step,:,:]
labels_test=labels_test[::step]

correct = 0.
pbar_epi = ProgressBar()
for i in pbar_epi(range(images_test.shape[0])):
	classif = propagate(images_test[i,:,:], L1_conv_W, L2_feedf_W, L3_class_W, A, 0.01, size_params)
	if classif == labels_test[i]: correct += 1.
print 'test error: ' + str(np.round((1.-correct/images_test.shape[0])*100,2)) +'%'


""" plot convolutional filter """
# nRows = int(np.sqrt(L1_mapNum))
# nCols = np.ceil(L1_mapNum/float(nRows))
# fig = plt.figure(figsize=(nCols,nRows))
# for f in range(L1_mapNum):
# 	plt.subplot(nRows, nCols, f)
# 	plt.imshow(np.reshape(L1_conv_W[:,f], (L1_conv_filterSide,L1_conv_filterSide)), interpolation='nearest', cmap='Greys', vmin=np.min(L1_conv_W), vmax=np.max(L1_conv_W))
# 	# plt.imshow(np.reshape(L1_conv_W[:,f], (L1_conv_filterSide,L1_conv_filterSide)), interpolation='nearest', cmap='Greys', vmin=np.min(L1_conv_W[:,f]), vmax=np.max(L1_conv_W[:,f]))
# 	plt.xticks([])
# 	plt.yticks([])
# fig.patch.set_facecolor('white')
# plt.subplots_adjust(left=0., right=1., bottom=0., top=1., wspace=0., hspace=0.)
# plt.show(block=False)

""" plot output neuron RF reconstruction """
# nRows = int(np.sqrt(L2_feedf_neuronNum))
# nCols = L2_feedf_neuronNum/nRows
# fig = plt.figure(figsize=(nCols,nRows))
# for n in range(L2_feedf_neuronNum):
# 	plt.subplot(nRows, nCols, n)
# 	W = np.reshape(L2_feedf_W[:,n], (L1_subs_mapSide, L1_subs_mapSide, L1_mapNum))
# 	rc.recon(L1_conv_W, W, display_all=False)
# 	plt.xticks([])
# 	plt.yticks([])
# fig.patch.set_facecolor('white')
# plt.subplots_adjust(left=0., right=1., bottom=0., top=1., wspace=0., hspace=0.)
# plt.show(block=False)
	



































