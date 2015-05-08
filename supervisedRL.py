import numpy as np
import matplotlib.pyplot as plt
import support.mnist as mnist
import support.external as ex

ex = reload(ex)

np.random.seed(12)

def sigmoid(x):
	# print 'x : ' + str(np.round(x[0][0],0))
	try:
		x[x<-100]=-100
		return 1/(1+np.exp(-x))
	except:
		raise OverflowError('math range error: x='+str(np.round(x[0][0],1)))

def evenLabels(images, labels, nClasses):
	nDigits, bins = np.histogram(labels, bins=nClasses, range=(0,nClasses-1))
	m = np.min(nDigits)
	images_even = np.zeros((m*nClasses, np.size(images,1)))
	labels_even = np.zeros(m*nClasses, dtype=int)
	for c in range(nClasses):
		images_even[c*m:(c+1)*m,:] = images[labels==c,:][0:m,:]
		labels_even[c*m:(c+1)*m] = labels[labels==c][0:m]
	images, labels = np.copy(images_even), np.copy(labels_even)
	return images, labels

def update(step, dE, prevdE, dW, W):
	#compute weight update and adaptation
	s = np.sign(prevdE * dE)

	step[s==+1] *= npos
	step[s==-1] *= nneg
	step = np.clip(step, dmin, dmax)
	dW[s==+1] = step[s==+1] * np.sign(dE[s==+1])
	dW[s==0] = step[s==0] * np.sign(dE[s==0])
	dE[s==-1] = 0.

	W -= dW
	prevdE = dE
	return step, dE, prevdE, dW, W

# classes =  [ 0 , 1 ]
classes =  [ 4 , 9 ]
# classes =  [ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ]

nClasses = len(classes)
dataset = 'train'
imPath = '/Users/raphaelholca/Documents/data-sets/MNIST'
images, labels = mnist.read_images_from_mnist(classes = classes, dataset = dataset, path = imPath)
##? Normalize each image to the sum of its pixel values (due to feedforward inhibition in model)
# A=900
# images = (A-images.shape[1])*images/np.sum(images,1)[:,np.newaxis] + 1.
images/=255. #normalize images to range [0,1]
images, labels = ex.evenLabels(images, labels, classes)
# images, labels = evenLabels(images, labels, nClasses) #even out label class distribution

nImages = np.size(images,0)
nDimActions = 0 ##
nDimStates = np.size(images,1)
nInpNeurons = nDimStates + nDimActions + 1 # +1 for bias node
nHidNeurons = 49
nEpi = 50
npos, nneg = 1.2, 0.5
dmin, dmax, dinit = 0.000001, 1.0, 0.1
Q = 0. #output neuron; Q-value
hidNeurons = np.zeros(nHidNeurons)[np.newaxis,:] #activation of hidden neurons
#Output weights (hidden -> output)
W_out = np.random.rand(nHidNeurons,1)-0.5 #weight matrix
stepSize_out = np.ones(np.shape(W_out))*dinit
dW_out = np.zeros(np.shape(W_out))
dE_out = np.zeros(np.shape(W_out)) #error gradient
prevdE_out = np.zeros(np.shape(W_out)) #gradient at previous step
#Hidden weights (input -> hidden)
W_in = np.random.rand(nInpNeurons, nHidNeurons)-0.5 #weight matrix
stepSize_in = np.ones(np.shape(W_in))*dinit
dW_in = np.zeros(np.shape(W_in))
dE_in = np.zeros(np.shape(W_in)) #error gradient
prevdE_in = np.zeros(np.shape(W_in)) #gradient at previous step

allStates = np.ones((nImages, nDimStates+1)) #add 1 to input vector for bias node
allStates[:,:-1] = images
concInput = np.zeros(nInpNeurons) #concatenated input vector with state and action input and bias node
# allActions = np.array([0, 1], dtype=int)
# allRewards = {'00':1, '01':0, '10':0, '11':1, '40':1, '41':0, '90':0, '91':1} #state+action

#training
for e in range(nEpi):
	batchErr = 0.
	dE_out = np.zeros(np.shape(W_out))
	dW_out = np.zeros(np.shape(W_out))
	dE_in = np.zeros(np.shape(W_in))
	dW_in = np.zeros(np.shape(W_in))
	for idxState in range(nImages):
		cState = allStates[idxState, :]
		# cAction = np.random.choice(allActions) #choose a random action
		concInput[:nDimStates+1] = cState #concatenate current state and action to be fed to the network
		##concInput[nDimStates+1:] = cAction
		##cReward = allRewards[str(labels[idxState])+str(cAction)] #get reward associated with state and action
		cReward = labels[idxState] ##

		hidNeurons = sigmoid(np.dot(concInput, W_in))[np.newaxis,:]
		Q = sigmoid(np.dot(hidNeurons, W_out)) #compute Q-value/neuron activation
		# print cReward, np.round(Q[0][0],1)
		batchErr += np.abs(Q-cReward)
		dE_out += np.dot(hidNeurons.T, (Q-cReward)*Q*(1-Q)) #partial derivatives of the error with respect to the output weights
		dE_in += np.dot((concInput*(1-concInput))[:,np.newaxis], hidNeurons)*(dE_out*W_out).T/hidNeurons

	print 'epi ' + str(e+1) + ' train err : ' + str(np.round(batchErr[0][0]/nImages,5))
	# print np.sum(stepSize_in), np.sum(stepSize_out)

	stepSize_in, dE_in, prevdE_in, dW_in, W_in, = update(stepSize_in, dE_in, prevdE_in, dW_in, W_in,)
	stepSize_out, dE_out, prevdE_out, dW_out, W_out = update(stepSize_out, dE_out, prevdE_out, dW_out, W_out)


#testing
if True:
	print 'testing...'
	dataset = 'test'
	images_t, labels_t = mnist.read_images_from_mnist(classes = classes, dataset = dataset, path = imPath)
	images_t/=255.
	images_t, labels_t = evenLabels(images_t, labels_t)
	nImages_t = np.size(images_t,0)

	allStates = np.ones((nImages_t, nDimStates+1)) #add 1 to input vector for bias node
	allStates[:,:-1] = images_t

	totErr = 0.
	countErr = 0
	for idxState in range(nImages_t):
		cState = allStates[idxState, :]
		# cAction = np.random.choice(allActions) #choose a random action
		concInput[:nDimStates+1] = cState #concatenate current state and action to be fed to the network
		##concInput[nDimStates+1:] = cAction
		##cReward = allRewards[str(labels[idxState])+str(cAction)] #get reward associated with state and action
		cReward = labels_t[idxState] ##

		hidNeurons = sigmoid(np.dot(concInput, W_in))[np.newaxis,:]
		Q = sigmoid(np.dot(hidNeurons, W_out))
		# print cReward, np.round(Q[0][0],1)
		err = np.abs(Q-cReward)
		totErr += err
		if err >= 0.1: countErr += 1
	print '\ntest error: ' + str(np.round(totErr[0][0]/nImages_t,3))
	print 'test error count: ' + str(countErr) + '/' + str(nImages_t) + ' : ' + str(np.round(float(countErr)/nImages_t,3))






