""" evaluates the quality of a representation using a neural classifier """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pyplot
import pickle
from configobj import ConfigObj
import support.external as ex
import support.plots as pl
import support.mnist as mnist
from support import accel
ex = reload(ex)
pl = reload(pl)

""" parameters """
runName 		= 'test'		# results to load
nEpi 			= 5					# number of episode for the training of the classifier
nBatch 			= 100				# size of the mini-batch for the training
dataset_train 	= 'train'			# dataset to use for training
dataset_test 	= 'test'			# dataset to use for testing
lr 				= 0.1				# learning rate

""" load data from files """
settings = {}
settings = ConfigObj('output/' + runName + '/settings.txt')
classes  = np.array(map(int, settings['classes']))
rActions = np.array(map(str, settings['rActions']))
nHidNeurons = int(settings['nHidNeurons'])
nDimStates = int(settings['nDimStates'])
pFile = open('output/' + runName + '/W_in', 'r')
W_in_save = pickle.load(pFile)
pFile.close()
pFile = open('output/' + runName + '/W_class', 'r')
W_class_save = pickle.load(pFile)
pFile.close()

""" load and pre-process images """
print "importing data..."
imPath = '../data-sets/MNIST'
# images_train, labels_train = mnist.read_images_from_mnist(classes=classes, dataset=dataset_train, path=imPath)
# images_train = ex.normalize(images_train, int(settings['A']))
# images_train, labels_train = ex.evenLabels(images_train, labels_train, classes)

images_test, labels_test = mnist.read_images_from_mnist(classes=classes, dataset=dataset_test, path=imPath)
images_test = ex.normalize(images_test, int(settings['A']))
images_test, labels_test = ex.evenLabels(images_test, labels_test, classes)

""" variable initialization """
nClasses = len(classes)
# nImages_train = np.size(images_train, 0)
nImages_test = np.size(images_test, 0)
lr *= nClasses/np.float(nBatch) #learning rate optimize for number of neuron and mini-batch size
allCMs = []
allPerf = []

""" training of the neural classifier """
for iw in sorted(W_in_save.keys()):
	print 'run: ' + str(int(iw)+1)
	hidNeurons = np.zeros((nBatch, nHidNeurons))
	classNeurons = np.zeros((nBatch, nClasses))
	W_in = W_in_save[iw][0:nDimStates,:]
	W_class = W_class_save[iw] #np.random.random_sample(size=(nHidNeurons, nClasses)) + 1.

	# for e in range(nEpi):
		# #shuffle input
		# rndInput, rndLabel, rndIdx = ex.shuffle(images_train, labels_train)

		# #compute activation of hid and class neurons
		# hidNeurons = ex.propL1(rndInput, W_in)
		# classNeurons = np.zeros((nImages_train, nClasses))
		# labelsIdx = ex.label2idx(classes, rndLabel)
		# classNeurons[np.arange(nImages_train),labelsIdx] = 1.0


		# #train network, with mini-batch training
		# for b in range(int(nImages_train/nBatch)): #may leave a few training examples out
		# 	bHidNeurons = hidNeurons[b*nBatch:(b+1)*nBatch,:]
		# 	bClassNeurons = classNeurons[b*nBatch:(b+1)*nBatch,:]
		# 	W_class += ex.learningStep(bHidNeurons, bClassNeurons, W_class, lr)

	""" testing of the classifier """
	hidNeurons = ex.propL1(images_test, W_in)
	classNeurons = ex.propL2_class(hidNeurons, W_class)
	classIdx = np.argmax(classNeurons, 1)
	classResults = classes[classIdx]
	
	""" compute classification performance """
	allPerf.append(float(np.sum(classResults==labels_test))/len(labels_test))
	allCMs.append(ex.computeCM(classResults, labels_test, classes))

""" print and save """
avgCM = np.mean(allCMs,0)
steCM = np.std(allCMs,0)/np.sqrt(np.shape(allCMs)[0])
avgPerf = np.mean(allPerf)
stePerf = np.std(allPerf)/np.sqrt(len(allPerf))

pFile = open('output/' + runName + '/W_class', 'w')
pickle.dump(W_class, pFile)
pFile.close()

pFile = open('output/' + runName + '/classResults', 'w')
pDict = {'allCMs':allCMs, 'avgCM':avgCM, 'steCM':steCM, 'allPerf':allPerf, 'avgPerf':avgPerf, 'stePerf':stePerf}
pickle.dump(pDict, pFile)
pFile.close()

print '\naverage confusion matrix:'
print '   0     1     2     3     4     5     6     7     8     9    '
print '--------------------------------------------------------------'
print np.round(avgCM,2)
print '\naverage correct classification:'
print str(np.round(100*avgPerf,1)) + ' +/- ' + str(np.round(100*stePerf,1)) + '%'

fig = pl.plotCM(avgCM, classes)
pyplot.savefig('output/' + runName + '/avgCM.png')
# pyplot.close(fig)
pyplot.show(block=False)






















