""" Hebbian learning-based classifier used to assess the performance of the representation """

import numpy as np
import matplotlib.pyplot as pyplot
import helper.external as ex
import helper.assess_network as an
import helper.bayesian_decoder as bc
import pickle

ex = reload(ex)
an = reload(an)
bc = reload(bc)

def neural(net, images, labels):
	"""
	evaluates the quality of a representation using the action neurons of the network.

	Args:
		images (numpy array): test images
		labels (numpy array): test labels
	"""

	if net.verbose: print "\nassessing performance..."

	""" variable initialization """
	CM_all = []
	perf_all = []

	for iw in range(net.n_runs):
		if net.verbose: print 'run: ' + str(iw+1)
		W_in = net.hid_W_runs[iw,:,:]
		W_act = net.out_W_runs[iw,:,:]

		""" testing of the classifier """
		hidNeurons = ex.propagate_layerwise(images, W_in, t=net.t)
		actNeurons = ex.propagate_layerwise(hidNeurons, W_act)
		classIdx = np.argmax(actNeurons, 1)
		classResults = net.classes[classIdx]
		
		""" compute classification performance """
		correct_classif = float(np.sum(classResults==labels))
		perf_all.append(correct_classif/len(labels))
		CM = compute_CM(classResults, labels, net.classes)
		CM_all.append(CM)

	""" print and save performance measures """
	perf_dict = print_save(CM_all, perf_all, net.classes, net.name, net.verbose, net.save_data)
	return perf_dict

def bayesian(net, images, labels, images_test, labels_test):
	"""
	evaluates the performance of the newtork using a bayesian decoder

	Args:
		images (numpy array): train images
		labels (numpy array): train labels
		images_test (numpy array): test images
		labels_test (numpy array): test labels
	"""

	if net.verbose: print "\nassessing performance..."

	""" variable initialization """
	CM_all = []
	perf_all = []

	for iw in sorted(net.hid_W_runs.keys()):
		if net.verbose: print 'run: ' + str(int(iw)+1)
		W_in = net.hid_W_runs[iw]

		""" compute pdf """
		pdf_marginals, pdf_evidence, pdf_labels = bc.pdf_estimate(images, labels, W_in, net.pdf_method, net.t)

		""" testing of the classifier """
		posterior = bc.bayesian_decoder(ex.propagate_layerwise(images_test, W_in, t=net.t), pdf_marginals, pdf_evidence, pdf_labels, net.pdf_method)
		classIdx = np.argmax(posterior, 1)
		classResults = net.classes[classIdx]
		
		""" compute classification performance """
		correct_classif = float(np.sum(classResults==labels_test))
		perf_all.append(correct_classif/len(labels_test))
		CM = compute_CM(classResults, labels_test, net.classes)
		CM_all.append(CM)

	""" print and save performance measures """
	perf_dict = print_save(CM_all, perf_all, net.classes, net.name, net.verbose, net.save_data)
	return perf_dict

def compute_CM(classResults, labels_test, classes):
	"""
	Computes the confusion matrix for a set of classification results

	Args:
		classResults (numpy array): result of the classifcation task
		labels_test (numpy array): labels of the test dataset
		classes (numpy array): all classes of the MNIST dataset used in the current run

	returns:
		numpy array: confusion matrix of shape (actual class x predicted class)
	"""

	nClasses = len(classes)
	confusMatrix = np.zeros((nClasses, nClasses))
	for ilabel,label in enumerate(classes):
		for iclassif, classif in enumerate(classes):
			classifiedAs = np.sum(np.logical_and(labels_test==label, classResults==classif))
			overTot = np.sum(labels_test==label)
			confusMatrix[ilabel, iclassif] = float(classifiedAs)/overTot
	
	return confusMatrix

def print_save(CM_all, perf_all, classes, name, verbose, save_data):
	""" print and save performance measures """
	CM_avg = np.mean(CM_all,0)
	CM_ste = np.std(CM_all,0)/np.sqrt(np.shape(CM_all)[0])
	perf_avg = np.mean(perf_all)
	perf_avg = np.std(perf_all)/np.sqrt(len(perf_all))

	if verbose:
		perf_print = ''
		perf_print += '\naverage confusion matrix:' + '\n'
		c_str = ''
		for c in classes: c_str += str(c).rjust(6)
		perf_print += c_str + '\n'
		perf_print += '-'*(len(c_str)+3) + '\n'
		perf_print += str(np.round(CM_avg,2)) + '\n'
		perf_print += '\naverage correct classification:' + '\n'
		perf_print += str(np.round(100*perf_avg,2)) + ' +/- ' + str(np.round(100*perf_avg,2)) + ' %' + '\n'
		if len(perf_all)>1:
			perf_print += '\nof which best performance is:' + '\n'
			perf_print += str(np.round(100*(np.max(perf_all)),2)) + '%' + ' (run ' + str(np.argmax(perf_all)) + ')' + '\n'
			perf_print += 'and worse performance is:' + '\n'
			perf_print += str(np.round(100*(np.min(perf_all)),2)) + '%' + ' (run ' + str(np.argmin(perf_all)) + ')' + '\n'

		print perf_print

	if save_data:
		perf_file = open('./output/' + name + '/' +name+ '_perf.txt', 'w')
		perf_file.write(perf_print)
		perf_file.close()

		fig = an.plot_CM(CM_avg, classes)
		pyplot.savefig('./output/' + name + '/' +name+ '_avgCM.pdf')
		pyplot.close(fig)

	return {'CM_all':CM_all, 'CM_avg':CM_avg, 'CM_ste':CM_ste, 'perf_all':perf_all, 'perf_avg':perf_avg, 'perf_ste':perf_avg}


















