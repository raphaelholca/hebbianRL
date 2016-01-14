""" Hebbian learning-based classifier used to assess the performance of the representation """

import numpy as np
import matplotlib.pyplot as pyplot
import helper.mnist as mnist
import helper.external as ex
import helper.plots as pl
import helper.bayesian_decoder as bc
import pickle
from sklearn.svm import SVC

ex = reload(ex)
pl = reload(pl)
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
	allCMs = []
	allPerf = []

	for iw in sorted(net.hid_W_runs.keys()):
		if net.verbose: print 'run: ' + str(int(iw)+1)
		W_in = net.hid_W_runs[iw]
		W_act = net.out_W_runs[iw]

		""" testing of the classifier """
		hidNeurons = ex.propagate_layerwise(images, W_in, t=net.t)
		actNeurons = ex.propagate_layerwise(hidNeurons, W_act)
		classIdx = np.argmax(actNeurons, 1)
		classResults = net.classes[classIdx]
		
		""" compute classification performance """
		correct_classif = float(np.sum(classResults==labels))
		allPerf.append(correct_classif/len(labels))
		CM = ex.computeCM(classResults, labels, net.classes)
		allCMs.append(CM)

	""" print and save performance measures """
	print_save(allCMs, allPerf, net.classes, net.name, net.verbose, net.save_data)
	return allCMs, allPerf

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
	allCMs = []
	allPerf = []

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
		allPerf.append(correct_classif/len(labels_test))
		CM = ex.computeCM(classResults, labels_test, net.classes)
		allCMs.append(CM)

	""" print and save performance measures """
	print_save(allCMs, allPerf, net.classes, net.name, net.verbose, net.save_data)
	return allCMs, allPerf

def print_save(allCMs, allPerf, classes, name, verbose, save_data):
	""" print and save performance measures """
	avgCM = np.mean(allCMs,0)
	steCM = np.std(allCMs,0)/np.sqrt(np.shape(allCMs)[0])
	avgPerf = np.mean(allPerf)
	stePerf = np.std(allPerf)/np.sqrt(len(allPerf))

	if verbose:
		perf_print = ''
		perf_print += '\naverage confusion matrix:' + '\n'
		c_str = ''
		for c in classes: c_str += str(c).rjust(6)
		perf_print += c_str + '\n'
		perf_print += '-'*(len(c_str)+3) + '\n'
		perf_print += str(np.round(avgCM,2)) + '\n'
		perf_print += '\naverage correct classification:' + '\n'
		perf_print += str(np.round(100*avgPerf,2)) + ' +/- ' + str(np.round(100*stePerf,2)) + ' %' + '\n'
		if len(allPerf)>1:
			perf_print += '\nof which best performance is:' + '\n'
			perf_print += str(np.round(100*(np.max(allPerf)),2)) + '%' + ' (run ' + str(np.argmax(allPerf)) + ')' + '\n'
			perf_print += 'and worse performance is:' + '\n'
			perf_print += str(np.round(100*(np.min(allPerf)),2)) + '%' + ' (run ' + str(np.argmin(allPerf)) + ')' + '\n'

		print perf_print

	if save_data:
		pFile = open('output/' + name + '/classResults', 'w')
		pDict = {'allCMs':allCMs, 'avgCM':avgCM, 'steCM':steCM, 'allPerf':allPerf, 'avgPerf':avgPerf, 'stePerf':stePerf}
		pickle.dump(pDict, pFile)
		pFile.close()

		perf_file = open('./output/' + name + '/' +name+ '_perf.txt', 'w')
		perf_file.write(perf_print)
		perf_file.close()

		fig = pl.plot_CM(avgCM, classes)
		pyplot.savefig('./output/' + name + '/' +name+ '_avgCM.pdf')
		pyplot.close(fig)















