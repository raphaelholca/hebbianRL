#!/usr/bin/env python

# to be used for Timm's classification challenge as: 
# import support.svmutils as su
# su = reload(su)
# labels, images = su.svm_read_problem('digits/digit_train')

import numpy as np

def svm_read_problem(data_file_name):
	"""
	svm_read_problem(data_file_name) -> [y, x]
	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x.
	"""
	prob_y = []
	prob_x = []
	for line in open(data_file_name):
		line = line.split(None, 1)
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		label, features = line
		xi = {}
		for e in features.split():
			ind, val = e.split(":")
			xi[int(ind)] = float(val)
		prob_y += [float(label)]
		prob_x += [xi]

	images = np.zeros((len(prob_x), 784))

	for i in range(len(prob_x)):
		for k in prob_x[i].keys():
			images[i,k] = prob_x[i][k]

	labels = np.array(map(int, prob_y))

	return (labels, images)
