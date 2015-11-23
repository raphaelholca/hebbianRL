import numpy as np
import external as ex
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsRegressor

ex = reload(ex)

def pdf_estimate(images, labels, W, kwargs, method='fit'):
	"""
	Uses kernel density extimation to the compute the pdf of neural activation data.

	Args:
		images (numpy array): input images
		labels (numpy array): input labels associated with the neuron activations
		W (numpy array): weights of the hidden neurons
		kwargs (dict): parameters of the model
		method (str, optional): method used to approximate the pdf; valid: 'fit' (fits the pdf with a regressor), 'subsample' (computes the pdf with a smaller data sample), 'full' (uses full dataset, slower)

	returns:
		(list of regressor or kde objects): list of marginal pdfs
		(regressor or kde object): pdf
		(numpy array): labels of the data points used to compute the pdf (useful to compute prior)
	"""

	n_classes = len(np.unique(labels))
	n_trials = len(labels)

	""" computes the activation of the hidden neurons for the given input images """
	activ = ex.propL1(images, W, t=kwargs['t_hid'])

	n_subsample = 1000 #number of data points to use to compute the pdf in the 'subsample' and 'fit' methods
	subsample_idx = np.random.choice(n_trials, size=n_subsample, replace=False)
	activ_subs = activ[subsample_idx, :]

	n_train_fit = 500 #number of data point to use to fit the pdf in the 'fit' method
	train_fit_idx = np.random.choice(n_trials, size=n_train_fit, replace=False)
	activ_fit = activ[train_fit_idx, :]

	if method=='full':
		pdf_labels = np.copy(labels)
		pdf_evidence = KernelDensity(bandwidth=5e-1, kernel='gaussian', rtol=1e-100).fit(activ)
		pdf_marginals = []
		for i in range(n_classes):
			pdf_marginals.append(KernelDensity(bandwidth=5e-1, kernel='gaussian', rtol=1e-100).fit(activ[pdf_labels==i]))

	if method=='subsample':
		pdf_labels = labels[subsample_idx]
		pdf_evidence = KernelDensity(bandwidth=5e-1, kernel='gaussian', rtol=1e-100).fit(activ_subs)
		pdf_marginals = []
		for i in range(n_classes):
			pdf_marginals.append(KernelDensity(bandwidth=5e-1, kernel='gaussian', rtol=1e-100).fit(activ_subs[pdf_labels==i]))

	if method=='fit':
		pdf_labels = labels[subsample_idx]
		pdf_evidence_full = KernelDensity(bandwidth=5e-1, kernel='gaussian', rtol=1e-100).fit(activ_subs)
		pdf_evidence = KNeighborsRegressor().fit(activ_fit, pdf_evidence_full.score_samples(activ_fit))
		pdf_marginals = []
		for i in range(n_classes):
			pdf_marginal_full = KernelDensity(bandwidth=5e-1, kernel='gaussian', rtol=1e-100).fit(activ_subs[pdf_labels==i])
			pdf_marginals.append(KNeighborsRegressor().fit(activ_fit, pdf_marginal_full.score_samples(activ_fit)))

	return pdf_marginals, pdf_evidence, pdf_labels

def bayesian_decoder(activ, pdf_marginals, pdf_evidence, pdf_labels, method):
	"""
	Computes the posterior probability of the input classes for a given population neural activity vector

	Args:
		activ (numpy array): activation of hidden neurons to decode (n_trials x n_neurons)
		pdf_marginals (list of regressor or kde objects): list of marginal pdfs
		pdf_evidence (regressor or kde object): pdf of neural activation
		pdf_labels (numpy array): labels of the data used to compute the pdf
		method (str): method used to approximate the pdf in pdf_estimate()
	"""

	classes = np.unique(pdf_labels)
	n_classes = len(classes)
	n_activ = np.size(activ,0)
	n_labels = float(len(pdf_labels))

	""" computes the prior distribution for all input classes """
	priors = np.zeros(n_classes)
	for i, c in enumerate(classes):
		priors[i] = np.sum(pdf_labels==c)/n_labels

	""" computes the evidence """
	if method=='fit':
		evidence = np.exp(pdf_evidence.predict(activ))
	else:
		evidence = np.exp(pdf_evidence.score_samples(activ))

	""" computes the marginals (conditional probability) for all input classes  """
	marginals = np.zeros((n_activ, n_classes))
	for i, pdf_m in enumerate(pdf_marginals):
		if method=='fit':
			marginals[:, i] = np.exp(pdf_m.predict(activ))
		else:
			marginals[:, i] = np.exp(pdf_m.score_samples(activ))
	
	""" computes the posterior probability """
	posterior = (marginals*priors)/evidence[:, np.newaxis]
	
	return posterior
































