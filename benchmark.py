import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

train_mlp = True
train_svm = False
train_kmeans = False
grid_search = True

""" import, rescale, shuffle, and split MNIST data """
mnist = fetch_mldata("MNIST original")
X, y = mnist.data / 255., mnist.target
idx_shuffle = np.arange(len(y))
np.random.shuffle(idx_shuffle)
X = X[idx_shuffle]
y = y[idx_shuffle]
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

""" train MLP """

if train_mlp:
	mlp = MLPClassifier(hidden_layer_sizes=(300,), activation='relu', algorithm='adam', alpha=0.0001, batch_size='auto', learning_rate_init=0.001, max_iter=10, shuffle=True, random_state=None, tol=1e-4, verbose=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	
	if grid_search:
		params_gs = {'activation': ['relu', 'tanh', 'logistic']}
		# params_gs = [{'algorithm': ['adam'], 'learning_rate_init':[10.0 ** -(i) for i in range(4)], 'beta_1'[0.8, 0.9, 0.99]:, 'beta_2':[0.9, 0.99, 0.999]}, {'algorithm': ['sgd'], 'learning_rate':['adaptive'], 'learning_rate_init':[10.0 ** -(i) for i in range(4)], 'momentum':[0.8, 0.9, 0.99], 'nesterovs_momentum':True}]
		# params_gs = {'alpha': [10.0 ** -(i+1) for i in range(6)]}
		
		gs = GridSearchCV(mlp, params_gs, n_jobs=3, verbose=2)
		gs.fit(X_train, y_train)

	else:
		mlp.fit(X_train, y_train)
		mlp_score = mlp.score(X_test, y_test)

		print mlp_score

""" train SVM """
if train_svm:
	svm = SVC(C=1.0, kernel='rbf', gamma=0.001, tol=1e-3, verbose=True, random_state=None)
	
	if grid_search:
		params_gs = {'C': [10.0 ** (i-2) for i in range(5)], 'gamma': [10.0 ** (i-5) for i in range(5)]}
		gs = GridSearchCV(svm, params_gs, n_jobs=3, verbose=2)
		gs.fit(X_train, y_train)

	else:
		svm.fit(X_train, y_train)
		svm_score = svm.score(X_test, y_test)

		print svm_score

""" train K-means """

if train_kmeans:
	pass










































