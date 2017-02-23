import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.model_selection import GridSearchCV

method = 'rnn' #'mlp', 'svm', 'knn', 'rnn'
grid_search = True

n_runs = 10
n_epi = 1 #50 #(x20)
seed =  974

np.random.seed(seed) 

def shuffle_data(X, y):
	idx_shuffle = np.arange(len(y))
	np.random.shuffle(idx_shuffle)
	X = X[idx_shuffle]
	y = y[idx_shuffle]
	X_train, X_test = X[:60000], X[60000:]
	y_train, y_test = y[:60000], y[60000:]
	return X_train, y_train, X_test, y_test

""" import, rescale, shuffle, and split MNIST data """
mnist = fetch_mldata("MNIST original")
X, y = mnist.data / 255., mnist.target
X_train, y_train, X_test, y_test = shuffle_data(X, y)

""" train MLP """

if method == 'mlp':
	if grid_search:
		pass
		mlp = MLPClassifier(hidden_layer_sizes=(300,), activation='relu', algorithm='adam', alpha=1e-6, batch_size='auto', learning_rate_init=1e-3, max_iter=100, shuffle=True, random_state=None, tol=1e-4, verbose=True, beta_1=0.8, beta_2=0.9, epsilon=1e-08)

		# params_gs = {'activation': ['relu', 'tanh', 'logistic']} 
			#best: 'activation': 'relu'; perf: 0.97805
		# params_gs = {'algorithm': ['adam'], 'learning_rate_init':[10.0 ** (-i-3) for i in range(4)], 'beta_1':[0.5, 0.7 0.8], 'beta_2':[0.5, 0.8, 0.9]}
			#best: 'beta_1': 0.8,'beta_2': 0.9, 'learning_rate_init': 0.001; perf:0.97893333333333332
		# params_gs = {'algorithm': ['sgd'], 'learning_rate':['adaptive'], 'learning_rate_init':[10.0 ** -(i) for i in range(4)], 'momentum':[0.8, 0.9, 0.99], 'nesterovs_momentum':[True]}
			#best: 'learning_rate_init': 0.1, 'momentum': 0.9; perf: 0.97831666666666661
		# params_gs = {'alpha': [10.0 ** -(i+1) for i in range(6)]}
			#best: 'alpha': 1e-06; perf: 0.97841667
		gs = GridSearchCV(mlp, params_gs, n_jobs=10, verbose=2, refit=False)
		gs.fit(X_train, y_train)

	else:
		train_scores_mlp = np.zeros((n_runs, n_epi))
		test_scores_mlp = np.zeros((n_runs, n_epi))
		for r in range(n_runs):
			print "run %d" % r
			#SGD
			# X_train, y_train, X_test, y_test = shuffle_data(X, y)
			# mlp = MLPClassifier(hidden_layer_sizes=(10000,), activation='relu', algorithm='sgd', alpha=0, batch_size=50, learning_rate='constant', max_iter=10, tol=0, learning_rate_init=5e-3, momentum=0, early_stopping=False, warm_start=True, shuffle=True, random_state=seed+r, verbose=True)
			
			#Adam
			# mlp = MLPClassifier(hidden_layer_sizes=(300,), activation='relu', algorithm='adam', alpha=1e-6, batch_size='auto', learning_rate_init=1e-3, max_iter=10, shuffle=True, random_state=seed+r, tol=-100, verbose=True, beta_1=0.8, beta_2=0.9, epsilon=1e-08, warm_start=True)
			for e in range(n_epi):
				#best
				# mlp = MLPClassifier(hidden_layer_sizes=(25,), activation='relu', algorithm='adam', alpha=1e-6, batch_size='auto', learning_rate_init=1e-3, max_iter=100, shuffle=True, random_state=None, tol=1e-4, verbose=True, beta_1=0.8, beta_2=0.9, epsilon=1e-08)
				
				mlp = MLPClassifier(hidden_layer_sizes=(300,), activation='relu', algorithm='sgd', alpha=1e-06, batch_size='auto', learning_rate='adaptive', max_iter=200, tol=1e-4, learning_rate_init=0.1, momentum=0.9, early_stopping=False, nesterovs_momentum=True, warm_start=False, shuffle=True, random_state=seed+r, verbose=True)

				#overfit
				# mlp = MLPClassifier(hidden_layer_sizes=(1000,), activation='relu', algorithm='sgd', alpha=0, batch_size=50, learning_rate='constant', max_iter=1000, tol=1e-4, learning_rate_init=1e-3, momentum=0, early_stopping=False, shuffle=True, random_state=None, verbose=True)
				# mlp = MLPClassifier(hidden_layer_sizes=(300,), activation='relu', algorithm='l-bfgs', alpha=1e-6, max_iter=100, shuffle=True, random_state=None, tol=1e-4, verbose=True)
				# X_train, y_train, X_test, y_test = shuffle_data(X, y)

				mlp.fit(X_train, y_train)
				train_scores_mlp[r, e] = mlp.score(X_train, y_train)
				test_scores_mlp[r, e] = mlp.score(X_test, y_test)
				print "epi %d finished, train score %.3f, test score %.3f" %(e, train_scores_mlp[r, e], test_scores_mlp[r, e])
			print "run %d finished, train score %.3f, test score %.3f" %(r, train_scores_mlp[r, -1], test_scores_mlp[r, -1])
		print "\nall runs finished; mean score: %.3f +/- %.3f" %(np.mean(test_scores_mlp[:,-1]), np.std(test_scores_mlp[:,-1]))

""" train SVM """
if method == 'svm':
	if grid_search:
		pass
		svm = SVC(C=10.0, kernel='rbf', gamma=0.01, tol=1e-3, verbose=True, random_state=None)

		# params_gs = {'C': [10.0 ** (i-2) for i in range(5)], 'gamma': [10.0 ** (i-5) for i in range(5)]}
			#best: 'C': 10.0, 'gamma': 0.01; perf: 0.98078333333333334
		gs = GridSearchCV(svm, params_gs, n_jobs=20, verbose=2, refit=True)
		gs.fit(X_train, y_train)
	else:
		all_scores_svm = np.zeros(n_runs)
		for r in range(n_runs):
			print "run %d" % r
			svm = SVC(C=10.0, kernel='rbf', gamma=0.01, tol=1e-3, verbose=True, random_state=None)
			X_train, y_train, X_test, y_test = shuffle_data(X, y)
			svm.fit(X_train, y_train)
			svm_score = svm.score(X_test, y_test)
			all_scores_svm[r] = svm_score
			print "run %d finished, score %.3f" %(r, svm_score)
		print "\nall runs finished; mean score: %.3f +/- %.3f" %(np.mean(all_scores_svm), np.std(all_scores_svm))

""" train K-NN """

if method == 'knn':
	if grid_search:
		knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
		params_gs = {'n_neighbors': [5,10,20,50,100,200], 'weights': ['uniform', 'distance']}
		gs = GridSearchCV(knn, params_gs, n_jobs=12, verbose=2, refit=False)
		gs.fit(X_train, y_train)

		###results
		# [mean: 0.96867, std: 0.00105, params: {'n_neighbors': 5, 'weights': 'uniform'},
		#  mean: 0.96983, std: 0.00103, params: {'n_neighbors': 5, 'weights': 'distance'},
		#  mean: 0.96445, std: 0.00109, params: {'n_neighbors': 10, 'weights': 'uniform'},
		#  mean: 0.96688, std: 0.00135, params: {'n_neighbors': 10, 'weights': 'distance'},
		#  mean: 0.95835, std: 0.00107, params: {'n_neighbors': 20, 'weights': 'uniform'},
		#  mean: 0.95990, std: 0.00138, params: {'n_neighbors': 20, 'weights': 'distance'},
		#  mean: 0.94523, std: 0.00044, params: {'n_neighbors': 50, 'weights': 'uniform'},
		#  mean: 0.94672, std: 0.00027, params: {'n_neighbors': 50, 'weights': 'distance'},
		#  mean: 0.93248, std: 0.00118, params: {'n_neighbors': 100, 'weights': 'uniform'},
		#  mean: 0.93417, std: 0.00109, params: {'n_neighbors': 100, 'weights': 'distance'},
		#  mean: 0.91512, std: 0.00160, params: {'n_neighbors': 200, 'weights': 'uniform'},
		#  mean: 0.91738, std: 0.00156, params: {'n_neighbors': 200, 'weights': 'distance'}]
		###
		
""" train r-NN """ 

if method == 'rnn':
	if grid_search:
		rnn = RadiusNeighborsClassifier(radius=1.0, weights='uniform')
		params_gs = {'radius': [0.5, 1.0, 2.0, 10.0, 30.0, 100.0], 'weights': ['uniform', 'distance']}
		gs = GridSearchCV(rnn, params_gs, n_jobs=12, verbose=2, refit=False)
		gs.fit(X_train, y_train)








































