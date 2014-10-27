import numpy as np
import pickle
import sys
from configobj import ConfigObj
import matplotlib.pyplot as pyplot
import support.plots as pl
pl = reload(pl)

runName = 'all_reward_5_rLow1_3'
nClasses = 10

#load classifier from file
print "importing data..."
pfile = open('support/SVM-MNIST-proba', 'r')
svm_mnist = pickle.load(pfile)
pfile.close()

#parameters of the model from:
#http://peekaboo-vision.blogspot.co.uk/2010/09/mnist-for-ever.html
#svm_mnist = SVC(kernel="rbf", C=2.8, gamma=.0073, probability=True, verbose=True)

#load weight file
pfile = open('output/'+runName+'/W_in', 'r')
W = pickle.load(pfile)
pfile.close()

settings = {}
settings = ConfigObj('output/' + runName + '/settings.txt')
classes  = np.array(map(int, settings['classes']))

nRun = len(W.keys())

RFproba = []
perf = np.zeros((nRun,nClasses))
RFsharp = np.zeros((nRun,nClasses))
RFclass = np.zeros((nRun,nClasses))
for i,r in enumerate(sorted(W.keys())):
	print 'run: ' + str(i)
	RFproba.append(np.round(svm_mnist.predict_proba(W[r].T),2))
	RFclass[i,:], _ = np.histogram(np.argmax(RFproba[i],1), bins=nClasses, range=(-0.5,9.5))

RFclass_mean = np.mean(RFclass, 0)
RFclass_ste = np.std(RFclass, 0)/np.sqrt(np.size(RFclass,0))

pRFclass = {'RFclass_all':RFclass, 'RFclass_mean':RFclass_mean, 'RFclass_ste':RFclass_ste}

pfile = open('output/'+runName+'/RFclass', 'w')
pickle.dump(pRFclass, pfile)
pfile.close()

fig = pl.plotHist(RFclass_mean[classes], classes, h_err=RFclass_ste)
pyplot.savefig('output/'+runName+'/RFhist.png')
pyplot.show(block=False)
# pyplot.close(fig)



















