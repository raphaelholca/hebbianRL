import pickle
import sys
import support.plots as pl
pl = reload(pl)


pfile = open('output/'+runName+'/classResults', 'r')
classResults = pickle.load(pfile)
pfile.close()