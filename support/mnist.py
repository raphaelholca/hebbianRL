import numpy as np
import os
import struct
from array import array

def read_images_from_mnist(classes, dataset = "train",
                           path = "./data-sets/MNIST"):
    """
    Python function for importing the MNIST data set.
    """

    if dataset is "train":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "test":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'test' or 'train'"

    flbl = open(fname_lbl, 'rb')
    #magic_nr, size = struct.unpack(">II", flbl.read(8))
    struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    #lbl = np.array(flbl.read(),dtype='b')
    flbl.close()

    fimg = open(fname_img, 'rb')
    #magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    size, rows, cols = struct.unpack(">IIII", fimg.read(16))[1:4]
    img = array("B", fimg.read())
    #img = np.array(fimg.read(),dtype='B')
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in classes ]
    images = np.zeros(shape=(len(ind), rows*cols))
    labels = np.zeros(shape=(len(ind)), dtype=int)
    for i in xrange(len(ind)):
        images[i, :] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
        labels[i] = lbl[ind[i]]

    return images, labels