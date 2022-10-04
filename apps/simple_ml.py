from re import I
import struct
import gzip
import numpy as np

import sys

from needle.autograd import Tensor
sys.path.append('python/')
import needle as ndl

def read_images(b:bytes):
    """ Read images from a byte array in MNIST format.
        Return a numpy array which shape is (images, rows, columns)
    """
    import struct
    magic, images, rows, columns = struct.unpack_from('>IIII' , b , 0)
    if magic != 0x00000803:
        raise ValueError('Magic number mismatch, expected 2051, got %d' % magic)
    if images < 0:
        raise ValueError('Invalid number of images: %d' % images)
    base = struct.calcsize('>IIII')
    format = '>' + str(rows * columns) + 'B'
    offset = struct.calcsize(format)
    # now we get a numpy array whose shape is (images, rows, columns)
    result_array = np.array([ struct.unpack_from(format,b,base+i*offset) for i in range(images)],dtype="float32")
    return result_array
def read_labels(b:bytes):
    """ Read labels from a byte array in MNIST format.
        Return a numpy array which shape is (labels, rows, columns)
    """
    import struct
    magic, labels = struct.unpack_from('>II' , b , 0)
    if magic != 0x00000801:
        raise ValueError('Magic number mismatch, expected 2049, got %d' % magic)
    if labels < 0:
        raise ValueError('Invalid number of images: %d' % labels)
    base = struct.calcsize('>II')
    format = '>' +'B'
    offset = struct.calcsize(format)
    # now we get a numpy array which shape is (images, rows, columns)
    result_array = np.array([ struct.unpack_from(format,b,base+i*offset) for i in range(labels)],dtype=np.uint8).reshape((labels,))
    return result_array
def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname,'rb') as f:
        b = f.read()
        X = read_images(b)
        # normalize
        X = (X - X.min()) / (X.max() - X.min())
    with gzip.open(label_filename,'rb') as f:
        b = f.read()
        y = read_labels(b)
    return X, y
    ### END YOUR SOLUTION

def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    y = (Z * y_one_hot).sum(axes=(1,))
    n = Z.shape[0]
    return (ndl.log(ndl.exp(Z).sum(axes=(1,))) - y).sum() / n
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    for i in range(0, len(X), batch):
        # get the batch
        batch_x = Tensor(X[i:i+batch],dtype=X.dtype,requires_grad=True)
        batch_y = y[i:i+batch]
        m = batch_x.shape[0]
        k = W2.shape[1]
        Z1 = ndl.relu(batch_x @ W1)
        Iy = np.zeros((m,k))
        Iy[np.arange(len(Iy)), batch_y] = 1
        Iy = Tensor(Iy,dtype=batch_y.dtype,requires_grad=True)
        f = softmax_loss(Z1 @ W2,Iy)
        f.backward()
        W2 -= lr * (W2.grad.data)
        W1 -= lr * (W1.grad.data)
    ### END YOUR SOLUTION
    return W1, W2

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
