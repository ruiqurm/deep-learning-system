from cProfile import label
import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE

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
def parse_mnist(image_filename, label_filename):
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
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename,'rb') as f:
        b = f.read()
        X = read_images(b)
        # normalize
        X = (X - X.min()) / (X.max() - X.min())
    with gzip.open(label_filename,'rb') as f:
        b = f.read()
        y = read_labels(b)
    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # take z_y by index
    zy = Z[np.arange(len(Z)), y]

    # use the formula to compute loss
    return (np.log(np.exp(Z).sum(axis=1)) - zy).mean()
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    for i in range(0, len(X), batch):
        # get the batch
        batch_x = X[i:i+batch]
        batch_y = y[i:i+batch]
        # compute the gradient
        m = batch_x.shape[0]
        k = theta.shape[1]
        tmp_Z = np.exp(batch_x.dot(theta))
        Z = tmp_Z / tmp_Z.sum(axis=1).reshape((-1,1))
        I_y = np.zeros((m,k))
        I_y[np.arange(len(I_y)), batch_y] = 1
        diff_softmax = (batch_x.T @ (Z - I_y)) / m

        theta -= lr * diff_softmax
    
    ### END YOUR CODE

def relu(Z):
    """ Compute the ReLU activation function on the input.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.

    Returns:
        np.ndarray[np.float32]: 2D numpy array of shape
            (batch_size, num_classes), containing the ReLU activation of the
            input.
    """
    return np.maximum(Z, 0)

def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    for i in range(0, len(X), batch):
        # get the batch
        batch_x = X[i:i+batch]
        batch_y = y[i:i+batch]
        m = batch_x.shape[0]
        k = W2.shape[1]
        Z1 = relu(batch_x @ W1)
        Iy = np.zeros((m,k))
        Iy[np.arange(len(Iy)), batch_y] = 1
        G2 = np.exp(Z1 @ W2)
        G2 = G2 / G2.sum(axis=1).reshape((-1,1)) - Iy
        G1 = ((Z1>0)+0) * (G2 @ W2.T)
        W2 -= lr * (Z1.T @ G2) / m
        W1 -= lr * (batch_x.T @ G1) / m
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)