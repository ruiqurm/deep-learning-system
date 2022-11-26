import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.flip(img, axis=1)
        else:
            return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        assert len(img.shape) == 3 
        h,w,_ = img.shape
        if shift_x > 0:
            x_padding,x_range = (0,shift_x),(shift_x,h+shift_x)
        else:
            x_padding,x_range =  (-shift_x,0),(0,h)
        if shift_y > 0:
            y_padding,y_range = (0,shift_y),(shift_y,w+shift_y)
        else:
            y_padding,y_range =  (-shift_y,0),(0,w)
        new_img = np.pad(img, (x_padding,y_padding,(0,0)))
        return new_img[x_range[0]:x_range[1], y_range[0]:y_range[1], :]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x

### MY IMPL
def split_range(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
### END MY IMPL

class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            # self.ordering = np.array_split(
            #     np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            # )
            n = len(dataset)
            strdie = batch_size
            self.ordering = [slice(i,min(n,i+strdie)) for i in range(0,n,strdie)]

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.pointer = 0
        if self.shuffle:
            tmp_range = np.arange(len(self.dataset))
            np.random.shuffle(tmp_range)
            n = len(self.dataset)
            strdie = self.batch_size
            self.ordering = [slice(i,min(n,i+strdie)) for i in range(0,n,strdie)]
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.pointer >= len(self.ordering):
            raise StopIteration
        else:
            result = tuple([Tensor(obj) for obj in self.dataset[self.ordering[self.pointer]]])
            # result = (Tensor(data),Tensor(label))
        self.pointer += 1 
        return result
        ### END YOUR SOLUTION

import gzip
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

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename,'rb') as f:
            b = f.read()
            X = read_images(b)
            X = (X - X.min()) / (X.max() - X.min())
            self.images = X
        with gzip.open(label_filename,'rb') as f:
            b = f.read()
            self.labels = read_labels(b)
        assert len(self.images) == len(self.labels) 
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        images = self.images[index].reshape((28,28,-1))
        if self.transforms is not None:
            for transform in self.transforms:
                images = transform(images)
        return images.reshape((-1,784,)), self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION

from functools import lru_cache
class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        import pickle
        if train:
            for _,_,filenames in os.walk(base_folder):
                for filename in sorted(filenames):
                    if filename.startswith("data_batch"):
                        with open(os.path.join(base_folder,filename),'rb') as f:
                            data = pickle.load(f,encoding='bytes')
                            if 'X' in locals():
                                X = np.concatenate((X,data[b'data']),axis=0)
                                y = np.concatenate((y,data[b'labels']),axis=0)
                            else:
                                X = data[b'data']
                                y = data[b'labels']
        else:
            with open(os.path.join(base_folder,"test_batch"),'rb') as f:
                data = pickle.load(f,encoding='bytes')
                X = data[b'data']
                y = data[b'labels']
        X = X.astype('float32')
        X /= 255
        assert len(X) == len(y)
        self.X = X
        self.y = y
        ### END YOUR SOLUTION
    def __getitem__(self, index: Union[int,slice,tuple]) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if isinstance(index,int):
            X,y = self.X[index].reshape((3,32,32)),self.y[index]
        elif isinstance(index,slice):
            X,y = self.X[index].reshape((-1,3,32,32)),self.y[index]
        elif isinstance(index,tuple):
            X,y = self.X[index,].reshape((-1,3,32,32)),self.y[index,]
        else:
            raise
        if hasattr(self,'transforms'):
            for transform in self.transforms:
                X = transform(X)
        return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION