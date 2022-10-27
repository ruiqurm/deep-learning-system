from email.mime import image
import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as a H x W x C NDArray.
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
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
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
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
        # else:
        #     arr = np.arange(len(dataset))
        #     np.random.shuffle(arr)
        #     self.ordering = np.array_split(arr, range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.pointer = 0
        if self.shuffle:
            tmp_range = np.arange(len(self.dataset))
            np.random.shuffle(tmp_range)
            self.ordering = np.array_split(tmp_range,
                range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.pointer >= len(self.ordering):
            raise StopIteration
        else:
            result = [Tensor(obj) for obj in self.dataset[self.ordering[self.pointer]]]
            # result = (Tensor(data),Tensor(label))
        self.pointer += 1 
        return result
        ### END YOUR SOLUTION

### My code
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
###

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

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
