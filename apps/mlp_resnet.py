import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
            nn.Linear(dim, hidden_dim),
            norm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, dim),
            norm(dim),
            )
        ),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    loss_func = nn.SoftmaxLoss()
    total_loss = 0
    total_error_rate = 0
    counter = 0
    for (X0,y0) in dataloader:
        counter += 1
        if opt:
            opt.reset_grad()
        out = model(X0)
        loss = loss_func(out,y0)
        loss.backward()
        total_loss += loss.data.numpy()
        total_error_rate += (out.numpy().argmax(axis=1) != y0.numpy()).mean()
        if opt:
            opt.step()
    return (total_error_rate / counter,total_loss / counter)
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
            os.path.join(data_dir,"train-images-idx3-ubyte.gz"),
            os.path.join(data_dir,"train-labels-idx1-ubyte.gz"),
            # transforms = [ndl.data.RandomCrop(12), ndl.data.RandomFlipHorizontal(0.4)]
        )
    test_dataset = ndl.data.MNISTDataset(os.path.join(data_dir,"t10k-images-idx3-ubyte.gz"),os.path.join(data_dir,"t10k-labels-idx1-ubyte.gz"))
    train_loader = ndl.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_dataset,batch_size=batch_size, shuffle=False)
    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_error, train_loss = epoch(train_loader, model, opt)
    test_error, test_loss = epoch(test_loader, model)
    return (train_error, train_loss, test_error, test_loss)
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
