import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

N_EPOCHS = 300


# linear regression model
class LogisticNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(LogisticNet, self).__init__()
        self.linear = nn.Linear(D_in, D_out)

    def forward(self, x):
        lin = self.linear(x)
        return lin


def train(model, loss_func, optimizer, trX, trY):
    x = Variable(trX, requires_grad=False)
    y = Variable(trY, requires_grad=False)
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_func(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.data[0]


def predict(model, x_val):
    output = model.forward(x_val)
    return output.data.numpy().argmax(axis=1)


def valid(model, loss_func, valX, valY):
    x = Variable(valX, requires_grad=False)
    y = Variable(valY, requires_grad=False)

    outputs = model(x)
    val_loss = loss_func(outputs, y)
    # calculate accuracy
    _, predY = torch.max(outputs.data, 1)
    correct = (predY == y.data).sum()
    val_acc = float(correct) / y.size(0)
    return val_loss.data[0], val_acc


def main():
    digits = load_digits()
    data = digits['data']
    target = digits['target']
    # separate data
    trX, teX, trY, teY = train_test_split(data, target, test_size=0.2, random_state=0)

    n_samples = trX.shape[0]
    input_dim = trX.shape[1]
    n_classes = 10
    model = LogisticNet(input_dim, n_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    trX = torch.from_numpy(trX).float()
    teX = torch.from_numpy(teX).float()
    trY = torch.from_numpy(trY.astype(np.int64))
    teY = torch.from_numpy(teY.astype(np.int64))

    for epoch in range(N_EPOCHS):
        loss = train(model, loss_func, optimizer, trX, trY)
        val_loss, val_acc = valid(model, loss_func, teX, teY)
        print 'val loss:%.3f val acc:%.3f' % (val_loss, val_acc)


if __name__ == '__main__':
    main()
