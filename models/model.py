import torch.nn as nn
import torch
import os
import time
from progress.bar import Bar
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from ml.misc import AverageMeter
from torch.autograd import Variable

fig_dir = "ml/plot"


class Model(nn.Module):
    def __init__(self, input_dimension):
        super(Model, self).__init__()
        self.dimension = input_dimension
        hidden_dimension = int(input_dimension / 2)
        self.layer1 = nn.Linear(input_dimension, hidden_dimension)
        self.layer2 = nn.Linear(hidden_dimension, 1)

    def forward(self, x):
        x = self.layer1(x)
        # x = torch.sigmoid(x)
        x = self.layer2(x)
        return x

    def train1(self, train_loader, val_loader, lr, weight_decay, momentum, max_epochs, batch_size, plot=True, save=True):
        loss_function = torch.nn.L1Loss()
        optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

        train_loss, val_loss = [], []
        for epoch in range(1, max_epochs+1):
            epoch_train_loss, epoch_val_loss = [], []
            # training part
            self.train()
            for data, target in train_loader:
                optimizer.zero_grad()
                # forward propagation
                output = self(data)

                # loss calculation
                loss = loss_function(output, target.double())

                # backward propagation
                loss.backward()

                # weight optimization
                optimizer.step()

                epoch_train_loss.append(loss.item())

            # evaluation part
            self.eval()
            for data, target in val_loader:
                output = self(data)
                loss = loss_function(output, target.double())
                epoch_val_loss.append(loss.item())
            train_loss.append(np.mean(epoch_train_loss))
            val_loss.append(np.mean(epoch_val_loss))
            print("Epoch:", epoch, "Training Loss: ", np.mean(epoch_train_loss), "Validation Loss: ", np.mean(epoch_val_loss))

        def plot_loss(train_loss, val_loss, save=True):
            fig, (axis1, axis2) = plt.subplots(2, 1, num="Loss Plots", figsize=(7, 10))
            axis1.plot(train_loss)
            axis1.title.set_text('Train Loss')
            axis1.set_xlabel('Epochs')
            axis1.set_ylabel('Loss')
            start, end, step = plot_tick_helper(train_loss)
            axis1.set_yticks(np.arange(start, end, step))
            axis1.set_xticks(np.linspace(0, max_epochs, 11))
            axis2.plot(val_loss)
            axis2.title.set_text('Validation Loss')
            axis2.set_xlabel('Epochs')
            axis2.set_ylabel('Loss')
            start, end, step = plot_tick_helper(val_loss)
            axis2.set_yticks(np.arange(start, end, step))
            axis2.set_xticks(np.linspace(0, max_epochs, 11))
            plt.subplots_adjust(hspace = 0.3)
            if save:
                if not os.path.isdir(fig_dir):
                    os.mkdir(fig_dir)
                plt.savefig(f'{fig_dir}/lr:{lr}_weight_decay:{weight_decay}_momentum:{momentum}_max_epochs:{max_epochs}_batch_size:{batch_size}.png')
                plt.clf()
            else:
                plt.show()

        if plot:
            plot_loss(train_loss, val_loss, save=save)


def plot_tick_helper(loss):
    start = int(min(loss)) - 1
    end = int(max(loss)) + 1
    if (end - start) > 100:
        step = 10
    elif (end - start) > 50:
        step = 5
    elif (end - start) > 20:
        step = 2
    else:
        step = 1
    return start, end, step

# def train_helper(batchloader, model, criterion, optimizer=None, penalty=None, test=False, use_cuda=False):
#     # switch to train or evaluate mode
#     if test:
#         model.eval()
#     else:
#         model.train()
#
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     end = time.time()
#
#     if test:
#         bar = Bar('Testing', max=len(batchloader))
#     else:
#         bar = Bar('Training', max=len(batchloader))
#
#     print(batchloader)
#     for batch_idx, (inputs, targets) in enumerate(batchloader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         if use_cuda:
#             inputs = inputs.cuda()
#             targets = targets.cuda()
#
#         inputs = Variable(inputs)
#         targets = Variable(targets)
#
#         # compute output
#         outputs = model(inputs)
#
#         # calculate loss
#         loss = 0
#         loss = loss + criterion(outputs, targets.double())
#         if penalty is not None:
#             loss = loss + penalty(model)
#
#         # record loss
#         losses.update(loss.data.item(), inputs.size(0))
#
#         if not test:
#             # compute gradient and do SGD step
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         # plot progress
#         bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f}'.format(
#             batch=batch_idx + 1,
#             size=len(batchloader),
#             data=data_time.avg,
#             bt=batch_time.avg,
#             total=bar.elapsed_td,
#             loss=losses.avg)
#         print(bar.suffix)
#         bar.next()
#
#     bar.finish()
#     return losses.avg
#
#
# def train2(dataloader, model, data, labels, lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=False, max_epochs=10):
#
#     loss_function = torch.nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
#
#     for i in range(max_epochs):
#         print("EPOCH: {0}".format(i + 1))
#         train_helper(dataloader, model, loss_function, optimizer=optimizer)
#
#     while True:
#         idx = int(input())
#         print("TARGET: {0}".format(labels[idx]))
#         value = torch.from_numpy(data[idx])
#         print("ANSWER: {0}".format(model.forward(value).item()))
