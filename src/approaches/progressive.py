import sys, time
import numpy as np
import torch
from torch import nn

import utils


class Appr(object):

    def __init__(self, model, epochs=50, batch=128, lr=0.025, lamb=3e-4,
                 lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=5,
                 writer=None, exp_name="None", device='cuda', args=None):
        self.device = device
        self.writer = writer
        self.exp_name = exp_name

        self.model = model

        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.lamb = lamb
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        return

    def _get_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr
        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=lr, momentum=0.9, weight_decay=self.lamb)

    def train(self, t, train_data, valid_data, device='cuda'):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr

        # train only the column for the current task
        self.model.unfreeze_column(t)
        # 1 define the optimizer and scheduler
        self.optimizer = self._get_optimizer(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=self.batch, shuffle=False, num_workers=4, pin_memory=True)

        # Loop epochs
        for e in range(self.epochs):
            # Train
            self.train_epoch(t, train_loader, device=device)
            train_loss, train_acc = self.eval(t, train_loader, mode='train', device=device)
            # Valid
            valid_loss, valid_acc = self.eval(t, valid_loader, mode='train', device=device)
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | Valid: loss={:.3f}, acc={:5.1f}% |'.format(
                e, train_loss, 100 * train_acc, valid_loss, 100 * valid_acc))
            self.writer.add_scalars('Train_Loss/Task: {}'.format(t),
                                    {'train_loss': train_loss, 'valid_loss': valid_loss},
                                    global_step=e)
            self.writer.add_scalars('Train_Accuracy/Task: {}'.format(t),
                                    {'train_acc': train_acc * 100, 'valid_acc': valid_acc * 100},
                                    global_step=e)
            # Adapt lr
            scheduler.step()
            # update the best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)

        utils.set_model_(self.model, best_model)

        return

    def train_epoch(self, t, train_loader, device='cuda'):
        self.model.train()
        # Loop batches
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Forward
            outputs = self.model.forward(x, t)
            output = outputs[t]
            loss = self.criterion(output, y)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

        return

    def eval(self, t, test_loader, mode, device='cuda'):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                length = x.size()[0]
                # Forward
                outputs = self.model.forward(x, t)
                output = outputs[t]
                loss = self.criterion(output, y)
                _, pred = output.max(1)
                hits = (pred == y).float()

                # Log
                total_loss += loss.item() * length
                total_acc += hits.sum().item()
                total_num += length

        return total_loss/total_num, total_acc/total_num
