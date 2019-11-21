import sys,time
import numpy as np
import torch
from copy import deepcopy

import utils


class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, epochs=200, batch=64, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=5, lamb=0.1, lamb_ewc=5000, writer=None, exp_name=None, device='cuda',
                 args=None):
        self.model = model
        self.model_old = None
        self.fisher = None

        self.writer = writer
        self.device = device
        self.exp_name = exp_name

        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.lamb = lamb
        self.lamb_ewc = lamb_ewc
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = None

        if args.mode == 'search':
            self.epochs = args.epochs
            self.batch = args.batch
            self.lr = args.lr
            self.lamb = args.lamb
            self.lamb_ewc = args.lamb_ewc  # special in ewc
            self.lr_patience = args.lr_patience
            self.lr_factor = args.lr_factor

        return

    def _get_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr
        return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.lamb,
                               momentum=0.9)

    def train(self, t, train_data, valid_data, device='cuda'):
        self.writer.add_text("ModelSize/Task_{}".format(t),
                             "model size = {}".format(utils.get_model_size(self.model)))
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        # 1 define the optimizer and scheduler
        self.optimizer = self._get_optimizer(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=self.batch, shuffle=False, num_workers=4, pin_memory=True)
        # 3 training the model
        for e in range(self.epochs):
            # 3.1 train
            self.train_epoch(t, train_loader, device=device)
            # 3.2 compute training loss
            train_loss, train_acc = self.eval(t, train_loader, mode='train', device=device)
            # 3.3 compute valid loss
            valid_loss, valid_acc = self.eval(t, valid_loader, mode='train', device=device)
            # 3.4 logging
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | Valid: loss={:.3f}, acc={:5.1f}% |'.format(
                e, train_loss, 100 * train_acc, valid_loss, 100 * valid_acc))
            self.writer.add_scalars('Train_Loss/Task: {}'.format(t),
                                    {'train_loss': train_loss, 'valid_loss': valid_loss},
                                    global_step=e)
            self.writer.add_scalars('Train_Accuracy/Task: {}'.format(t),
                                    {'train_acc': train_acc * 100, 'valid_acc': valid_acc * 100},
                                    global_step=e)
            # 3.5 Adapt learning rate
            scheduler.step()
            # 3.6 update the best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)

        # 4 Restore best model
        utils.set_model_(self.model, best_model)

        # Update old
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old)  # Freeze the weights

        # Fisher ops
        if t > 0:
            fisher_old = {}
            for n, _ in self.model.named_parameters():
                fisher_old[n] = self.fisher[n].clone()
        self.fisher = utils.fisher_matrix_diag(t, train_loader, self.model,
                                               self.criterion, device, self.batch)
        if t > 0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory,
            # therefore we have to merge fisher diagonals
            for n, _ in self.model.named_parameters():
                self.fisher[n] = (self.fisher[n] + fisher_old[n]*t)/(t+1)
        return

    def train_epoch(self, t, train_loader, device='cuda'):
        self.model.train()
        # Loop batches
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Forward current model
            outputs = self.model.forward(x)
            output = outputs[t]
            loss = self.criterion(t, output, y)
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
                outputs = self.model.forward(x)
                output = outputs[t]
                loss = self.criterion(t, output, y)
                _, pred = output.max(1)
                hits = (pred == y).float()
                # Log
                total_loss += loss.item()*length
                total_acc += hits.sum().item()
                total_num += length

        return total_loss/total_num, total_acc/total_num

    def criterion(self, t, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                loss_reg += torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

        return self.ce(output, targets) + self.lamb_ewc * loss_reg

