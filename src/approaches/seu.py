import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import utils

from copy import deepcopy

from automl.mdenas_search import AutoSearch
from models.seu_model import Network
from automl.darts_genotypes import mdenas_pmnist, mdenas_mixture


class Appr(object):
    """ Class implementing the new approach """
    def __init__(self, input_size=None, taskcla=None,
                 c_epochs=100, c_batch=512, c_lr=0.025, c_lr_a=0.01, c_lamb=3e-4, c_lr_min=1e-3,
                 o_epochs=100, o_batch=512, o_lr=0.025, o_lr_a=0.01, o_lamb=3e-4, o_size=1, o_lr_min=1e-3,
                 epochs=20, batch=128,  lr=0.025, lamb=3e-4,
                 lr_factor=3, lr_patience=5, clipgrad=5,
                 writer=None, exp_name="None", device='cuda', args=None):
        # the number of cells
        self.model = None
        self.search_layers = args.search_layers
        self.eval_layers = args.eval_layers
        self.input_size = input_size
        self.taskcla = taskcla
        # the best model architecture for each task
        self.archis = []
        # the device and tensorboard writer for training
        self.device = device
        self.writer = writer
        self.exp_name = exp_name

        # the hyper parameters
        # the hyper parameters in cell search stage
        self.c_epochs = c_epochs
        self.c_batch = c_batch
        self.c_lr = c_lr
        self.c_lr_a = c_lr_a
        self.c_lamb = c_lamb
        # the hyper parameters in operation search stage
        self.o_epochs = o_epochs
        self.o_batch = o_batch
        self.o_lr = o_lr
        self.o_lr_a = o_lr_a
        self.o_lamb = o_lamb
        self.o_size = o_size
        # the hyper parameters in training stage
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.lamb = lamb

        self.lr_patience = lr_patience  # for dynamically update learning rate
        self.lr_factor = lr_factor  # for dynamically update learning rate
        self.clipgrad = clipgrad

        self.args = args

        if args.mode == 'search':
            # mode: search the best hyper-parameter
            # the hyper parameters in cell search stage
            self.c_epochs = args.c_epochs
            self.c_batch = args.c_batch
            self.c_lr = args.c_lr
            self.c_lr_a = args.c_lr_a
            self.c_lamb = args.c_lamb
            # the hyper parameters in operation search stage
            self.o_epochs = args.o_epochs
            self.o_batch = args.o_batch
            self.o_lr = args.o_lr
            self.o_lr_a = args.o_lr_a
            self.o_lamb = args.o_lamb
            self.o_size = args.o_size
            # the hyper parameters in training stage
            self.epochs = args.epochs
            self.batch = args.batch
            self.lr = args.lr
            self.lamb = args.lamb

            self.lr_patience = args.lr_patience  # for dynamically update learning rate
            self.lr_factor = args.lr_factor  # for dynamically update learning rate

        # define the search method
        self.auto_ml = AutoSearch(self.search_layers, self.taskcla[0][1], self.input_size,
                                  device=self.device, writer=self.writer, exp_name=self.exp_name, args=args)
        # define optimizer and loss function
        self.optimizer = None
        self.optimizer_o = None
        self.ce = nn.CrossEntropyLoss()

    def _get_optimizer(self, lr):
        # optimizer to train the model parameters
        if lr is None:
            lr = self.lr

        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=lr, weight_decay=self.lamb, momentum=0.9)

    def _get_optimizer_o(self, lr=None):
        if lr is None:
            lr = self.o_lr
        params = self.model.get_param(self.model.new_models)

        return torch.optim.SGD(params=params, lr=lr, momentum=0.9, weight_decay=self.o_lamb)

    def train(self, t, train_data, valid_data, device='cuda'):
        # training network for task t
        # 1 search cell for task t
        genotype = self.search_cell(t, train_data, valid_data, self.c_batch, self.c_epochs, device=device)
        # 2 search operation for task t > 0
        if t > 0:
            # 2.1 expand
            self.model.expand(t, genotype, device)
            # 2.2 freeze the model
            utils.freeze_model(self.model)
            self.model.modify_param(self.model.new_models, True)
            # 1.2.3 search the best expand action, the best action, and the best architecture
            self.search_t(t, train_data, valid_data, self.o_batch, self.o_epochs, device=device)
            best_archi = self.model.select(t)
            print("best_archi is {}".format(best_archi))
            self.writer.add_text("Archi for task {}".format(t),
                                "{}".format(best_archi))
            self.archis.append(best_archi)
            self.writer.add_text("ModelSize/Task_{}".format(t),
                                 "model size = {}".format(utils.get_model_size(self.model)))
            # 1.2.4 unfreeze the model that need to train
            utils.freeze_model(self.model)
            self.model.modify_param(self.model.model_to_train, True)

        # 3 training model for task t
        self.train_t(t, train_data, valid_data, self.batch, self.epochs, device)

    def train_t(self, t, train_data, valid_data, batch_size, epochs, device='cuda'):
        # training model for task t
        # 0 prepare
        print("Training stage of task {}".format(t))
        best_loss = np.inf
        best_model = utils.get_model(self.model)

        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.lr_patience,
        #                                                        factor=self.lr_factor, threshold=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # 3 training the model
        for e in range(epochs):
            # 3.1 train
            self.train_epoch(t, train_loader, device=device)
            # 3.2 compute training loss
            train_loss, train_acc = self.eval(t, train_loader, mode='train', device=device)
            # 3.3 compute valid loss
            valid_loss, valid_acc = self.eval(t, valid_loader, mode='train', device=device)
            # 3.4 logging
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | Valid: loss={:.3f}, acc={:5.1f}% |'.format(
                e, train_loss, 100*train_acc, valid_loss, 100 * valid_acc))
            self.writer.add_scalars('Train_Loss/Task: {}'.format(t),
                                    {'train_loss': train_loss, 'valid_loss': valid_loss},
                                    global_step=e)
            self.writer.add_scalars('Train_Accuracy/Task: {}'.format(t),
                                    {'train_acc': train_acc*100, 'valid_acc': valid_acc*100},
                                    global_step=e)

            # 3.5 Adapt learning rate
            scheduler.step(valid_loss)
            # 3.6 update the best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
        # 4 Restore best model
        utils.set_model_(self.model, best_model)
        return

    def train_epoch(self, t, train_loader, device='cuda'):
        self.model.train()
        # set tht mode of models which are reused (BN)
        if t > 0:
            for i in range(self.model.length['stem']):
                if i not in self.model.model_to_train['stem']:
                    self.model.stem[i].eval()
            for i in range(len(self.model.cells)):
                for k in range(self.model.length['cell' + str(i)]):
                    if k not in self.model.model_to_train['cell' + str(i)]:
                        self.model.cells[i][k].eval()
        # Loop batch
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # forward
            outputs = self.model.forward(x, t, self.archis[t])
            # todo test
            # outputs = self.model.forward(x, t, self.archis[0])
            output = outputs[t]
            loss = self.criterion(output, y)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def search_cell(self, t, train_data, valid_data, batch_size, nepochs, device):
        print("Search cell for task {}".format(t))
        self.auto_ml = AutoSearch(self.search_layers, self.taskcla[t][1], self.input_size,
                                  device=self.device, writer=self.writer, exp_name=self.exp_name, args=self.args)
        genotype = deepcopy(self.auto_ml.search(t, train_data, valid_data, batch_size, nepochs))

        if t == 0:
            self.model = Network(self.input_size, self.taskcla, self.eval_layers, 36, genotype, device).to(device)
            self.archis.append(self.model.arch_init)

            self.writer.add_text("Task_0/genotype",
                                 "genotype = {}".format(genotype),
                                 global_step=0)
            self.writer.add_text("ModelSize/Task_0",
                                 "model size = {}".format(utils.get_model_size(self.model)))
        return genotype

    def search_t(self, t, train_data, valid_data, batch_size, epochs, device):
        # search operations for task t(t>0)
        # 0 prepare
        print("Search Stage of task {}".format(t))
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.o_lr
        # 1 define optimizers and scheduler
        self.optimizer_o = self._get_optimizer_o(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_o, epochs, eta_min=0.001)
        # 2 define the dataloader
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(0.5 * num_train))
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            num_workers=4, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            num_workers=4, pin_memory=True)

        h_e = [torch.full(pro.size(), 0, dtype=torch.long) for pro in self.model.p]
        h_a = [torch.full(pro.size(), 0.0, dtype=torch.float) for pro in self.model.p]
        
        for k in range(len(h_e)):
            h_e[k][0:-1] = self.o_size

        # 3 search the best model architecture
        for e in range(epochs):
            # 3.0 prepare
            for k, pro in enumerate(self.model.p):
                self.writer.add_histogram('Archi_task_{}/{}'.format(t, k),
                                          pro, global_step=e)
            # 3.1 sample
            selected_ops = [torch.multinomial(pro, 1).item() for pro in self.model.p]
            # print("Selected ops: {}".format(selected_ops))
            self.writer.add_text("Selected ops for task {}".format(t),
                                "{}".format(selected_ops))

            # 3.2 train
            train_loss, train_acc = self.search_epoch(t, train_loader, selected_ops, device)
            # print('train_acc: {}'.format(train_acc))
            valid_loss, valid_acc = self.search_eval(t, valid_loader, selected_ops, device)
            # print('valid_acc: {}'.format(valid_acc))
            # logging
            self.writer.add_scalars('Search_Loss/Task: {}'.format(t),
                                    {'train_loss': train_loss, 'valid_loss': valid_loss},
                                    global_step=e)
            self.writer.add_scalars('Search_Accuracy/Task: {}'.format(t),
                                    {'train_acc': train_acc, 'valid_acc': valid_acc},
                                    global_step=e)
            # 3.3 update h_e and h_a
            for i, idx in enumerate(selected_ops):
                h_e[i][idx] += 1
                h_a[i][idx] = valid_acc
                # print("layer {}".format(i).center(50, "*"))
                # print("h_e in layer {}".format(i).center(50, "*"))
                # print(h_e[i])
                # print("h_a in layer {}".format(i).center(50, "*"))
                # print(h_a[i])

            # 3.4 update the probability
            for k in range(len(self.model.p)):
                dh_e_k = torch.reshape(h_e[k], (1, -1)) - torch.reshape(h_e[k], (-1, 1))
                dh_a_k = torch.reshape(h_a[k], (1, -1)) - torch.reshape(h_a[k], (-1, 1))

                # modify
                # vector1 = torch.sum((dh_e_k < 0) * (dh_a_k > 0), dim=1)
                vector1 = torch.sum((dh_e_k < 0) * (dh_a_k > 0), dim=0)
                # vector1[-1] /= self.o_size
                # print("vector1: {}".format(vector1))
                # vector2 = torch.sum((dh_e_k > 0) * (dh_a_k < 0), dim=1)
                vector2 = torch.sum((dh_e_k > 0) * (dh_a_k < 0), dim=0)
                # print("vector2: {}".format(vector2))
                update = (vector1 - vector2).float()
                # update[-1] /= self.o_size
                self.model.p[k] += (self.o_lr_a * update)
                self.model.p[k] = F.softmax(self.model.p[k])

            # 3.5 Adapt learning rate
            scheduler.step()
            # 3.6 update the best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)

        # 4 Restore best model
        utils.set_model_(self.model, best_model)
        return

    def search_epoch(self, t, train_loader, selected_ops, device='cuda'):
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_num = 0
        # set tht mode of models which are reused (BN)
        for i in range(self.model.length['stem']):
            self.model.stem[i].eval()
        for i in range(len(self.model.cells)):
            for k in range(self.model.length['cell' + str(i)]):
                self.model.cells[i][k].eval()

        # 2 Loop batches
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            length = x.size()[0]

            # 2.2.3 Forward current model
            outputs = self.model.search_forward(x, selected_ops)
            output = outputs[t]
            loss = self.criterion(output, y)
            # 2.2.4 Backward
            self.optimizer_o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer_o.step()

            with torch.no_grad():
                _, pred = output.max(1)
                hits = (pred == y).float()
                total_loss += loss.item() * length
                total_acc += hits.sum().item()
                total_num += length

        return total_loss / total_num, total_acc / total_num

    def eval(self, t, test_loader, mode, device):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                length = x.size()[0]
                # forward
                outputs = self.model.forward(x, t, self.archis[t])
                output = outputs[t]
                # compute loss
                loss = self.criterion(output, y)

                _, pred = output.max(1)
                hits = (pred == y).float()
                # Log
                total_loss += loss.item() * length
                total_acc += hits.sum().item()
                total_num += length

        return total_loss / total_num, total_acc / total_num

    def search_eval(self, t, test_loader, selected_ops, device):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                length = x.size()[0]
                # forward
                outputs = self.model.search_forward(x, selected_ops)
                output = outputs[t]
                # compute loss
                loss = self.criterion(output, y)
                _, pred = output.max(1)
                hits = (pred == y).float()
                # Log
                total_loss += loss.item() * length
                total_acc += hits.sum().item()
                total_num += length

        return total_loss / total_num, total_acc / total_num

    def criterion(self, output, targets):

        return self.ce(output, targets)

