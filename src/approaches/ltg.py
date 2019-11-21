import sys,time
import numpy as np
import torch

import utils


class Appr(object):
    """ Class implementing the Learn to Grow approach """

    def __init__(self, model,
                 o_epochs=100, o_batch=512, o_lr=0.025, o_lr_a=0.01, o_lamb=3e-4, o_lamb_a=3e-4, o_lamb_size=1, o_lr_min=1e-3,
                 epochs=20, batch=128, lr=0.025, lamb=3e-4,
                 lr_factor=3, lr_patience=5, clipgrad=5,
                 writer=None, exp_name="None", device='cuda', args=None):
        self.model = model
        # architecture of each submodel of each layer of each task
        self.archis = [self.model.init_archi]  # initial with architecture of task 0
        self.device = device
        self.writer = writer
        self.exp_name = exp_name

        # the hyper parameters
        # the hyper parameters in operation search stage
        self.o_epochs = o_epochs
        self.o_batch = o_batch
        self.o_lr = o_lr
        self.o_lr_a = o_lr_a
        self.o_lamb = o_lamb
        self.o_lamb_a = o_lamb_a
        self.o_lamb_size = o_lamb_size
        # the hyper parameters in training stage
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.lamb = lamb

        self.lr_factor = lr_factor
        self.lr_patience = lr_patience

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = None
        self.optimizer_oa = None
        self.optimizer_o = None
        self.clipgrad = clipgrad
        if args.mode == 'search':
            # mode: search the best hyper-parameter
            # the hyper parameters in operation search stage
            self.o_epochs = args.o_epochs
            self.o_batch = args.o_batch
            self.o_lr = args.o_lr
            self.o_lr_a = args.o_lr_a
            self.o_lamb = args.o_lamb
            self.o_lamb_a = args.o_lamb_a
            self.o_lamb_size = args.o_lamb_size
            # the hyper parameters in training stage
            self.epochs = args.epochs
            self.batch = args.batch
            self.lr = args.lr
            self.lamb = args.lamb

            self.lr_patience = args.lr_patience  # for dynamically update learning rate
            self.lr_factor = args.lr_factor  # for dynamically update learning rate

        return

    def _get_optimizer(self, lr):
        # optimizer to train the model parameters
        if lr is None:
            lr = self.lr

        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=lr, momentum=0.9, weight_decay=self.lamb)

    def _get_optimizer_oa(self, lr=None):
        if lr is None:
            lr = self.o_lr_a
        all_params = self.model.get_archi_param()
        params = []
        for p in all_params:
            for _, param in p.items():
                params.append(param)

        return torch.optim.Adam(params=params, lr=lr, betas=(0.5, 0.999), weight_decay=self.o_lamb_a)

    def _get_optimizer_o(self, lr=None):
        if lr is None:
            lr = self.o_lr
        params = self.model.get_param(self.model.new_models)

        return torch.optim.SGD(params=params, lr=lr, momentum=0.9, weight_decay=self.o_lamb)

    def train(self, t, train_data, valid_data, device='cuda'):
        # train network for task t
        # 1 search the best model for task 1:n
        if t > 0:
            # 1.2.1 expand
            self.model.expand(t, device)
            # 1.2.2 freeze the model
            utils.freeze_model(self.model)
            # 1.2.3 search the best expand action
            self.search_network(t, train_data, valid_data, self.o_batch, self.o_epochs, device=device)
            # 1.2.4 select the best action
            best_archi = self.model.select(t)
            self.archis.append(best_archi)
            # 1.2.5 unfreeze the model that need to train
            utils.freeze_model(self.model)
            self.model.modify_param(self.model.model_to_train, True)
            # 1.2.6 look up the super model
            print(best_archi)
            utils.print_model_report(self.model)

        # 2 training
        self.train_network(t, train_data, valid_data, self.batch, self.epochs, device)

    def train_network(self, t, train_data, valid_data, batch_size, epochs, device='cuda'):
        # 0 prepare
        print("Training Begin")
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        # 1 define the optimizer and scheduler
        self.optimizer = self._get_optimizer(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.lr_patience,
        #                                                        factor=self.lr_factor)
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

    def train_epoch(self, t, train_loader, device='cuda'):
        self.model.train()
        # Loop batch
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # forward
            outputs = self.model.forward(x, t, self.archis[t])
            output = outputs[t]
            loss = self.criterion(output, y)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def search_network(self, t, train_data, valid_data, batch_size, epochs, device='cuda'):
        # 0 prepare
        print("Search Stage")
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr_a = self.o_lr_a
        lr = self.o_lr
        # 1 define optimizers
        self.optimizer_oa = self._get_optimizer_oa(lr_a)
        self.optimizer_o = self._get_optimizer_o(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_o, epochs)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_o, patience=self.lr_patience,
        #                                                        factor=self.lr_factor)
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

        # 3 training the model
        for e in range(epochs):
            # 3.1 search
            self.search_epoch(t, train_loader, valid_loader, device)
            # 3.2 compute training loss
            train_loss, train_acc = self.eval(t, train_loader, mode='search', device=device)
            # 3.3 compute valid loss
            valid_loss, valid_acc = self.eval(t, valid_loader, mode='search', device=device)
            # 3.4 logging
            print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | Valid: loss={:.3f}, acc={:5.1f}% |'.format(
                e, train_loss, 100 * train_acc, valid_loss, 100 * valid_acc))
            self.writer.add_scalars('Search_Loss/Task: {}'.format(t),
                                    {'train_loss': train_loss, 'valid_loss': valid_loss},
                                    global_step=e)
            self.writer.add_scalars('Search_Accuracy/Task: {}'.format(t),
                                    {'train_acc': train_acc * 100, 'valid_acc': valid_acc * 100},
                                    global_step=e)
            # 3.5 Adapt lr
            scheduler.step()

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)

        # 4 Restore best model
        utils.set_model_(self.model, best_model)

    def search_epoch(self, t, train_loader, valid_loader, device='cuda'):
        # using first-order approximation in DARTS
        self.model.train()

        # 2 Loop batches
        for x_m, y_m in train_loader:
            x_m, y_m = x_m.to(device), y_m.to(device)
            x_a, y_a = next(iter(valid_loader))
            x_a, y_a = x_a.to(device), y_a.to(device)

            # 2.1 optimize the archi parameter
            # 2.1.1 freeze the new model parameter and task specific layer
            self.model.modify_param(self.model.new_models, requires_grad=False)
            # 2.1.2 unfreeze architecture parameter
            self.model.modify_archi_param(requires_grad=True)
            # 2.1.3 Forward current model
            outputs = self.model.search_forward(x_a, t)
            output = outputs[t]
            loss = self.criterion_search(t, output, y_a)

            # 2.1.4 Backward
            self.optimizer_oa.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer_oa.step()

            # 2.2 optimize the model parameter
            # 2.2.1 freeze the new model parameter and task specific layer
            self.model.modify_param(self.model.new_models, requires_grad=True)
            # 2.2.2 unfreeze architecture parameter
            self.model.modify_archi_param(requires_grad=False)
            # 2.2.3 Forward current model
            outputs = self.model.search_forward(x_m, t)
            output = outputs[t]
            loss = self.criterion_search(t, output, y_m)

            # 2.2.4 Backward
            self.optimizer_o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer_o.step()

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
                # forward
                outputs = None
                if mode == 'train':  # in training stage
                    outputs = self.model.forward(x, t, self.archis[t])
                elif mode == 'search':  # in search stage
                    outputs = self.model.search_forward(x, t)
                output = outputs[t]
                # compute loss
                loss = None
                if mode == 'train':  # in training stage
                    loss = self.criterion(output, y)
                elif mode == 'search':
                    loss = self.criterion_search(t, output, y)
                _, pred = output.max(1)
                hits = (pred == y).float()
                # Log
                total_loss += loss.item() * length
                total_acc += hits.sum().item()
                total_num += length

        return total_loss / total_num, total_acc / total_num

    def criterion(self, output, targets):

        return self.ce(output, targets)

    def criterion_search(self, t, output, targets):
        loss_reg = self.model.regular_loss()

        return self.ce(output, targets) + self.o_lamb_size * loss_reg
