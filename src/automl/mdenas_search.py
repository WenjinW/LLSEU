import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

import automl.darts_utils_cnn as utils

from automl.mdenas_basicmodel import BasicNetwork
from automl.darts_model import Network


class AutoSearch(object):
    # Implements a NAS methods MdeNAS
    def __init__(self, num_cells, num_class=10, input_size=None, lr=0.025, lr_a=3e-4, lr_min=0.001, momentum=0.9,
                 weight_decay=3e-4, weight_decay_a=1e-3, grad_clip=5, unrolled=False,
                 device='cuda: 0', writer=None, exp_name=None, save_name='EXP', args=None):
        self.num_cells = num_cells
        self.num_classes = num_class
        self.input_size = input_size

        self.device = device
        self.writer = writer
        self.exp_name = exp_name

        self.lr = lr
        self.lr_a = lr_a
        self.lr_min = lr_min
        self.momentum = momentum
        self.weight_decay = weight_decay
        if args.mode == 'search':
            self.lr = args.c_lr
            self.lr_a = args.c_lr_a
            self.weight_decay = args.c_lamb
            self.c_epochs = args.c_epochs
            self.c_batch = args.c_batch

        self.grad_clip = grad_clip
        self.save_name = save_name

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.model = BasicNetwork(self.input_size[0], 16, self.num_classes, self.num_cells, self.criterion,
                                  device=self.device).to(device)
        logging.info("param size = %fMB", utils.count_parameters_in_MB(self.model))

        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)

    def search(self, t, train_data, valid_data, batch_size, nepochs):
        """ search a model genotype for the given task

        :param train_data: the dataset of training data of the given task
        :param valid_data: the dataset of valid data of the given task
        :param batch_size: the batch size of training
        :param nepochs: the number of training epochs
        :return:
            genotype: the selected architecture for the given task
        """
        # dataloader of training data
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(0.5 * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=4)
        # dataloader of valid date
        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=4)
        # the scheduler of learning rate of model parameters optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, nepochs,
                                                               eta_min=self.lr_min)
        best_loss = np.inf
        best_model = utils.get_model(self.model)

        h_e = {
            'normal': torch.full((self.model.num_edges, self.model.num_ops), 0, dtype=torch.long),
            'reduce': torch.full((self.model.num_edges, self.model.num_ops), 0, dtype=torch.long)
        }
        h_a = {
            'normal': torch.full((self.model.num_edges, self.model.num_ops), 0.0),
            'reduce': torch.full((self.model.num_edges, self.model.num_ops), 0.0)
        }
        for epoch in range(nepochs):
            # 0 prepare
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
            print('epoch: {} lr: {}'.format(epoch, lr))
            genotype = self.model.genotype()
            logging.info('genotype = %s', genotype)
            # 1 sample
            p_n = self.model.probability()["normal"]
            p_r = self.model.probability()["reduce"]
            self.writer.add_histogram('CellArchHist/Normal',
                                      p_n, global_step=epoch)
            self.writer.add_histogram('CellArchHist/Reduce',
                                      p_r, global_step=epoch)
            selected_ops = {
                'normal': torch.multinomial(p_n, 1).view(-1),
                'reduce': torch.multinomial(p_r, 1).view(-1)
            }
            # 2 train
            train_acc, train_obj = self.train(train_queue, selected_ops)
            print('train_acc: {}'.format(train_acc))
            valid_acc, valid_obj = self.eval(valid_queue, selected_ops)
            print('valid_acc: {}'.format(valid_acc))
            # logging
            self.writer.add_scalars('Search_Cell_Loss/Task: {}'.format(t),
                                    {'train_loss': train_obj, 'valid_loss': valid_obj},
                                    global_step=epoch)
            self.writer.add_scalars('Search_Cell_Accuracy/Task: {}'.format(t),
                                    {'train_acc': train_acc, 'valid_acc': valid_acc},
                                    global_step=epoch)

            # 3 update h_e and h_a
            for cell_type in ['normal', 'reduce']:
                # for each edge
                for i, idx in enumerate(selected_ops[cell_type]):
                    h_e[cell_type][i][idx] += 1
                    h_a[cell_type][i][idx] = valid_acc

            # 4 update the probability
            for k in range(self.model.num_edges):
                dh_e_k = {
                    'normal': torch.reshape(h_e['normal'][k], (1, -1)) - torch.reshape(h_e['normal'][k], (-1, 1)),
                    'reduce': torch.reshape(h_e['reduce'][k], (1, -1)) - torch.reshape(h_e['reduce'][k], (-1, 1))
                }
                dh_a_k = {
                    'normal': torch.reshape(h_a['normal'][k], (1, -1)) - torch.reshape(h_a['normal'][k], (-1, 1)),
                    'reduce': torch.reshape(h_a['reduce'][k], (1, -1)) - torch.reshape(h_a['reduce'][k], (-1, 1))
                }
                for cell_type in ['normal', 'reduce']:
                    # vector1 = torch.sum((dh_e_k[cell_type] < 0) * (dh_a_k[cell_type] > 0), dim=1)
                    vector1 = torch.sum((dh_e_k[cell_type] < 0) * (dh_a_k[cell_type] > 0), dim=0)
                    # vector2 = torch.sum((dh_e_k[cell_type] > 0) * (dh_a_k[cell_type] < 0), dim=1)
                    vector2 = torch.sum((dh_e_k[cell_type] > 0) * (dh_a_k[cell_type] < 0), dim=0)
                    self.model.p[cell_type][k] += (self.lr_a * (vector1-vector2).float())
                    self.model.p[cell_type][k] = F.softmax(self.model.p[cell_type][k])

            # adjust learning according the scheduler
            scheduler.step()

            if valid_obj < best_loss:
                best_model = utils.get_model(self.model)
                best_loss = valid_obj

        # the best model and its architecture
        utils.set_model(self.model, best_model)
        print("The best architecture is", self.model.genotype())
        return self.model.genotype()

    def train(self, train_queue, selected_ops):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        # top5 = utils.AverageMeter()

        for step, (x, y) in enumerate(train_queue):
            self.model.train()
            n = x.size(0)
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x, selected_ops)
            loss = self.criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            with torch.no_grad():
                prec1 = utils.accuracy(logits, y, topk=1)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                if step % 100 == 0:
                    logging.info('train %03d %e %f %f', step, objs.avg, top1.avg)
                    print('train: {} loss: {} acc: {}'.format(step, objs.avg, top1.avg))

        return top1.avg, objs.avg

    def eval(self, valid_queue, selected_ops):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(valid_queue):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x, selected_ops)
                loss = self.criterion(logits, y)
                prec1 = utils.accuracy(logits, y, topk=1)
                n = x.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                if step % 100 == 0:
                    logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg)
                    print('valid: {} loss: {} acc: {}'.format(step, objs.avg, top1.avg))

        return top1.avg, objs.avg

    def create_model(self, archi, num_cells, input_size, task_classes, init_channel):

        return Network(input_size, task_classes, num_cells, init_channel, archi, device=self.device)



