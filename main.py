#!/usr/bin/python
import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.utils.data as Data
from opts import parse_opts
from model import generate_model
from utils import Logger
from train import train_epoch
from val import val_epoch


def adjust_lr(optimizer, epoch):
    lr = opt.learning_rate * (0.5 ** (epoch // 200))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    opt = parse_opts()
    if opt.root_path != '':
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    device_ids = [1, 2]
    model_1, parameters_1 = generate_model(opt=opt)
    print(model_1)
    model_2, parameters_2 = generate_model(opt=opt)
    # save_model=torch.load('CK+_4(gray112)-TIM32-weights-improvement-TIM32-650-1.0.pth')
    # for k in save_model["state_dict"]:
    #   print(k)
    # del save_model["state_dict"]['fc1.weight']
    # del save_model["state_dict"]["fc1.bias"]
    # model_dict=model_2.state_dict()
    # state_dict={k:v for k,v in save_model.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # model_2.load_state_dict(model_dict)
    criterion = torch.nn.CrossEntropyLoss()
    if opt.no_cuda:
        criterion = criterion.cuda(device=device_ids[0])

    if not opt.no_train:
        train_data_1 = np.load('./S_C_C2_4_new/fold4/S_C_C2_4(flow112)_fold4_TIM32_train.npy')
        train_labels_1 = np.load('./S_C_C2_4_new/fold4/S_C_C2_4(flow112)_fold4_TIM32_train_labels.npy')
        train_data_1 = torch.from_numpy(train_data_1)
        train_labels_1 = torch.from_numpy(train_labels_1)
        training_data_1 = Data.TensorDataset(train_data_1, train_labels_1)
        train_loader_1 = torch.utils.data.DataLoader(
            training_data_1,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_data_2 = np.load('./S_C_C2_4_new/fold4/S_C_C2_4(gray112)_fold4_TIM32_train.npy')
        train_labels_2 = np.load('./S_C_C2_4_new/fold4/S_C_C2_4(gray112)_fold4_TIM32_train_labels.npy')
        print(train_data_2.shape)
        print(train_labels_2.shape)
        train_data_2 = torch.from_numpy(train_data_2)
        train_labels_2 = torch.from_numpy(train_labels_2)
        training_data_2 = Data.TensorDataset(train_data_2, train_labels_2)
        train_loader_2 = torch.utils.data.DataLoader(
            training_data_2,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'aux_loss', 'main_loss', 'main_cross_loss', 'main_KL_loss', 'aux_cross_loss', 'aux_KL_loss','aux_acc', 'main_acc', 'lr'])
             
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer_1 = optim.SGD(parameters_1, lr=opt.learning_rate, momentum=0.9, dampening=dampening,
                                weight_decay=1e-4, nesterov=True)
        optimizer_2 = optim.SGD(parameters_2, lr=opt.learning_rate, momentum=0.9, dampening=dampening,
                                weight_decay=1e-4, nesterov=True)
        # optimizer_1=optim.Adam(parameters_1,lr=opt.learning_rate,betas=(0.9,0.999),eps=1e-08, weight_decay=1e-4)
        # optimizer_2=optim.Adam(parameters_2,lr=opt.learning_rate,betas=(0.9,0.999),eps=1e-08, weight_decay=1e-4)

    if not opt.no_val:
        validation_data_1 = np.load('./S_C_C2_4_new/fold4/S_C_C2_4(flow112)_fold4_TIM32_test.npy')
        validation_labels_1 = np.load('./S_C_C2_4_new/fold4/S_C_C2_4(flow112)_fold4_TIM32_test_labels.npy')
        validation_data_1 = torch.from_numpy(validation_data_1)
        validation_labels_1 = torch.from_numpy(validation_labels_1)
        validation_data_1 = Data.TensorDataset(validation_data_1, validation_labels_1)

        val_loader_1 = torch.utils.data.DataLoader(
            validation_data_1,
            batch_size=64,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        validation_data_2 = np.load('./S_C_C2_4_new/fold4/S_C_C2_4(gray112)_fold4_TIM32_test.npy')
        validation_labels_2 = np.load('./S_C_C2_4_new/fold4/S_C_C2_4(gray112)_fold4_TIM32_test_labels.npy')
        print(validation_data_2.shape)
        print(validation_labels_2.shape)
        validation_data_2 = torch.from_numpy(validation_data_2)
        validation_labels_2 = torch.from_numpy(validation_labels_2)
        validation_data_2 = Data.TensorDataset(validation_data_2, validation_labels_2)

        val_loader_2 = torch.utils.data.DataLoader(
            validation_data_2,
            batch_size=64,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'aux_loss', 'main_loss', 'aux_acc', 'main_acc'])

    print('run')
    Val_aux_acc_max = 0
    Val_main_acc_max = 0
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        adjust_lr(optimizer_1, i)
        adjust_lr(optimizer_2, i)
        if not opt.no_train:
            train_epoch(i, train_loader_1, train_loader_2, model_1, model_2, criterion, opt, train_logger, optimizer_1,
                        optimizer_2)

        if not opt.no_val:
            print("val:")
            Val_aux_acc_max, Val_main_acc_max = val_epoch(i, val_loader_1, val_loader_2, model_1, model_2, criterion,opt,Val_aux_acc_max, Val_main_acc_max, val_logger)
                                                         
                                                          
    print("train over!")



