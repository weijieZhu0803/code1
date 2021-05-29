import torch
from torch.autograd import Variable
import time
import os
import torch.nn.functional as F
from torch import nn
# import tqdm
import numpy as np
import sys

from utils import AverageMeter


def train_epoch(epoch, data_loader_1, data_loader_2, model_1, model_2, criterion, opt, epoch_logger, optimizer_1,
                optimizer_2):
    device_ids = [0, 1]
    print('train at epoch {}'.format(epoch))
    total_loss_dic = {}
    cross_loss_dic = {}
    model_output_dic = {}
    total_model_dic = {}
    optimizer_dic = {'0': optimizer_1, '1': optimizer_2}
    acc_dic = {}
    label_1 = torch.from_numpy(np.array([]))
    label_2 = torch.from_numpy(np.array([]))
    total_out1 = torch.from_numpy(np.array([]))
    total_out2 = torch.from_numpy(np.array([]))
    total_loss_net1 = 0
    total_loss_net2 = 0
    main_cross_loss = 0
    main_KL_loss = 0
    aux_cross_loss = 0
    aux_KL_loss = 0
    total_model_dic[str(0)] = model_1
    total_model_dic[str(1)] = model_2
    kl_loss = {}
    acc_1 = 0
    acc_2 = 0
    loss = 0
    k = 0
    for i in range(2):
        cross_loss_dic["{0}".format(i)] = 0
        model_output_dic["{0}".format(i)] = 0
        total_loss_dic["{0}".format(i)] = 0
    for model in total_model_dic:
        total_model_dic[model] = nn.DataParallel(total_model_dic[model], device_ids=device_ids).cuda(
            device=device_ids[0]).train()
        # total_model_dic[model].train()
    losses = AverageMeter()
    accuracies = AverageMeter()

    for (inputs_1, targets_1), (inputs_2, targets_2) in zip(data_loader_1, data_loader_2):
        k = k + 1
        targets_1 = targets_1.long()
        inputs_1 = Variable(inputs_1.cuda(device=device_ids[0]))
        target_1 = Variable(targets_1.cuda(device=device_ids[0]))

        targets_2 = targets_2.long()
        inputs_2 = Variable(inputs_2.cuda(device=device_ids[0]))
        target_2 = Variable(targets_2.cuda(device=device_ids[0]))

        for model_num, _model in enumerate(total_model_dic):
            if model_num == 0:
                outputs = total_model_dic[str(model_num)](inputs_1)
                _, targets = torch.max(target_1, 1)
                label_1 = label_1.long()
                label_1 = torch.cat((label_1.cuda(device=device_ids[0]), targets), 0)
                total_out1 = total_out1.clone().detach().float().cuda(device=device_ids[0])
                total_out1 = torch.cat((total_out1, outputs), 0)
            elif model_num == 1:
                outputs = total_model_dic[str(model_num)](inputs_2)
                _, targets = torch.max(target_2, 1)
                label_2 = label_2.long()
                label_2 = torch.cat((label_2.cuda(device=device_ids[0]), targets), 0)
                total_out2 = total_out2.clone().detach().float().cuda(device=device_ids[0])
                total_out2 = torch.cat((total_out2, outputs), 0)

            cross_loss_dic[str(model_num)] = criterion(outputs, targets)
            acc_dic[str(model_num)] = calculate_acc(outputs, targets)
            model_output_dic[str(model_num)] = outputs



        for i, model_first in enumerate(model_output_dic):
            first_output = model_output_dic[model_first]
            for j, model_second in enumerate(model_output_dic):
                # Not Calculate self model
                if (i == j):
                    continue
                second_output = model_output_dic[model_second]
                first_output = F.log_softmax(first_output)
                second_output = F.softmax(second_output)
                kl_loss[str(i)] = kl_loss_compute(first_output, second_output)

        for i, key in enumerate(total_loss_dic):
            if i == 0:

                # if cross_loss_dic[str(i)]>0.2:
                #  total_loss_dic[str(i)] = 0.55*cross_loss_dic[str(i)]+0.45*kl_loss[str(i)]
                # else:
                #  total_loss_dic[str(i)] = 0.45*cross_loss_dic[str(i)]+0.55*kl_loss[str(i)]
                total_loss_dic[str(i)] = cross_loss_dic[str(i)] + kl_loss[str(i)]
                # total_loss_dic[str(i)] = cross_loss_dic[str(i)]
                aux_cross_loss += cross_loss_dic[str(i)] * targets.size(0)
                aux_KL_loss += kl_loss[str(i)] * targets.size(0)
                total_loss_net1 += total_loss_dic[str(i)] * targets.size(0)
            elif i == 1:

                # if cross_loss_dic[str(i)]>0.2:
                #  total_loss_dic[str(i)] = 0.55*cross_loss_dic[str(i)]+0.45*kl_loss[str(i)]
                # else:
                #  total_loss_dic[str(i)] = 0.45*cross_loss_dic[str(i)]+0.55*kl_loss[str(i)]
                total_loss_dic[str(i)] = cross_loss_dic[str(i)] + kl_loss[str(i)]
                # total_loss_dic[str(i)] = cross_loss_dic[str(i)]
                total_loss_net2 += total_loss_dic[str(i)] * targets.size(0)
                main_cross_loss += cross_loss_dic[str(i)] * targets.size(0)
                main_KL_loss += kl_loss[str(i)] * targets.size(0)
        print('Epoch: [{0}][{1}/{2}]\t'
              'aux_loss {aux_loss:.4f} \t'
              'main_loss {main_loss:.4f} \t'
              'aux_acc {aux_acc:.4f} \t'
              'main_acc {main_acc:.4f} '.format(
            epoch,
            k,
            len(data_loader_1),
            aux_loss=total_loss_dic[str(0)],
            main_loss=total_loss_dic[str(1)],
            aux_acc=acc_dic[str(0)],
            main_acc=acc_dic[str(1)]))

        for i, model in enumerate(total_model_dic):

            total_model_dic[str(i)].zero_grad()
            if i == 0:

                total_loss_dic[str(i)].backward(retain_graph=True)
            elif i == 1:
                total_loss_dic[str(i)].backward()

            optimizer_dic[str(i)].step()

    for i, n_model in enumerate(model_output_dic):
        if i == 0:
            acc_1 = calculate_acc(total_out1, label_1)
        elif i == 1:
            acc_2 = calculate_acc(total_out2, label_2)

    print('Epoch:[{0}]\t'
          'aux_loss  {aux_loss:.4f}\t'
          'main_loss  {main_loss:.4f}\t'
          'main_cross_loss  {main_cross_loss:.4f}\t'
          'main_KL_loss  {main_KL_loss:.4f}\t'
          'aux_acc {aux_acc:.4f}\t'
          'main_acc {main_acc:.4f}'.format(
        epoch,
        aux_loss=total_loss_net1 / label_1.size(0),
        main_loss=total_loss_net2 / label_2.size(0),
        main_cross_loss=main_cross_loss / label_2.size(0),
        main_KL_loss=main_KL_loss / label_2.size(0),
        aux_acc=acc_1,
        main_acc=acc_2
    ))
    main_cross_loss = main_cross_loss.detach().cpu().numpy() / label_2.size(0)
    main_KL_loss = main_KL_loss.detach().cpu().numpy() / label_2.size(0)
    aux_cross_loss = aux_cross_loss.detach().cpu().numpy() / label_1.size(0)
    aux_KL_loss = aux_KL_loss.detach().cpu().numpy() / label_1.size(0)
    aux_loss = total_loss_net1.detach().cpu().numpy() / label_1.size(0)
    aux_loss = np.round(aux_loss, 4)
    main_loss = total_loss_net2.detach().cpu().numpy() / label_2.size(0)
    main_loss = np.round(main_loss, 4)
    aux_acc = acc_1.cpu().numpy()
    aux_acc = np.round(aux_acc, 4)
    main_acc = acc_2.cpu().numpy()
    main_acc = np.round(main_acc, 4)
    epoch_logger.log({
        'epoch': epoch,
        'aux_loss': aux_loss,
        'main_loss': main_loss,
        'main_cross_loss': main_cross_loss,
        'main_KL_loss': main_KL_loss,
        'aux_cross_loss': aux_cross_loss,
        'aux_KL_loss': aux_KL_loss,
        'aux_acc': aux_acc,
        'main_acc': main_acc,
        'lr': optimizer_dic[str(1)].param_groups[0]['lr']
    })


def kl_loss_compute(logits1, logits2):
    device_ids = [0, 1]
    kl_criterion = nn.KLDivLoss().cuda(device=device_ids[0])
    kl_loss = kl_criterion(logits1, logits2)
    return kl_loss


def calculate_acc(outputs, targets):
    device_ids = [0, 1]
    _, pre = torch.max(outputs, dim=1)
    correct = torch.zeros(1).squeeze().cuda(device=device_ids[0])
    # pre = pre.t()
    # correct = pre.eq(targets.view(1, -1))
    correct += (pre == targets).sum().float()
    # n_correct_elems = correct.float().sum().item()
    num = targets.size(0)
    return correct / num
