import torch
from torch.autograd import Variable
import time
import sys
import os
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch, data_loader_1, data_loader_2, model_1, model_2, criterion, opt, Val_aux_acc_max, Val_main_acc_max,
              val_logger):
    print('validation at epoch {}'.format(epoch))

    device_ids = [1, 2]

    total_loss_dic = {}
    cross_loss_dic = {}
    model_output_dic = {}
    total_model_dic = {}
    acc_dic = {}
    label_1 = torch.from_numpy(np.array([]))
    label_2 = torch.from_numpy(np.array([]))
    total_out1 = torch.from_numpy(np.array([]))
    total_out2 = torch.from_numpy(np.array([]))
    total_model_dic[str(0)] = model_1
    total_model_dic[str(1)] = model_2
    kl_loss = {}
    for i in range(2):
        cross_loss_dic["{0}".format(i)] = 0
        model_output_dic["{0}".format(i)] = 0
        total_loss_dic["{0}".format(i)] = 0
        kl_loss["{0}".format(i)] = 0
    model_1 = nn.DataParallel(model_1, device_ids=device_ids).cuda(device=device_ids[0]).eval()
    model_2 = nn.DataParallel(model_2, device_ids=device_ids).cuda(device=device_ids[0]).eval()




    with torch.no_grad():
        for k, (inputs, targets) in enumerate(data_loader_1):
            targets = targets.long()
            inputs = Variable(inputs.cuda(device=device_ids[0]))
            target_1 = Variable(targets.cuda(device=device_ids[0]))
            outputs_1 = total_model_dic[str(0)](inputs)
            _, targets_1 = torch.max(target_1, 1)
            cross_loss_dic[str(0)] = criterion(outputs_1, targets_1)
            total_out1 = total_out1.clone().detach().float().cuda(device=device_ids[0])
            total_out1 = torch.cat((total_out1, outputs_1), 0)
            label_1 = label_1.long().cuda(device=device_ids[0])
            label_1 = torch.cat((label_1, targets_1), 0)
            total_loss_dic[str(0)] += cross_loss_dic[str(0)] * targets.size(0)
            model_output_dic[str(0)] = total_out1

    with torch.no_grad():
        for k, (inputs, targets) in enumerate(data_loader_2):
            targets = targets.long()
            inputs = Variable(inputs.cuda(device=device_ids[0]))
            target_2 = Variable(targets.cuda(device=device_ids[0]))
            outputs_2 = total_model_dic[str(1)](inputs)
            _, targets_2 = torch.max(target_2, 1)
            cross_loss_dic[str(1)] = criterion(outputs_2, targets_2)
            total_out2 = total_out2.clone().detach().float().cuda(device=device_ids[0])
            total_out2 = torch.cat((total_out2, outputs_2), 0)
            label_2 = label_2.long().cuda(device=device_ids[0])
            label_2 = torch.cat((label_2, targets_2), 0)
            total_loss_dic[str(1)] += cross_loss_dic[str(1)] * targets.size(0)
            model_output_dic[str(1)] = total_out2

    for i, n_model in enumerate(model_output_dic):
        if i == 0:
            acc_dic[str(i)] = calculate_acc(total_out1, label_1)
        elif i == 1:
            acc_dic[str(i)] = calculate_acc(total_out2, label_2)

    print('Epoch:[{0}]\t'
          'Val_aux_loss  {aux_loss:.4f}\t'
          'Val_main_loss  {main_loss:.4f}\t'
          'Val_aux_acc {aux_acc:.4f}\t'
          'Val_main_acc {main_acc:.4f}'.format(
        epoch,
        aux_loss=total_loss_dic[str(0)] / (label_1.size(0)),
        main_loss=total_loss_dic[str(1)] / (label_2.size(0)),
        aux_acc=acc_dic[str(0)],
        main_acc=acc_dic[str(1)]
    ))
    if (acc_dic[str(0)] >= Val_aux_acc_max):
        if acc_dic[str(0)] >= Val_aux_acc_max:
            Val_aux_acc_max = acc_dic[str(0)]
            save_file_path0 = os.path.join(opt.result_path,'S_C_C2_4(flow112)-fold4(H+S)-(flow-gray)-net1-TIM32-weights-improvement-TIM32-1500-{}-{:.4f}.pth'.format(epoch, Val_aux_acc_max))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'aux': model_1.module.state_dict(),

        }
        torch.save(states, save_file_path0)
    if (acc_dic[str(1)] >= Val_main_acc_max):
        if acc_dic[str(1)] >= Val_main_acc_max:
            Val_main_acc_max = acc_dic[str(1)]
            save_file_path1 = os.path.join(opt.result_path, 'S_C_C2_4(gray112)-fold4(H+S)-(flow-gray)-net2-TIM32-weights-improvement-TIM32-1500-{}-{:.4f}.pth'.format(epoch, Val_main_acc_max))

        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'main': model_2.module.state_dict(),
        }

        torch.save(states, save_file_path1)
    print('Val_aux_acc_max:', Val_aux_acc_max)
    print('Val_main_acc_max:', Val_main_acc_max)
    aux_loss = total_loss_dic[str(0)].detach().cpu().numpy() / (label_1.size(0))
    aux_loss = np.round(aux_loss, 4)
    main_loss = total_loss_dic[str(1)].detach().cpu().numpy() / (label_2.size(0))
    main_loss = np.round(main_loss, 4)
    aux_acc = acc_dic[str(0)].cpu().numpy()
    aux_acc = np.round(aux_acc, 4)
    main_acc = acc_dic[str(1)].cpu().numpy()
    main_acc = np.round(main_acc, 4)
    val_logger.log({
        'epoch': epoch,
        'aux_loss': aux_loss,
        'main_loss': main_loss,
        'aux_acc': aux_acc,
        'main_acc': main_acc,
    })

    return Val_aux_acc_max, Val_main_acc_max


def calculate_acc(outputs, targets):
    device_ids = [1, 2]
    _, pre = torch.max(outputs, dim=1)
    correct = torch.zeros(1).squeeze().cuda(device=device_ids[0])
    # pre = pre.t()
    # correct = pre.eq(targets.view(1, -1))
    correct += (pre == targets).sum().float()
    # n_correct_elems = correct.float().sum().item()
    num = targets.size(0)
    return correct / num
