import torch
from torch.autograd import Variable
import time
import sys
import os
from torch import nn
import torch.nn.functional as F
import numpy as np
from opts import parse_opts
from utils import AverageMeter, calculate_accuracy
from model import generate_model
import torch.utils.data as Data


def calculate_acc(outputs, targets):
    device_ids = [1, 2]
    _, pre = torch.max(outputs, dim=1)
    correct = torch.zeros(1).squeeze()
    # pre = pre.t()
    # correct = pre.eq(targets.view(1, -1))
    correct += (pre == targets).sum().float()
    # n_correct_elems = correct.float().sum().item()
    num = targets.size(0)
    return correct / num


def test():
    print('Start test!!!')
    device_ids = [1, 2]
    opt = parse_opts()
    # if opt.root_path != '':
    #     opt.result_path = os.path.join(opt.root_path, opt.result_path)
    #     if opt.resume_path:
    #         opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
    #     if opt.pretrain_path:
    #         opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    # opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    model_1, parameters_1 = generate_model(opt=opt)

    checkpoint = torch.load(
        '/mnt/Dual_resnet_pytorch_unbantu/S_C_C2_4_new/new/S_C_C2_4(gray112)-fold1(H+S)-(flow-gray)-net2-TIM32-weights-improvement-TIM32-1500-241-0.5391.pth')

    for k, v in checkpoint.items():
        print(k)
    a = checkpoint['arch']
    print(a)
    b = checkpoint['main']
    # for k,v in b.items():
    #   print(k)
    print('*****************************************')
    model_1.load_state_dict(b)

    # model_2, parameters_2 = generate_model(opt=opt)
    model_1 = nn.DataParallel(model_1, device_ids=device_ids).cuda(device=device_ids[0]).eval()
    print('model_1:', model_1)
    model_output_dic = {}
    total_model_dic = {}
    acc_dic = {}
    label_1 = torch.from_numpy(np.array([]))
    # label_2 = torch.from_numpy(np.array([]))
    total_out1 = torch.from_numpy(np.array([]))
    # total_out2 = torch.from_numpy(np.array([]))
    # total_model_dic[str(1)] = model_2

    # for i in range(1):

    model_output_dic[0] = 0

    # model_1 = nn.DataParallel(model_1, device_ids=device_ids).cuda(device=device_ids[0]).eval()
    # model_2 = nn.DataParallel(model_2, device_ids=device_ids).cuda(device=device_ids[0]).eval()
    # for model in total_model_dic:
    # total_model_dic[model].eval()
    # total_model_dic[model] = nn.DataParallel(total_model_dic[model], device_ids=device_ids).cuda(device=device_ids[0]).eval()

    validation_data_1 = np.load('./S_C_C2_4_new/fold1/S_C_C2_4(gray112)_fold1_TIM32_test.npy')
    validation_labels_1 = np.load('./S_C_C2_4_new/fold1/S_C_C2_4(gray112)_fold1_TIM32_test_labels.npy')

    validation_data_1 = torch.from_numpy(validation_data_1)
    validation_labels_1 = torch.from_numpy(validation_labels_1)
    validation_data_1 = Data.TensorDataset(validation_data_1, validation_labels_1)

    val_loader_1 = torch.utils.data.DataLoader(
        validation_data_1,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False)

    with torch.no_grad():
        for k, (inputs, targets) in enumerate(val_loader_1):
            targets = targets.long()
            inputs = Variable(inputs.cuda(device=device_ids[0]))
            target_1 = Variable(targets.cuda(device=device_ids[0]))
            outputs_1 = model_1(inputs)
            _, targets_1 = torch.max(target_1, 1)

            total_out1 = total_out1.clone().detach().float().cuda(device=device_ids[0])
            total_out1 = torch.cat((total_out1, outputs_1), 0)
            label_1 = label_1.long().cuda(device=device_ids[0])
            label_1 = torch.cat((label_1, targets_1), 0)

            model_output_dic[str(0)] = total_out1

    # with torch.no_grad():
    #     for k, (inputs, targets) in enumerate(data_loader_2):
    #         targets = targets.long()
    #         inputs = Variable(inputs)
    #         target_2 = Variable(targets)
    #         outputs_2 = total_model_dic[str(1)](inputs)
    #         _, targets_2 = torch.max(target_2, 1)
    #
    #         total_out2 = total_out2.clone().detach().float()
    #         total_out2 = torch.cat((total_out2, outputs_2), 0)
    #         label_2 = label_2.long()
    #         label_2 = torch.cat((label_2, targets_2), 0)
    #
    #         model_output_dic[str(1)] = total_out2

    acc_dic[str(0)] = calculate_acc(model_output_dic[str(0)], label_1)
    # for i, n_model in enumerate(model_output_dic):
    #     if i == 0:
    #
    #     elif i == 1:
    #         acc_dic[str(i)] = calculate_acc(total_out2, label_2)

    print('test_model1_acc {aux_acc:.4f}'.format(aux_acc=acc_dic[str(0)]))

    # aux_acc = acc_dic[str(0)].cpu().numpy()
    # aux_acc = np.round(aux_acc, 4)
    # main_acc = acc_dic[str(1)].cpu().numpy()
    # main_acc = np.round(main_acc, 4)


test()



