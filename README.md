# 基于双流网络信息交互的微表情识别

（1）	训练
Main.py（学习率、优化器、训练集、验证集以及记录参数等在此设置）

当各项参数设置完毕之后，进入到pytorch环境
进入到…/Dual_resnet_pytorch_unbantu/ 这个路径下，输入python main.py开始训练
train_2.py&val_1.py
分别为训练代码和验证代码
在train_2.py文件中可以通过设置total_loss_dic[str(i)]来进行消融实验。
val_1.py文件中可以设置模型参数保存策略，（本实验保存结果最好的模型参数），还可以保存文件路径、文件名。
resnet.py
文件为3D-Resnet网络结构，num_class参数设置微表情的类别数
resnext.py
文件为3D-Resnext网络结构，num_class参数设置微表情的类别数
opts.py
此文件中设置默认的保存文件路径，num_class、n_epochs、sample_size（输入序列空间尺寸）、sample_duration（时间尺寸）
（2）	测试
test_new.py
为测试文件，用于测试模型性能，可在其中设置，需要加载的模型参数以及需要测试的测试集。


