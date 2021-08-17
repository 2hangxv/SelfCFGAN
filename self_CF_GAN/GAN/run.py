import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from GAN.model import Generator, Discriminator
from GAN.utils import generate_random_seed, minmaxscaler, conto1, change_date

parser = argparse.ArgumentParser(description='')
parser.add_argument('--x_size', type=int, default=1682, help='item number')
parser.add_argument('--y_size', type=int, default=1682, help='condition lenth')
parser.add_argument('--z_size', type=int, default=128, help='batch size')
parser.add_argument('--w_size', type=int, default=943, help='user numbre')
parser.add_argument('--num_neighbors', type=int, default=3, help='virtual neighbor number')
parser.add_argument('--threshold_fun', type=float, default=0.5, help='')
parser.add_argument('--threshold_sat', type=float, default=0.5, help='')
parser.add_argument('--num_epochs', default=10, type=int, help='train epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='adam:learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='')
parser.add_argument('--seed', type=int, default=2021, help='random seed')



if __name__ == '__main__':
    opt = parser.parse_args()
    X_SIZE = opt.x_size
    Y_SIZE = opt.y_size
    Z_SIZE = opt.z_size
    SEED = opt.seed
    EPOCH = opt.num_epochs

    # 设置神经网络参数随机初始化种子，使每次训练初始参数可控
    torch.cuda.manual_seed(SEED)

    # model和rating保存路径
    g_save_path = 'saved_gan_model/{}-{}-{}/Generator'.format(opt.num_neighbrors,
                                                              opt.threshold_fun,
                                                              opt.threshold_sat)
    r_save_path = 'saved_gan_model/{}-{}-{}/Rating'.format(opt.num_neighbrors,
                                                           opt.threshold_fun,
                                                           opt.threshold_sat)
    os.makedirs(g_save_path, exist_ok=True)
    os.makedirs(r_save_path, exist_ok=True)

    # rating矩阵文件存放路径
    file_path = './data'

    # 读取数据
    fun_matrix = np.load('{}/fun_matirx.npy'.format(file_path))
    sat_matrix = np.load('{}/sat_matrix.npy'.format(file_path))
    raw_rating = np.load('{}/raw_rating.npy'.format(file_path))
    raw_fun = np.load('{}/raw_fun.npy'.format(file_path))
    real_rating = raw_rating.copy()
    raw_rating[np.isnan(raw_rating)] = 0
    raw_fun[np.isnan(raw_fun)] = 0

    # 实例化模型类
    fun_netG = Generator(input_size=(Z_SIZE, Y_SIZE), output_size=X_SIZE)
    fun_netD = Discriminator(input_size=(X_SIZE, Y_SIZE))
    sat_netG = Generator(input_size=(Z_SIZE, Y_SIZE), output_size=X_SIZE)
    sat_netD = Discriminator(input_size=(X_SIZE, Y_SIZE))

    # 定义损失函数类型和Optimizer
    criterion = nn.BCELoss()
    optim_fG = torch.optim.SGD(fun_netG.parameters(), lr=opt.lr, momentum=opt.momentum)
    optim_fD = torch.optim.SGD(fun_netD.parameters(), lr=opt.lr, momentum=opt.momentum)
    optim_sG = torch.optim.SGD(sat_netG.parameters(), lr=opt.lr, momentum=opt.momentum)
    optim_sD = torch.optim.SGD(sat_netD.parameters(), lr=opt.lr, momentum=opt.momentum)


    # 训练
    for epoch in range(1, EPOCH+1):
        print('epoch = {}'.format(epoch))
        running_loss_s = []
        running_loss_d = []
        running_loss_g = []
        for i in range(fun_matrix.shape[0]):
            # 取出兴趣矩阵和满意度矩阵的一行
            fv = torch.from_numpy(fun_matrix[i].astype(np.float32))
            sv = torch.from_numpy(fun_matrix[i].astype(np.float32))

            # 训练判别器
            fun_real_out = fun_netD(fv, fv).squeeze()
            sat_real_out = sat_netD(sv, sv).squeeze()

            fun_fake = fun_netG(generate_random_seed(128), fv).squeeze()
            sat_fake = sat_netG(generate_random_seed(128), sv).squeeze()
            fsmulf = torch.Tensor.mul(sat_fake.detach(), fv)

            fun_fake_out = fun_netD(fun_fake.detach(), fv).squeeze()
            sat_fake_out = sat_netD(fsmulf, sv).squeeze()

            dloss1 = criterion(fun_real_out, torch.FloatTensor([1]))
            dloss2 = criterion(fun_fake_out, torch.FloatTensor([0]))
            dloss3 = criterion(sat_real_out, torch.FloatTensor([1]))
            dloss4 = criterion(sat_fake_out, torch.FloatTensor([0]))
            dloss_total = dloss1 + dloss2 + dloss3 + dloss4

            optim_fD.zero_grad()
            optim_sD.zero_grad()
            dloss_total.backward()
            optim_fD.step()
            optim_sD.step()

            # 训练生成器
            fun_fake = fun_netG(generate_random_seed(128), fv).squeeze()
            sat_fake = sat_netG(generate_random_seed(128), sv).squeeze()
            fsmulf = torch.Tensor.mul(sat_fake, fv)

            fun_fake_out = fun_netD(fun_fake, fv).squeeze()
            sat_fake_out = sat_netD(fsmulf, sv).squeeze()

            gloss1 = criterion(fun_fake_out, torch.FloatTensor([1]))  # 为了让生成器生成更逼真的向量，因此相比判别器loss只有关于1的
            gloss2 = criterion(sat_fake_out, torch.FloatTensor([1]))
            gloss_total = gloss1 + gloss2

            optim_fG.zero_grad()
            optim_sG.zero_grad()
            gloss_total.backward()
            optim_fG.step()
            optim_sG.step()

            # 统计损失函数
            if (i % 100 == 0 and i != 0):
                print(" Line {}".format(i))
                print(" s_loss: ", dloss_total)
                print(" d_loss: ", gloss1)
                print(" g_loss: ", gloss2)

            running_loss_s.append(dloss_total.item())
            running_loss_d.append(gloss1.item())
            running_loss_g.append(gloss2.item())

        # 保存训练数据
        torch.save(fun_netG.state_dict(), g_save_path + '/fun/final_model_{}'.format(epoch))
        torch.save(sat_netG.state_dict(), g_save_path + '/sat/final_model_{}'.format(epoch))
        def saveRating(epoch):
            epoch_rating = real_rating.copy()
            for i in range(opt.w_size):
                fv = fun_matrix[i].astype(np.float32)
                fv = torch.from_numpy(fv)
                sv = sat_matrix[i].astype(np.float32)
                sv = torch.from_numpy(sv)
                fun_vs = []
                sat_vs = []
                for j in range(opt.num_neighbors):  # neighbor_num
                    fun_vs.append(fun_netG(generate_random_seed(128), fv).detach().numpy())
                    sat_vs.append(sat_netG(generate_random_seed(128), sv).detach().numpy())
                f1 = conto1(fun_vs, raw_fun[i])
                s1 = conto1(sat_vs, minmaxscaler(raw_rating[i]))
                for j in range(opt.x_size):
                    if (f1[0][j] < opt.threshold_fun and s1[0][j] < opt.threshold_sat and np.isnan(epoch_rating[i][j])):
                        epoch_rating[i][j] = 0
            change_date(epoch_rating, epoch, r_save_path)
        saveRating(epoch)
