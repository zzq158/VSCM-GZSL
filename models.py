from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class Semantic_Match(nn.Module):
    def __init__(self, args):
        super(Semantic_Match, self).__init__()
        self.X_dim = args.X_dim
        self.C_dim = args.C_dim
        self.vae_enc_drop = args.vae_enc_drop
        self.block_dim = args.block_dim
        self.channel = self.C_dim
        self.reduction = 16
        # 首先进行分好块
        self.blocks = nn.Sequential(
            nn.Linear(self.X_dim, self.X_dim),
            nn.Dropout(self.vae_enc_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.X_dim, self.C_dim*self.block_dim),
        )
        # 执行语义注意力模块
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.SE_attention = nn.Sequential(
            nn.Linear(self.channel, self.channel // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.channel // self.reduction, self.channel, bias=False),
            nn.Sigmoid()
        )
        # 生成属性
        self.syn_attribute = nn.Sequential(
            nn.Linear(self.block_dim,self.block_dim),
            nn.Dropout(self.vae_enc_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.block_dim, 1),
            nn.Sigmoid()
        )

    def attention(self,x):
        x = x.view(x.shape[0],self.C_dim,int(math.sqrt(self.block_dim)),int(math.sqrt(self.block_dim)))
        b,c,h,w = x.size()
        y = self.avgPool(x).view(b,c)
        y = self.SE_attention(y).view(b,c,1,1)
        z = x * y.expand_as(x)
        return z.view(x.shape[0],self.C_dim,-1)
    def forward(self, x):
        # 分块
        x_block = self.blocks(x).view(x.shape[0],self.C_dim,self.block_dim)
        # 注意力
        x_attention = self.attention(x_block)
        # 合成的属性信息
        x_attribute = self.syn_attribute(x_attention).view(x_block.shape[0],-1)
        x_final = torch.cat((x,x_attribute),1)
        return x_block,x_attention, x_final, x_attribute


class Relation(nn.Module):
    def __init__(self, args):
        super(Relation, self).__init__()
        self.block_dim = args.block_dim
        self.C_dim = args.C_dim
        self.vae_enc_drop = args.vae_enc_drop

        self.operator_x = nn.Sequential(
            nn.Linear(self.block_dim,self.block_dim),
            nn.Dropout(self.vae_enc_drop),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.connect = nn.Sequential(
            nn.Linear(self.block_dim + 1,self.block_dim + 1),
            nn.Dropout(self.vae_enc_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.block_dim + 1,1),
            nn.Softmax(dim=1)
        )
    def forward(self,x,a):
        x_1 = self.operator_x(x)
        a_1 = a.reshape(a.shape[0],a.shape[1],1)
        return self.connect(torch.cat((x_1,a_1),2)).view(x.shape[0],-1)

class Dis_net(nn.Module):
    def __init__(self, args):
        super(Dis_net, self).__init__()
        self.block_dim = args.block_dim
        self.C_dim = args.C_dim
        self.vae_enc_drop = args.vae_enc_drop
        self.fc1 = nn.Sequential(
            nn.Linear(self.block_dim,self.C_dim),
            nn.Dropout(self.vae_enc_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.C_dim,self.C_dim),
            nn.Dropout(self.vae_enc_drop),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self,x):
        return self.fc1(x)


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.z_size = args.Z_dim
        self.args = args
        self.q_z_nn_output_dim = args.q_z_nn_output_dim
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()
        self.FloatTensor = torch.FloatTensor

    def create_encoder(self):
        q_z_nn = nn.Sequential(
            nn.Linear(self.args.X_dim + self.args.C_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.Dropout(self.args.vae_enc_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, self.q_z_nn_output_dim)
        )
        q_z_mean = nn.Linear(self.q_z_nn_output_dim, self.z_size)

        q_z_var = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.z_size),
            nn.Dropout(0.2),
            nn.Softplus(),
        )
        return q_z_nn, q_z_mean, q_z_var


    def create_decoder(self):
        p_x_nn = nn.Sequential(
            nn.Linear(self.z_size + self.args.C_dim, 2048),
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.2, inplace=True)
        )
        p_x_mean = nn.Sequential(
            nn.Linear(2048, self.args.X_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return p_x_nn, p_x_mean


    def reparameterize(self, mu, var):
        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_().to(self.args.gpu)
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)
        return z

    def encode(self, x, c):
        input = torch.cat((x,c),1)
        h = self.q_z_nn(input)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)
        return mean, var

    def decode(self, z, c):
        input = torch.cat((z, c), 1)
        h = self.p_x_nn(input)
        x_mean = self.p_x_mean(h)
        return x_mean

    def forward(self, x, c, weights=None):
        z_mu, z_var = self.encode(x, c)
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z, c)
        return x_mean, z_mu, z_var, z



