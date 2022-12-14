import numpy
import torch
import json
import torch.optim as optim
import glob
import os
import random
import argparse
from time import gmtime, strftime
from models import *
from dataset_GBU import FeatDataLayer, DATA_LOADER
from utils import *
import torch.backends.cudnn as cudnn
import classifier
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB',help='dataset: CUB, AWA2, APY, FLO, SUN')
parser.add_argument('--dataroot', default='../SDGZSL/data', help='path to dataset')
# False 非直推式
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)
parser.add_argument('--gen_nepoch', type=int, default=600, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate to train generater')
parser.add_argument('--zsl', type=bool, default=False, help='Evaluate ZSL or GZSL')
parser.add_argument('--ga', type=float, default=3, help='relationNet weight')
parser.add_argument('--beta', type=float, default=0.3, help='tc weight')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
parser.add_argument('--kl_warmup', type=float, default=0.002, help='kl warm-up for VAE')
parser.add_argument('--tc_warmup', type=float, default=0.001, help='tc warm-up')
parser.add_argument('--vae_dec_drop', type=float, default=0.5, help='dropout rate in the VAE decoder')
parser.add_argument('--vae_enc_drop', type=float, default=0.4, help='dropout rate in the VAE encoder')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--classifier_steps', type=int, default=20, help='training steps of the classifier')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--nSample', type=int, default=1200, help='number features to generate per class')
parser.add_argument('--disp_interval', type=int, default=50)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval',  type=int, default=50)
parser.add_argument('--evl_start',  type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=3740, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=20, help='dimention of latent z')
parser.add_argument('--q_z_nn_output_dim', type=int, default=128, help='dimention of hidden layer in encoder')
parser.add_argument('--block_dim', type=int, default=144)

parser.add_argument('--gpu', default='2', type=str, help='index of GPU to use')
opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))
opt.gpu = torch.device("cuda:"+opt.gpu if torch.cuda.is_available() else "cpu")

def train():
    dataset = DATA_LOADER(opt)

    if opt.zsl:
        out_dir = 'results/{}/final/ZSL'.format(opt.dataset)
    else:
        out_dir = 'results/{}/train/GZSL'.format(opt.dataset)

    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))
    log_dir = out_dir + '/all_log_lr-{}_clr-{}_block-dim-{}_ga-{}.txt'.format(opt.lr, opt.classifier_lr, opt.block_dim,opt.ga)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    dataset.feature_dim = dataset.train_feature.shape[1]
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class

    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), opt)

    opt.niter = int(dataset.ntrain/opt.batchsize) * opt.gen_nepoch

    result_gzsl_soft = Result()
    result_zsl_soft = Result()

    model = VAE(opt).to(opt.gpu)
    match_model = Semantic_Match(opt).to(opt.gpu)

    relation = Relation(opt).to(opt.gpu)

    dis_net = Dis_net(opt).to(opt.gpu)
    print(model)

    with open(log_dir, 'a') as f:
        f.write('\n')
        f.write('Generative Model Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    match_model_optimizer = optim.Adam(match_model.parameters(),lr =opt.lr, weight_decay=opt.weight_decay)
    relation_optimizer = optim.Adam(relation.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    dis_net_optimizer = optim.Adam(dis_net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    mse = nn.MSELoss().to(opt.gpu)
    cross_entropy_Loss = nn.CrossEntropyLoss().to(opt.gpu)
    target = torch.from_numpy(numpy.arange(opt.C_dim).astype(int)).to(opt.gpu)

    iters = math.ceil(dataset.ntrain/opt.batchsize)
    beta = 0.01
    coin = 0
    gamma = 0
    for it in trange(start_step, opt.niter+1):
        if it % iters == 0:
            beta = min(opt.kl_warmup*(it/iters), 1)
            gamma = min(opt.tc_warmup * (it / iters), 1)

        blobs = data_layer.forward()  # 在这里拿到一个批次的数据
        feat_data = blobs['data']     # 该批次的特征
        labels_numpy = blobs['labels'].astype(int)  # 该批次的标签
        labels = torch.from_numpy(labels_numpy.astype('int')).to(opt.gpu) # 该批次的标签到gpu

        C = np.array([dataset.train_att[i,:] for i in labels])
        C = torch.from_numpy(C.astype('float32')).to(opt.gpu)
        X = torch.from_numpy(feat_data).to(opt.gpu) # 把特征和属性信息都放到gpu上

        x_mean, z_mu, z_var, z = model(X, C)
        loss_cVAE, ce, kl = multinomial_loss_function(x_mean, X, z_mu, z_var, z, beta=beta)

        x_block_mean, x_attention_mean, x_final_mean, a_final_mean = match_model(x_mean)
        x_block_real, x_attention_real, x_final_real, a_final_real = match_model(X)

        scl = opt.ga*(mse(a_final_mean,C)+mse(a_final_real,C)+mse(a_final_mean,a_final_real)).to(torch.float32)

        # 分块相独立函数的调用
        out_attention_mean = dis_net(x_attention_mean)
        out_attention_real = dis_net(x_attention_real)
        sil = 0
        for i in range(opt.batchsize):
            sil = sil + opt.ga * (cross_entropy_Loss(out_attention_mean[i], target)
                                            + cross_entropy_Loss(out_attention_real[i],target))

        # 相关函数的调用
        relation_real = relation(x_block_real,C)
        relation_mean = relation(x_block_mean,C)
        srl = opt.ga * (mse(relation_real,C) + mse(relation_mean,C))

        optimizer.zero_grad()
        match_model_optimizer.zero_grad()
        relation_optimizer.zero_grad()
        dis_net_optimizer.zero_grad()

        loss = loss_cVAE + scl + srl + sil
        loss.backward()

        dis_net_optimizer.step()
        relation_optimizer.step()
        match_model_optimizer.step()
        optimizer.step()

        if it % opt.disp_interval == 0 and it:
            log_text = '{};Iter-[{}/{}]; loss: {:.3f}; loss_cVAE:{:.3f}; scl:{:.3f};srl:{:.3f};sil:{:.3f};'.format(opt.dataset,
                                    it, opt.niter, loss.item(), loss_cVAE.item(), scl.item(), srl.item(), sil.item())
            log_print(log_text, log_dir)

        if it % opt.evl_interval == 0 and it > opt.evl_start:

            model.eval()
            match_model.eval()
            gen_feat, gen_label = synthesize_feature_match_model(model, match_model, dataset, opt)
            with torch.no_grad():
                _, _, train_feature, _ = match_model(dataset.train_feature.to(opt.gpu))
                _, _, test_unseen_feature, _ = match_model(dataset.test_unseen_feature.to(opt.gpu))
                _, _, test_seen_feature, _ = match_model(dataset.test_seen_feature.to(opt.gpu))
                train_feature = train_feature.cpu()
                test_unseen_feature = test_unseen_feature.cpu()
                test_seen_feature = test_seen_feature.cpu()
            train_X = torch.cat((train_feature, gen_feat), 0)
            train_Y = torch.cat((dataset.train_label, gen_label + dataset.ntrain_class), 0) # 为什么要加上 ntrain_class

            if opt.zsl:
                """ZSL"""
                cls = classifier.CLASSIFIER(opt, gen_feat, gen_label, dataset, test_seen_feature, test_unseen_feature,
                                            dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5, 20,
                                            opt.nSample, False)
                result_zsl_soft.update(it, cls.acc)
                log_print("ZSL Softmax:", log_dir)
                log_print("Acc {:.2f}%  Best_acc [{:.2f}% | Iter-{}]".format(
                    cls.acc, result_zsl_soft.best_acc, result_zsl_soft.best_iter), log_dir)
                if result_zsl_soft.save_model:
                    files2remove = glob.glob(out_dir + '/Best_model_ZSL_*')
                    for _i in files2remove:
                        os.remove(_i)
                    save_model_ae(it, model, match_model, opt.manualSeed, log_text,
                                  out_dir + '/Best_model_ZSL_{}_ACC_{:.2f}.tar'.format(
                                      it,result_zsl_soft.best_acc)
                                  )

            else:
                """ GZSL"""
                cls = classifier.CLASSIFIER(opt, train_X, train_Y, dataset, test_seen_feature, test_unseen_feature,
                                    dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5,
                                            opt.classifier_steps, opt.nSample, True)

                result_gzsl_soft.update_gzsl(it, cls.acc_seen, cls.acc_unseen, cls.H)

                log_print("GZSL Softmax:", log_dir)
                log_print("U->T {:.2f}%  S->T {:.2f}%  H {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
                    cls.acc_unseen, cls.acc_seen, cls.H,  result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
                    result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)

                if result_gzsl_soft.save_model:
                    files2remove = glob.glob(out_dir + '/Best_model_GZSL_*')
                    for _i in files2remove:
                        os.remove(_i)
                    save_model_ae(it, model,match_model, opt.manualSeed, log_text,
                               out_dir + '/Best_model_GZSL_{}_H_{:.2f}_S_{:.2f}_U_{:.2f}.tar'.format(it,result_gzsl_soft.best_acc,
                                                                                                 result_gzsl_soft.best_acc_S_T,
                                                                                                 result_gzsl_soft.best_acc_U_T))

            model.train()
            match_model.train()
            # ae.train()
        if it % opt.save_interval == 0 and it:
            save_model(it, model, opt.manualSeed, log_text,
                       out_dir + '/Iter_{:d}.tar'.format(it))
            print('Save model to ' + out_dir + '/Iter_{:d}.tar'.format(it))

    print('Dataset', opt.dataset)
    if opt.zsl:
        print("the best ZSL seen accuracy is",result_zsl_soft.best_acc)
    else:
        print('the best GZSL seen accuracy is',result_gzsl_soft.best_acc_S_T)
        print('the best GZSL unseen accuracy is',result_gzsl_soft.best_acc_U_T)
        print('the best GZSL H is', result_gzsl_soft.best_acc)


if __name__ == "__main__":
    train()
